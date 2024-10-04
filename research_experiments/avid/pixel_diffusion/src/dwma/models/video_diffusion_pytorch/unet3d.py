import math
from functools import partial

import torch
import torch.nn.functional as F
from avid_utils.helpers import default, exists, is_odd, prob_mask_like
from einops import rearrange
from einops_exts import rearrange_many
from rotary_embedding_torch import RotaryEmbedding
from torch import einsum, nn

from dwma.models.video_diffusion_pytorch.blocks import PreNorm, Residual, SinusoidalPosEmb


class RMSNorm3d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class Block3d(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = RMSNorm3d(dim_out)
        self.act = nn.SiLU()
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x


class ResnetBlock3d(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, act_emb_dim=None, dropout=0.0):
        super().__init__()
        emb_dim = (int(time_emb_dim) if exists(time_emb_dim) else 0) + (int(act_emb_dim) if exists(act_emb_dim) else 0)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim_out * 2)) if emb_dim > 0 else None
        self.block1 = Block3d(dim, dim_out, dropout=dropout)
        self.block2 = Block3d(dim_out, dim_out, dropout=dropout)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, act_emb=None):
        """
        x: input tensor of shape (b, c, t, h, w)
        time_emb: time embedding tensor of shape (b, emb_dim)
        act_emb: action embedding tensor of shape (b, t, emb_dim)
        """

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(act_emb)):
            t = x.shape[2]

            # this is the embedding of diffusion time step which is same for all frames
            if exists(time_emb):
                time_emb = time_emb.unsqueeze(1).repeat(1, t, 1)  # b, t, c

            # concat embedding according to what is provided
            if exists(time_emb) and exists(act_emb):
                cond_emb = torch.cat([time_emb, act_emb], dim=-1)  # b, t, c+c
            elif exists(time_emb):
                cond_emb = time_emb
            elif exists(act_emb):
                cond_emb = act_emb

            # process embedding via mlp
            cond_emb = self.mlp(cond_emb)  # pylint: disable=E1102
            cond_emb = rearrange(cond_emb, "b t c -> b c t 1 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class RelativePositionBias(nn.Module):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


# building block modules


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, _, _, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, "b (h c) x y -> b h c (x y)", h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, "(b f) c h w -> b c f h w", b=b)


# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(" "), shape)))
        x = rearrange(x, f"{self.from_einops} -> {self.to_einops}")
        x = self.fn(x, **kwargs)
        x = rearrange(x, f"{self.to_einops} -> {self.from_einops}", **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, rotary_emb=None):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, pos_bias=None, focus_present_mask=None):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, "... n (h d) -> ... h n d", h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum("... h i d, ... h j d -> ... h i j", q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():  # pylint: disable=E1130
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, "b -> b 1 1 1 1"),
                rearrange(attend_self_mask, "i j -> 1 1 1 i j"),
                rearrange(attend_all_mask, "i j -> 1 1 1 i j"),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("... h i j, ... h j d -> ... h i d", attn, v)
        out = rearrange(out, "... h n d -> ... n (h d)")
        return self.to_out(out)


def temporal_att(dim, heads, dim_head, rotary_emb):
    return EinopsToAndFrom(
        "b c f h w", "b (h w) f c", Attention(dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb)
    )


# model


class Unet3D(nn.Module):
    def __init__(
        self,
        dim=32,
        train_with_cfg=False,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        condition_frames=2,
        generation_frames=1,
        output_mask=False,
        attn_heads=8,
        attn_dim_head=32,
        num_actions=15,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
    ):
        super().__init__()
        self.channels = channels
        self.condition_frames = condition_frames
        input_channels = channels * (1 + condition_frames)
        self.generation_frames = generation_frames

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)

        # initial conv
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            input_channels, init_dim, (1, init_kernel_size, init_kernel_size), padding=(0, init_padding, init_padding)
        )
        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_att(init_dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
        )

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.train_with_cfg = train_with_cfg
        if self.train_with_cfg:
            self.cond_drop_prob = 0.2
        else:
            self.cond_drop_prob = 0.0

        # time conditioning
        time_emb_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, time_emb_dim), nn.GELU(), nn.Linear(time_emb_dim, time_emb_dim)
        )
        act_emb_dim = time_emb_dim
        self.action_emb = nn.Embedding(num_actions, dim)
        self.action_mlp = nn.Sequential(nn.Linear(dim, act_emb_dim), nn.GELU(), nn.Linear(act_emb_dim, act_emb_dim))
        self.null_action_emb = nn.Parameter(torch.randn(dim))

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = ResnetBlock3d
        block_klass_cond = partial(block_klass, time_emb_dim=time_emb_dim, act_emb_dim=act_emb_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass_cond(dim_in, dim_out),
                        block_klass_cond(dim_out, dim_out),
                        (
                            Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads=attn_heads)))
                            if use_sparse_linear_attn
                            else nn.Identity()
                        ),
                        Residual(
                            PreNorm(
                                dim_out,
                                temporal_att(dim_out, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb),
                            )
                        ),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom("b c f h w", "b f (h w) c", Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_att(mid_dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
        )

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass_cond(dim_out * 2, dim_in),
                        block_klass_cond(dim_in, dim_in),
                        (
                            Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads=attn_heads)))
                            if use_sparse_linear_attn
                            else nn.Identity()
                        ),
                        Residual(
                            PreNorm(
                                dim_in,
                                temporal_att(dim_in, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb),
                            )
                        ),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.out_dim = channels
        if output_mask:
            self.out_dim += 1
        self.final_conv = nn.Sequential(block_klass(dim * 2, dim), nn.Conv3d(dim, self.out_dim, 1))

    def forward(self, x, time, cond_frames=None, act=None, cond_drop_prob=None):
        """Forward pass through unet.

        Args:
            x: input tensor of shape (b, c, t, h, w)
            time: time tensor of shape (b, 1)
            act: action tensor of shape (b, t)
            cond_frames: tensor of shape (b, c, num_cond, h, w) containing the condition frames
        """
        if self.condition_frames > 0:
            n_cond_frames = cond_frames.shape[2]
            n_cond_frames += x.shape[1] // self.channels - 1  # extra cond frames passed as extra input channels
            assert n_cond_frames == self.condition_frames, "expected number of condition frames to match model"

        b, _, t, _, _ = x.shape

        if cond_frames is not None:
            cond_frames = rearrange(cond_frames, "b c num_cond h w -> b (c num_cond) h w")
            cond_frames = cond_frames.unsqueeze(2).repeat(1, 1, t, 1, 1)  # repeat conditioning at each time step
            x = torch.cat((cond_frames, x), dim=1)  # concatenate conditioning frames and input along channels

        time_rel_pos_bias = self.time_rel_pos_bias(t, device=x.device)
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        r = x.clone()

        time_emb = self.time_mlp(time)  # b, c
        if act is not None:
            act_emb = self.action_emb(act)  # b, t, c
        else:
            act_emb = self.null_action_emb.unsqueeze(0).unsqueeze(0).repeat(b, t, 1)

        if cond_drop_prob and cond_drop_prob > 0:
            keep_mask = prob_mask_like((act.shape[0],), 1 - cond_drop_prob, device=act.device)

            act_emb = torch.where(rearrange(keep_mask, "b -> b 1 1"), act_emb, self.null_action_emb)
        act_emb = self.action_mlp(act_emb)  # b, t, c

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, time_emb, act_emb)
            x = block2(x, time_emb, act_emb)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb, act_emb)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        x = self.mid_block2(x, time_emb, act_emb)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, time_emb, act_emb)
            x = block2(x, time_emb, act_emb)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)
