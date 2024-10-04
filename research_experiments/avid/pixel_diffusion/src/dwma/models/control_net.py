"""3D unet implementation from: https://github.com/lucidrains/video-diffusion-pytorch/blob/main/video_diffusion_pytorch/video_diffusion_pytorch.py"""

from functools import partial

import torch
from avid_utils.helpers import default, is_odd, prob_mask_like
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from dwma.models.video_diffusion_pytorch.blocks import PreNorm, Residual, SinusoidalPosEmb
from dwma.models.video_diffusion_pytorch.unet3d import (
    Attention,
    Downsample,
    EinopsToAndFrom,
    RelativePositionBias,
    ResnetBlock3d,
    SpatialLinearAttention,
    Unet3D,
    temporal_att,
)

# attention along space and time


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ControlNet(nn.Module):
    def __init__(
        self,
        dim=32,
        train_with_cfg=False,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        condition_frames=0,
        generation_frames=1,
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
        self.zero_linear_action_mlp = zero_module(nn.Linear(act_emb_dim, act_emb_dim))

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.zero_convs = nn.ModuleList([])

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
            self.zero_convs.append(zero_module(nn.Conv3d(dim_out, dim_out, 1, padding=0)))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom("b c f h w", "b f (h w) c", Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_att(mid_dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
        )
        self.mid_block_1_out = zero_module(nn.Conv3d(mid_dim, mid_dim, 1, padding=0))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
        self.mid_block_2_out = zero_module(nn.Conv3d(mid_dim, mid_dim, 1, padding=0))

    def forward(self, x, time, cond_frames=None, act=None, cond_drop_prob=None):
        """Forward pass through unet3d downsampling and middle blocks.

        Args:
            x: input tensor of shape (b, c, t, h, w)
            time: time tensor of shape (b, 1)
            act: action tensor of shape (b, t)
            cond_frames: tensor of shape (b, c, num_cond, h, w) containing the condition frames
        """
        if self.condition_frames > 0:
            assert cond_frames.shape[2] == self.condition_frames, "expected number of condition frames to match model"

        b, _, t, _, _ = x.shape

        if cond_frames is not None:
            cond_frames = rearrange(cond_frames, "b c num_cond h w -> b (c num_cond) h w")
            cond_frames = cond_frames.unsqueeze(2).repeat(1, 1, t, 1, 1)  # repeat conditioning at each time step
            x = torch.cat((cond_frames, x), dim=1)  # concatenate conditioning frames and input along channels

        time_rel_pos_bias = self.time_rel_pos_bias(t, device=x.device)
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        time_emb = self.time_mlp(time)  # b, c
        if act is not None:
            act_emb = self.action_emb(act)  # b, t, c
        else:
            act_emb = self.null_action_emb.unsqueeze(0).unsqueeze(0).repeat(b, t, 1)

        if cond_drop_prob and cond_drop_prob > 0:
            keep_mask = prob_mask_like((act.shape[0],), 1 - cond_drop_prob, device=act.device)

            act_emb = torch.where(rearrange(keep_mask, "b -> b 1 1"), act_emb, self.null_action_emb)
        act_emb = self.action_mlp(act_emb)  # b, t, c
        act_emb = self.zero_linear_action_mlp(act_emb)

        outs = []

        for (block1, block2, spatial_attn, temporal_attn, downsample), zero_conv in zip(self.downs, self.zero_convs):
            x = block1(x, time_emb, act_emb)
            x = block2(x, time_emb, act_emb)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            outs.append(zero_conv(x))
            x = downsample(x)

        x = self.mid_block1(x, time_emb, act_emb)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        outs.append(self.mid_block_1_out(x))
        x = self.mid_block2(x, time_emb, act_emb)
        outs.append(self.mid_block_2_out(x))

        return outs


class ControlledUnetModel(Unet3D):
    def forward(self, x, time, cond_frames=None, act=None, cond_drop_prob=None, control=None, only_mid_control=False):
        """Forward pass through unet3d with additional control output from ControlNet.

        Args:
            x: input tensor of shape (b, c, t, h, w)
            time: time tensor of shape (b, 1)
            act: action tensor of shape (b, t)
            cond_frames: tensor of shape (b, c, num_cond, h, w) containing the condition frames
            control: list of tensors containing the control outputs from ControlNet
            only_mid_control: if True, only use the control output from the middle block
        """
        if self.condition_frames > 0:
            assert cond_frames.shape[2] == self.condition_frames, "expected number of condition frames to match model"

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
        if control is not None:
            x += control.pop()
        x = self.mid_block2(x, time_emb, act_emb)
        if control is not None:
            x += control.pop()

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            if only_mid_control or control is None:
                x = torch.cat((x, h.pop()), dim=1)
            else:
                x = torch.cat((x, h.pop() + control.pop()), dim=1)
            x = block1(x, time_emb, act_emb)
            x = block2(x, time_emb, act_emb)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)


class ControlNetSmaller(ControlNet):
    def __init__(
        self,
        dim=32,
        new_dim=32,
        train_with_cfg=False,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        condition_frames=0,
        generation_frames=1,
        attn_heads=8,
        attn_dim_head=32,
        num_actions=15,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
    ):
        super().__init__(
            new_dim,  # changed from dim to new_dim
            train_with_cfg,
            dim_mults,
            channels,
            condition_frames,
            generation_frames,
            attn_heads,
            attn_dim_head,
            num_actions,
            init_dim,
            init_kernel_size,
            use_sparse_linear_attn,
        )

        # dimensions
        dims = [*map(lambda m: dim * m, dim_mults)]
        new_dims = [*map(lambda m: new_dim * m, dim_mults)]

        self.projection_layers = nn.ModuleList([])
        for new, old in zip(new_dims, dims):
            self.projection_layers.append(nn.Linear(new, old))
        self.projection_layers.append(nn.Linear(new_dims[-1], dims[-1]))
        self.projection_layers.append(nn.Linear(new_dims[-1], dims[-1]))

    def forward(self, x, time, cond_frames=None, act=None, cond_drop_prob=None):
        outs = super().forward(x, time, cond_frames, act, cond_drop_prob)
        new_outs = []
        for i, out in enumerate(outs):
            out_rearrange = rearrange(out, "b c t h w -> (b t h w) c")
            out_rearrange_output = self.projection_layers[i](out_rearrange)
            out_rearrange_output = rearrange(
                out_rearrange_output,
                "(b t h w) c -> b c t h w",
                b=out.shape[0],
                t=out.shape[2],
                h=out.shape[3],
                w=out.shape[4],
            )
            new_outs.append(out_rearrange_output)
        return new_outs
