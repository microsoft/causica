"""3D unet implementation from: https://github.com/lucidrains/video-diffusion-pytorch/blob/main/video_diffusion_pytorch/video_diffusion_pytorch.py"""

from functools import partial

import torch
from avid_utils.helpers import extract, is_odd
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from dwma.models.video_diffusion_pytorch.blocks import PreNorm, Residual, SinusoidalPosEmb
from dwma.models.video_diffusion_pytorch.diffusion import get_beta_schedule_fn
from dwma.models.video_diffusion_pytorch.unet3d import (
    Attention,
    Downsample,
    EinopsToAndFrom,
    RelativePositionBias,
    ResnetBlock3d,
    SpatialLinearAttention,
    temporal_att,
)

# model


class Unet3DEncoder(nn.Module):
    """Class to classify actions given sequences of frames."""

    def __init__(
        self,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        image_size=64,
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        mlp_dim=1024,
        num_actions=15,
        init_kernel_size=7,
        encoder_only=False,
        use_sparse_linear_attn=True,
        dropout=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_actions = num_actions

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)

        # initial conv
        init_dim = dim
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            self.channels, init_dim, (1, init_kernel_size, init_kernel_size), padding=(0, init_padding, init_padding)
        )
        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_att(init_dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
        )

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning
        time_emb_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, time_emb_dim), nn.GELU(), nn.Linear(time_emb_dim, time_emb_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # block type
        block_class = ResnetBlock3d
        block_class_cond = partial(block_class, time_emb_dim=time_emb_dim, dropout=dropout)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_class_cond(dim_in, dim_out),
                        block_class_cond(dim_out, dim_out),
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
        self.mid_block1 = block_class_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom("b c f h w", "b f (h w) c", Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_att(mid_dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
        )
        self.mid_block2 = block_class_cond(mid_dim, mid_dim)

        # MLP for processing the final feature map
        mlp_in_dim = image_size // (2**num_resolutions) * image_size // (2**num_resolutions) * mid_dim
        self.final_down = Downsample(mid_dim)

        self.encoder_only = encoder_only
        if not self.encoder_only:
            self.out_mlp = nn.Sequential(
                nn.LayerNorm(mlp_in_dim),
                nn.Linear(mlp_in_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, num_actions),
            )

    def forward(self, x, x_init, time=None):
        """Forward pass through classifier.

        Args:
            x: tensor of frames of shape (b, c, t, h, w) corresponding to: x_1, ..., x_t
            x_init: tensor of initial frames of shape (b, c, f, h, w) corresponding to: x_(-f+1), ..., x_0
            time: diffusion time step tensor of shape (b, 1)

        Returns:
            tensor: logits of shape (b, t, num_actions) corresponding to: a_0, ..., a_(t-1)
        """
        if time is None:
            time = torch.zeros((x.shape[0],), device=x.device).long()
        init_frames = x_init.shape[2]
        x = torch.cat((x_init, x), dim=2)
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        time_emb = self.time_mlp(time)
        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        x = self.mid_block2(x, time_emb)

        x = self.final_down(x)
        if getattr(self, "encoder_only", False):
            return x

        x = rearrange(x, "b c t h w -> b t (h w c)")
        x = x[:, init_frames:]  # remove the initial frames
        x = self.out_mlp(x)
        return x


class GaussianNoiseActionClassifier(nn.Module):
    """Class to train action classifier with noise added to frames."""

    def __init__(self, model: Unet3DEncoder, num_timesteps=1000, beta_schedule="sigmoid", schedule_fn_kwargs=None):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule

        # define schedule for adding noise to data
        beta_schedule_fn = get_beta_schedule_fn(beta_schedule)
        if schedule_fn_kwargs is None:
            schedule_fn_kwargs = {}
        betas = beta_schedule_fn(num_timesteps, **schedule_fn_kwargs)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(torch.float32))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).to(torch.float32))

    def q_sample(self, x_start, t):
        """Compute noisy sample from forward process."""
        noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, x, x_init, add_noise=True):
        """Given clean x and x_init add noise to x and pass through the classifier.

        Noise is added to the x frames but not the x_init initial conditioning frames before passing through the
        classifier.

        Args:
            x: tensor of frames of shape (b, c, t, h, w) corresponding to: x_1, ..., x_t
            x_init: tensor of initial frames of shape (b, c, f, h, w) corresponding to: x_(-f+1), ..., x_0
            add_noise: whether to add noise to the x frames before passing through the classifier

        Returns:
            tensor: logits of shape (b, t, num_actions) corresponding to: a_0, ..., a_(t-1)
        """
        if add_noise:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
            x = self.q_sample(x_start=x, t=t)
        else:
            t = torch.zeros((x.shape[0],), device=x.device).long()
        return self.model(x, x_init, t)

    def model_predictions(self, x, x_init, t):
        """Given noisy x, clean x_init, and timesteps of the noise schedule t, predict the actions."""
        return self.model(x, x_init, t)
