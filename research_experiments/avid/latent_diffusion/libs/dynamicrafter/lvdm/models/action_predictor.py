import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from avid_utils.helpers import exists
from avid_utils.image import preprocess_images
from einops import rearrange
from lvdm.common import extract_into_tensor
from lvdm.utils.utils import instantiate_from_config
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


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


class ActionPredictor(pl.LightningModule):
    def __init__(
        self,
        encoder_config,
        in_channels=3,
        logdir=None,
        in_block_channels=(),
        mlp_hidden_dims=(512, 512),
        input_resolution=(320, 512),
        encode_and_add_noise=False,  # whether to encode and add noise to input data
        input_key="video",
        target_key="act",
        discretize_actions=False,
        action_dims=7,
        discrete_action_bins=256,
        action_range=(-4, 4),  # as octo data normalized to mean 0 and std 1
        linear_warmup_steps=250,
        **kwargs,
    ):
        super().__init__()

        self.logdir = logdir
        self.discretize_actions = discretize_actions
        self.action_dims = action_dims
        self.discrete_action_bins = discrete_action_bins
        self.encoder_config = encoder_config
        self.input_key = input_key
        self.target_key = target_key
        self.linear_warmup_steps = linear_warmup_steps
        self.action_range = action_range
        self.encode_and_add_noise = encode_and_add_noise

        # in blocks to downsample resolution before passing to 3D unet encoder (which does attention so needs smaller resolution)
        current_in_channels = in_channels
        layers = []
        for out_channels in in_block_channels:
            layers.append(ResnetBlock3d(current_in_channels, out_channels))
            layers.append(Downsample(out_channels))
            current_in_channels = out_channels
        self.in_blocks = nn.Sequential(*layers)

        # set the input channels to the 3D unet to the output of the last downsample layer
        encoder_config["params"]["channels"] = current_in_channels
        self.encoder = instantiate_from_config(encoder_config)

        # raise not implemented error if discretize_actions is True
        if not discretize_actions:
            mlp_out = action_dims
            self.loss = nn.MSELoss()
            self.mae = nn.L1Loss()
        else:
            mlp_out = action_dims * discrete_action_bins
            self.accuracy = Accuracy("multiclass", num_classes=self.discrete_action_bins)
            self.loss = nn.CrossEntropyLoss()

        # compute the number of input features for the mlp
        num_downs = len(self.encoder.downs) + len(in_block_channels)
        enc_dim = encoder_config.params.dim * encoder_config.params.dim_mults[-1]
        mlp_in_dim = input_resolution[0] // (2**num_downs) * input_resolution[1] // (2**num_downs) * enc_dim

        # create the MLP
        layers = []
        layers.append(nn.Linear(mlp_in_dim, mlp_hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(mlp_hidden_dims) - 1):
            layers.append(nn.Linear(mlp_hidden_dims[i], mlp_hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(mlp_hidden_dims[-1], mlp_out))
        self.mlp = nn.Sequential(*layers)
        self.save_hyperparameters()

    def set_diffusion_model(self, diffusion_model):
        self.diffusion_model = diffusion_model
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

    def on_save_checkpoint(self, checkpoint):
        # Create a new dict with all keys that don't start with "diffusion_model" as we only want to save classifier
        new_state_dict = {
            key: value for key, value in checkpoint["state_dict"].items() if not key.startswith("diffusion_model")
        }
        checkpoint["state_dict"] = new_state_dict

    def forward(self, x, time=None):
        x = self.in_blocks(x)
        x_init = x[:, :, 0:1]
        x = x[:, :, 1:]
        z = self.encoder(x=x, x_init=x_init, time=time)
        z = rearrange(z, "b c t h w -> b t (h w c)")
        return self.mlp(z)

    def step(self, batch, batch_idx):
        x = batch[self.input_key]
        metrics = {}

        if self.encode_and_add_noise:
            assert hasattr(self, "diffusion_model"), "Diffusion model not set for encoding and adding noise."
            x = self.diffusion_model.encode_first_stage(x)  # pass through encoder
            t = torch.randint(0, self.diffusion_model.num_timesteps, (x.shape[0],), device=self.device).long()
            if self.diffusion_model.use_dynamic_rescale:
                x = x * extract_into_tensor(self.diffusion_model.scale_arr, t, x.shape)
            noise = torch.randn_like(x)
            x = self.diffusion_model.q_sample(x, t, noise)  # add noise according to diffusion schedule
        else:
            t = None

        if not self.discretize_actions:
            preds = self(x, t)
            targs = batch[self.target_key]
            targs = rearrange(targs, "b t a -> b (t a)")
            preds = rearrange(preds, "b t a -> b (t a)")
        else:
            logits = self(x, t)
            targs = self.to_discrete_actions(batch[self.target_key])
            preds = rearrange(logits, "b t (a d) -> (b t a) d", d=self.discrete_action_bins)
            targs = rearrange(targs, "b t a -> (b t a)")

            max_preds = torch.argmax(preds, dim=-1)
            acc = self.accuracy(max_preds, targs)
            metrics["acc"] = acc.item()

        loss = self.loss(preds, targs)
        return loss, metrics

    def to_discrete_actions(self, actions):
        """Takes action tensor (b, t, a) returns a tensor of discretized actions of the same size."""
        assert actions.ndim == 3

        actions = torch.clamp(actions, self.action_range[0], self.action_range[1])
        actions = (actions - self.action_range[0]) / (self.action_range[1] - self.action_range[0])  # norm to [0, 1]
        actions = (actions * (self.discrete_action_bins - 1)).long()
        return actions

    def training_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        self.log("train/loss", loss)
        for k, v in metrics.items():
            self.log(f"train/{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        for k, v in metrics.items():
            self.log(f"val/{k}", v)
        self.log("val/loss", loss)

    def get_prediction_error(self, _, frames, act):
        """Compute the MAE prediction error of the model."""
        assert not self.discretize_actions, "Prediction error not defined for discretized actions"

        if frames.dtype == torch.uint8:
            frames = preprocess_images(frames)
        preds = self(frames)

        targs = rearrange(act, "b t a -> b (t a)")
        preds = rearrange(preds, "b t a -> b (t a)")

        return self.mae(preds, targs).item()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        # linear warmup multiplier
        def lr_lambda(step: int) -> float:
            return min(1.0, step / self.linear_warmup_steps)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",  # update the learning rate every step
        }
        return [optimizer], [scheduler]
