from contextlib import contextmanager

import torch
import torch.nn.functional as F
from lvdm.ema import LitEma
from lvdm.models.ddpm3d import DiffusionWrapper, LatentVisualDiffusion


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class AVIDAdapter(LatentVisualDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_cond_model = None
        self.condition_adapter_on_base_outputs = False
        self.learnt_mask = False
        self.init_mask_bias = None
        self.pretrain_steps = 0

    def on_save_checkpoint(self, checkpoint):
        """Only save the action-conditioned unet in the checkpoint as no need to resave the base model."""

        if "state_dict" in checkpoint:
            del checkpoint["state_dict"]

        checkpoint["state_dict"] = self.action_cond_model.state_dict()
        checkpoint["ema_state_dict"] = self.ema_action_cond_model.state_dict()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema_action_cond_model.store(self.action_cond_model.parameters())
            self.ema_action_cond_model.copy_to(self.action_cond_model)
            if context is not None:
                print(f"{context}: Switched to EMA weights for adapter")
        try:
            yield None
        finally:
            if self.use_ema:
                self.ema_action_cond_model.restore(self.action_cond_model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights for adapter")

    def on_train_batch_end(self, *_args, **_kwargs):
        self.ema_action_cond_model(self.action_cond_model)

    def prepare_adapter(
        self,
        action_cond_unet: DiffusionWrapper,
        condition_adapter_on_base_outputs,
        learnt_mask,
        init_mask_bias,
        ema_action_cond_unet=None,
        use_base_ema=False,
        pretrain_steps=0,
    ):
        """Freeze the base model and set the action-conditioned Unet for the adapter model."""

        # use the EMA parameters from the base model
        if use_base_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            print("Using EMA parameters for base model.")

        # Freeze the base model
        for param in self.parameters():
            param.requires_grad = False

        # Set the unet for the action conditioned model
        self.action_cond_model = action_cond_unet
        if ema_action_cond_unet is not None:
            print("Restoring EMA for action conditioned model.")
            self.ema_action_cond_model = ema_action_cond_unet
        else:
            print("Creating EMA for action conditioned model.")
            self.ema_action_cond_model = LitEma(self.action_cond_model)

        # whether to condition the adapter on the base model outputs
        self.condition_adapter_on_base_outputs = condition_adapter_on_base_outputs

        # whether mask is output from the action conditioned model
        self.learnt_mask = learnt_mask
        self.init_mask_bias = init_mask_bias
        self.pretrain_steps = pretrain_steps

    def get_param_list(self):
        """Get the full list of parameters including those in the adapter."""
        base_params = super().get_param_list()
        action_params = list(self.action_cond_model.parameters())
        return base_params + action_params

    def apply_model(self, x_noisy, t, cond, **kwargs):
        """Override apply model function by first calling base model and then adapter model."""
        assert self.action_cond_model is not None, "Action conditioned model not set."

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}

        with torch.no_grad():
            base_output = self.model(x_noisy, t, **cond, **kwargs)

        if self.condition_adapter_on_base_outputs:
            adapter_input = torch.cat([x_noisy, base_output], dim=1)
        else:
            adapter_input = x_noisy

        # note: no zero conv here as unet is already initialized with zero conv output
        out = self.action_cond_model(adapter_input, t, **cond, **kwargs)

        if not self.learnt_mask:
            act_cond_output = out
            combined_output = base_output + act_cond_output
            info = {}
        else:
            if self.global_step < self.pretrain_steps:
                act_cond_output, mask = out
                mask = torch.zeros_like(mask)
                combined_output = act_cond_output
                if self.global_step % 10 == 0:
                    print(f"Pretrain step: {self.global_step} / {self.pretrain_steps}")
            else:
                act_cond_output, mask = out
                mask = F.sigmoid(mask + self.init_mask_bias)
                combined_output = base_output * mask + act_cond_output * (1 - mask)
            info = {
                "mask_mean": mask.mean().item(),
                "mask_std": mask.std().item(),
                "mask_min": mask.min().item(),
                "mask_max": mask.max().item(),
            }
        return combined_output, info
