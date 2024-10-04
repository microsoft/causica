from contextlib import contextmanager

from lvdm.ema import LitEma
from lvdm.models.ddpm3d import DiffusionWrapper, LatentVisualDiffusion


class ControlNetAdapter(LatentVisualDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.eval()
        self.model.requires_grad_(False)
        self.control_net = None

    def on_save_checkpoint(self, checkpoint):
        """Only save the action-conditioned control net in the checkpoint as no need to resave the base model."""

        if "state_dict" in checkpoint:
            del checkpoint["state_dict"]

        checkpoint["state_dict"] = self.control_net.state_dict()
        checkpoint["ema_state_dict"] = self.ema_control_net.state_dict()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema_control_net.store(self.control_net.parameters())
            self.ema_control_net.copy_to(self.control_net)
            if context is not None:
                print(f"{context}: Switched to EMA weights for adapter")
        try:
            yield None
        finally:
            if self.use_ema:
                self.ema_control_net.restore(self.control_net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights for adapter")

    def on_train_batch_end(self, *args, **kwargs):
        self.ema_control_net(self.control_net)

    def prepare_adapter(
        self,
        control_net: DiffusionWrapper,
        ema_control_net=None,
        use_base_ema=False,
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
        self.control_net = control_net
        if ema_control_net is not None:
            print("Restoring EMA for control net model.")
            self.ema_control_net = ema_control_net
        else:
            print("Creating EMA for control net model.")
            self.ema_control_net = LitEma(self.control_net)

    def get_param_list(self):
        """Get the full list of parameters including those in the adapter."""
        base_params = super().get_param_list()
        control_net_params = list(self.control_net.parameters())
        return base_params + control_net_params

    def apply_model(self, x_noisy, t, cond, **kwargs):
        """Override apply model function by first calling base model and then adapter model."""
        assert self.control_net is not None, "Control net model not set."

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}

        control_outputs = self.control_net(x_noisy, t, **cond, **kwargs)
        combined_output = self.model(x_noisy, t, control=control_outputs, only_mid_control=False, **cond, **kwargs)

        info = {}
        return combined_output, info
