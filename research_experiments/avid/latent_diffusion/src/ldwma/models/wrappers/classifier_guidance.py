import torch
import torch.nn.functional as F
from einops import rearrange
from lvdm.common import extract_into_tensor


class ClassifierGuidanceWrapper:
    def __init__(self, base_model, classifier, classifier_wt, use_base_ema=False):
        super().__init__()
        self.base_model = base_model
        self.classifier = classifier
        self.classifier_wt = classifier_wt
        self.use_base_ema = use_base_ema

        assert self.classifier.discretize_actions, "Classifier must have discretize_actions set to True"

        # use the EMA parameters from the base model
        if use_base_ema:
            self.base_model.model_ema.store(self.base_model.model.parameters())
            self.base_model.model_ema.copy_to(self.base_model.model)
            print("Using EMA parameters for base model.")

    def __getattr__(self, name):
        if name in self.__dict__:
            return super().__getattr__(name)
        return getattr(self.base_model, name)

    def model_predictions(self, x_noisy, t, cond, **kwargs):
        """Override the model predictions method to combine predictions from base and classifier models."""

        pred_noise, _, _ = self.base_model.model_predictions(x_noisy, t, cond, **kwargs)

        targs = self.classifier.to_discrete_actions(cond["act"])
        targs = rearrange(targs, "b t a -> (b t a)")

        # compute gradient of classifier
        with torch.enable_grad():
            x_in = x_noisy.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            logits = rearrange(logits, "b t (a d) -> (b t a) d", d=self.classifier.discrete_action_bins)
            log_probs = F.log_softmax(logits, dim=-1)
            real_act_prob = log_probs[torch.arange(len(targs)), targs]
            grad = torch.autograd.grad(real_act_prob.sum(), x_in)[0]

        # compute pred noise for classifier guided ddim https://arxiv.org/pdf/2105.05233
        pred_noise = (
            pred_noise
            - extract_into_tensor(self.base_model.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
            * grad
            * self.classifier_wt
        )

        # recompute pred_v and x_start from new pred_noise
        x_start = self.base_model.predict_start_from_noise(x_noisy, t, pred_noise)
        pred_v = self.base_model.get_v(x_start, pred_noise, t)
        return pred_noise, pred_v, x_start
