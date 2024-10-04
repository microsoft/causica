import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from avid_utils.image import preprocess_images
from einops import rearrange
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy

from dwma.models.classifier import GaussianNoiseActionClassifier


class ClassifierModule(pl.LightningModule):
    def __init__(
        self,
        classifier: GaussianNoiseActionClassifier,
        learning_rate: float = 1e-4,
        init_frames: int = 1,
        next_frames: int = 1,
        linear_warmup_steps: int = 1000,
        train_with_noise: bool = True,
    ):
        """Initialize the diffusion module.

        Args:
            classifier: the classifier model that will classify actions from noisy frame sequence
            learning_rate: learning rate for the optimizer
            init_frames: number of clean frames initial frames to pass to classifier
            next_frames: number of noisy frames to pass the classifier
        """
        super().__init__()
        self.classifier = classifier
        self.num_actions = classifier.model.num_actions
        self.learning_rate = learning_rate
        self.init_frames = init_frames
        self.next_frames = next_frames
        self.train_with_noise = train_with_noise
        self.linear_warmup_steps = linear_warmup_steps
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass", num_classes=self.num_actions)
        self.save_hyperparameters()

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        loss, acc = self.step(batch, add_noise=self.train_with_noise)
        self.log("train/loss", loss)
        self.log("train/acc", acc)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            loss, acc = self.step(batch, add_noise=self.train_with_noise)
            self.log("val/loss", loss)
            self.log("val/acc", acc)
            if self.train_with_noise:
                loss_no_noise, acc_no_noise = self.step(batch, add_noise=False)
                self.log("val/loss_no_noise", loss_no_noise)
                self.log("val/acc_no_noise", acc_no_noise)

    def step(self, batch: dict[str, torch.Tensor], add_noise=True) -> torch.Tensor:
        init_frames = preprocess_images(batch["obs"][:, 0 : self.init_frames])
        next_frames = preprocess_images(batch["obs"][:, self.init_frames : self.init_frames + self.next_frames])

        act_targs = batch["act"][:, self.init_frames - 1 : self.init_frames + self.next_frames - 1]
        act_preds = self.classifier(x=next_frames, x_init=init_frames, add_noise=add_noise)

        act_targs = rearrange(act_targs, "b t -> (b t)")
        act_preds = rearrange(act_preds, "b t a -> (b t) a")
        loss = self.loss_fn(act_preds, act_targs)

        preds = torch.argmax(act_preds, dim=-1)
        acc = self.accuracy(preds, act_targs)
        return loss, acc

    def get_prediction_accuracy(
        self, cond_frames: torch.Tensor, next_frames: torch.Tensor, real_act: torch.Tensor
    ) -> torch.Tensor:
        """Given cond frames and next frames, predict the action and return the accuracy.

        Args:
            cond_frames: tensor of initial frames in shape (b, t_init, h, w, c)
            next_frames: tensor of next frames in shape (b, t_next, h, w, c)
            real_act: tensor of real actions in shape (b, t_init + t_next)
        """
        init_frames = preprocess_images(cond_frames)
        next_frames = preprocess_images(next_frames)

        act_preds = self.classifier(x=next_frames, x_init=init_frames, add_noise=False)
        act_targs = real_act[:, init_frames.shape[2] - 1 : init_frames.shape[2] + next_frames.shape[2] - 1]

        act_targs = rearrange(act_targs, "b t -> (b t)")
        act_preds = rearrange(act_preds, "b t a -> (b t) a")
        preds = torch.argmax(act_preds, dim=-1)
        acc = self.accuracy(preds, act_targs)

        # get the probability assigned to true action
        act_probs = F.softmax(act_preds, dim=-1)
        real_act_prob = act_probs[torch.arange(act_probs.shape[0]), act_targs].mean()
        return acc, real_act_prob

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
