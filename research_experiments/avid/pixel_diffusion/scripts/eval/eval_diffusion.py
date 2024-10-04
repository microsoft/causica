import pytorch_lightning as pl
import torch
from avid_utils.eval import get_run_logger

from dwma.lightning.cli.cli import LightningCLIDiffusion
from dwma.lightning.data_modules.procgen_data import ProcgenDataModule
from dwma.lightning.modules.diffusion_module import DiffusionModule

if __name__ == "__main__":
    cli = LightningCLIDiffusion(
        model_class=DiffusionModule,
        datamodule_class=ProcgenDataModule,
        run=False,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )

    device = torch.device(
        f"cuda:{cli.config.trainer.devices[0]}" if isinstance(cli.config.trainer.devices, list) else "cuda:0"
    )

    # checkpoint to evaluate
    checkpoint = torch.load(cli.config.checkpoint_path, map_location="cpu")
    cli.model.load_state_dict(checkpoint["state_dict"])
    cli.model = cli.model.to(device)

    # preload data
    pl.seed_everything(0)
    val_dataloader = cli.datamodule.val_dataloader()
    run_logger = get_run_logger(cli.config, run_kwargs={})

    # evaluate a number of batches using these parameters
    for batch_num, batch in enumerate(val_dataloader):
        if batch_num >= cli.config.num_batches:
            break
        val_batch = {k: v.to(device) for k, v in batch.items()}
        cli.model.generate_videos(
            obs=val_batch["obs"],
            act=val_batch["act"],
            prefix="val",
            step=batch_num,
            logger=run_logger,
        )
        if batch_num >= cli.config.num_batches:
            break
