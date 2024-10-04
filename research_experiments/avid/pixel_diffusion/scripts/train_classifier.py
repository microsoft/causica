from dwma.lightning.cli.cli import ClassifierCLI
from dwma.lightning.data_modules.procgen_data import ProcgenDataModule
from dwma.lightning.modules.classifier_module import ClassifierModule

if __name__ == "__main__":
    cli = ClassifierCLI(
        model_class=ClassifierModule,
        datamodule_class=ProcgenDataModule,
        run=False,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
