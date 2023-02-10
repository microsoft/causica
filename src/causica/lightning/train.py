from causica.lightning.cli import LightningCLIWithDefaults
from causica.lightning.data_modules import CSuiteDataModule
from causica.lightning.modules import DECIModule

if __name__ == "__main__":
    LightningCLIWithDefaults(DECIModule, CSuiteDataModule, run=True, save_config_callback=None)
