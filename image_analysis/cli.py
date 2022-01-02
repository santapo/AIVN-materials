from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint

from model import ClassifcationModel
from data import FlowerDataModule


class ClassifcationTrainingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        self._add_callbacks(parser)

    def _add_callbacks(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "save_checkpoint")
        parser.set_defaults({
            "save_checkpoint.dirpath": None,
            "save_checkpoint.filename": '{epoch}-{step}-{val_loss:.2f}',
            "save_checkpoint.save_top_k": -1
        })

def cli_main():
    ClassifcationTrainingCLI(ClassifcationModel, FlowerDataModule)

if __name__ == "__main__":
    cli_main()
