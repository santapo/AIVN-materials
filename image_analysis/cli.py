from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import ClassifcationModel
from data import FlowerDataModule


class ClassifcationTrainingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        self._add_callbacks(parser)
        self._add_wandb_args(parser)

    def _add_wandb_args(self, parser):
        parser.add_argument('--use_wandb',
                            action='store_true',
                            help="Use Weight&Bias to track the training experiments")

        parser.add_argument('--wandb_project_name',
                            type=str,
                            default='Classfication',
                            help="Name of the Weight&Bias project when track the training experiments")

        parser.add_argument('--wandb_task_name',
                            type=str,
                            default='classification_test',
                            help="Name of the Weight&Bias task when track the training experiments")

    def _add_callbacks(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "save_checkpoint")
        parser.set_defaults({
            "save_checkpoint.dirpath": None,
            "save_checkpoint.filename": '{epoch}-{step}-{val_loss:.2f}',
            "save_checkpoint.save_top_k": -1
        })

    def before_fit(self):
        if self.config['use_wandb']:
            wandb = WandbLogger(project=self.config["wandb_project_name"], name=self.config["wandb_task_name"])
            wandb.log_hyperparams(self.config)

def cli_main():
    ClassifcationTrainingCLI(ClassifcationModel, FlowerDataModule)

if __name__ == "__main__":
    cli_main()
