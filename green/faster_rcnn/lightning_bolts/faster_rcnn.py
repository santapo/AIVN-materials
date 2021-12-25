import pytorch_lightning as pl
from argparse import ArgumentParser

from pl_bolts.models.detection.faster_rcnn import FasterRCNN



def run_cli():
    from pl_bolts.datamodules import VOCDetectionDataModule

    pl.seed_everything(42)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=1)
    parser = FasterRCNN.add_model_specific_args(parser)

    args = parser.parse_args()

    datamodule = VOCDetectionDataModule.from_argparse_args(args)
    args.num_classes = datamodule.num_classes

    model = FasterRCNN(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    run_cli()