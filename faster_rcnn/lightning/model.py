from pytorch_lightning.core import datamodule
import torch
import pytorch_lightning as pl

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn


class FasterRCNN(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool = False,
        optimizer: str = "adam",
        lr: float = 0.0001,
    ):
        super().__init__()

        self.pretrained = pretrained
        self.optimizer = optimizer
        self.lr = lr

        self.model = self._build_model()
       
    def _build_model(self):
        model = fasterrcnn_resnet50_fpn(
            pretrained=self.pretrained
        )
        return model

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum([loss for loss in loss_dict.values()])
        self.log("training_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # import ipdb; ipdb.set_trace()
        result = self.model(images, targets)
        # TODO: log mean Average Precision
        # loss = sum([loss for loss in loss_dict.values()])
        # self.log("validation_loss", loss, prog_bar=True, on_step=True, on_epoch=True)


if __name__ == "__main__":
    from pl_bolts.datamodules import VOCDetectionDataModule

    datamodule = VOCDetectionDataModule(data_dir="../lightning_bolts/.", num_workers=4)
    model = FasterRCNN(pretrained=True)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, datamodule)
