import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1, MetricCollection

from models import get_backbone


class ClassifcationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "lenet",
        num_classes: int = 100,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        momentum: float = 0.9
    ):
        super().__init__()

        self.model = get_backbone(model_name, num_classes)
        self.loss_fn = nn.NLLLoss(reduction="mean")
        self.optimizer = optimizer
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

        self._build_metrics()


    def _build_metrics(self):
        metrics = MetricCollection([Accuracy(), F1()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        y = self.model.forward(x)
        x = F.log_softmax(y, dim=-1)
        return x

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)

        if self.optimizer == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.lr,
                                    momentum=self.momentum, weight_decay=self.weight_decay) 
        return optim

    def training_step(self, train_batch, batch_idx):
        # import ipdb; ipdb.set_trace()
        images, labels = train_batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, labels)
        self.log("train_loss", loss, logger=True)
        self.train_metrics.update(preds, labels)
        return loss

    def training_epoch_end(self, outputs):
        calculated_metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        self.log_dict(calculated_metrics)
        super().training_epoch_end(outputs)

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, labels)
        self.log("val_loss", loss, logger=True)
        self.val_metrics.update(preds, labels)
    
    def validation_epoch_end(self, outputs):
        calculated_metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        self.log_dict(calculated_metrics)
        super().validation_epoch_end(outputs)

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, labels)
        self.log("test_loss", loss, logger=True)


if __name__ == "__main__":
    from data import CIFAR100DataModule
    model = ClassifcationModel()
    datamodule = CIFAR100DataModule(num_workers=0)
    datamodule.prepare_data()
    datamodule.setup()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, datamodule)