import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1, MetricCollection

from models import get_backbone


class ClassifcationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vanila",
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
        self.train_metrics = MetricCollection([Accuracy(), F1()])
        self.val_metrics = MetricCollection([Accuracy(), F1()])

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
        self.log("loss/train", loss, logger=True)
        self.train_metrics.update(preds, labels)
        return loss

    def training_epoch_end(self, outputs):
        calculated_metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        calculated_metrics = {f"{key}/train": val for key, val in calculated_metrics.items()}
        self.log_dict(calculated_metrics)
        super().training_epoch_end(outputs)

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, labels)
        self.log("loss/val", loss, logger=True)
        self.val_metrics.update(preds, labels)
    
    def validation_epoch_end(self, outputs):
        calculated_metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        calculated_metrics = {f"{key}/val": val for key, val in calculated_metrics.items()}
        self.log_dict(calculated_metrics)
        super().validation_epoch_end(outputs)

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, labels)
        self.log("test_loss", loss, logger=True)