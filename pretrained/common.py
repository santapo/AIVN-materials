import os
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    loss_fn: nn.Module,
                    optimizer: Optimizer,
                    scheduler = None,
                    device: str = "cpu",
                    use_wandb: bool = False,
                    **kwargs) -> None:
    """
    Training Loop
    """
    model.train()
    loss = 0
    for idx, (images, labels) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if use_wandb:
            wandb.log({"train_loss": loss.detach()})
    if scheduler is not None:
        scheduler.step()


def validate_network(model: nn.Module, 
                     dataloader: DataLoader, 
                     loss_fn: nn.Module, 
                     device: str = "cpu", 
                     use_wandb: bool = False, 
                     **kwargs) -> float:
    """
    Validation Loop
    """
    model.eval()
    correct = 0
    total = 0
    val_losses = []
    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
        val_losses.append(loss_fn(outputs, labels))
        preds = outputs.argmax(dim=1)
        correct += torch.sum(preds == labels)
        total += labels.numel()
    avg_loss = sum(val_losses) / len(val_losses)
    accuracy = correct.float() / total
    if use_wandb:
        wandb.log({"val_loss": avg_loss})
        wandb.log({"accuracy": accuracy})
    return accuracy


class FineTuningModel(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int, device: str = "cpu", freeze: bool = True) -> None:
        super(FineTuningModel, self).__init__()
        self.model = model
        self.model.to(device)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.num_classes = num_classes
        self.model.classifier[1] = nn.Linear(1280, self.num_classes)
        self.model.classifier[1].parameters().requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x


class TransferLearningModel(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int, device: str = "cpu") -> None:
        super(TransferLearningModel, self).__init__()
        self.model = model
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.num_classes = num_classes
        self.fc = nn.Linear(self.model.out_channels, self.num_classes)
        self.fc.to(device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = self.fc(x)
        return x