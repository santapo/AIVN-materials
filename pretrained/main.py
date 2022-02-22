import argparse
import logging

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from common import train_one_epoch, validate_network, set_seed
from common import FineTuningModel, TransferLearningModel


logger = logging.getLogger("AIVN_pretrained")


def get_model(mode, device="cpu"):
    if args.mode == "finetuning":
        model = models.mobilenet_v2(pretrained=True).to(device)
        model = FineTuningModel(model, args.num_classes, device=device, freeze=True)
    if args.mode == "pretrained":
        model = models.mobilenet_v2(pretrained=True).to(device)
        model = FineTuningModel(model, args.num_classes, device=device, freeze=False)
    if args.mode == "transfer":
        model = models.mobilenet_v2(pretrained=True).to(device)
        model = TransferLearningModel(model, args.num_classes, device=device)
    if args.mode == "scratch":
        model = models.mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(device)
    return model

def main(args):
    if args.use_wandb:
        wandb.init(project=args.project_name, name=args.run_name, config=args)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                            std=(0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                            std=(0.2675, 0.2565, 0.2761))
    ])

    data_train = datasets.ImageFolder(root=args.data_dir + "/ll_train", transform=train_transform)
    data_val = datasets.ImageFolder(root=args.data_dir + "/valid", transform=test_transform)
    data_test = datasets.ImageFolder(root=args.data_dir + "/test", transform=test_transform)

    train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(dataset=data_val, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(dataset=data_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    logger.info(f"Data loaded with {len(data_train)} train, {len(data_test)} val imgs and {len(data_test)} test imgs")

    model = get_model(args.mode, device)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None

    for epoch in range(args.num_epochs):
        train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler=scheduler, device=device, use_wandb=args.use_wandb)
        if (epoch % args.val_every_n_epochs) == 0:
            accuracy = validate_network(model, val_loader, loss_fn, device=device, use_wandb=args.use_wandb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb")
    parser.add_argument("--project_name", type=str, default="AIVN pretrained")
    parser.add_argument("--run_name", type=str, default="mobilenet_v2")
    parser.add_argument("--mode", type=str, default="finetuning", help="finetuning, pretrained, transfer, scratch")
    parser.add_argument("--lr", type=float, default=0.1),
    parser.add_argument("--gpus", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, default="data/little_classes")
    parser.add_argument("--batch_size", type=int, default=512),
    parser.add_argument("--num_workers", type=int, default=4),
    parser.add_argument("--num_classes", type=int, default=100),
    parser.add_argument("--val_every_n_epochs", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)