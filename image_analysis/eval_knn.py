import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torchmetrics import Accuracy

from data import FlowerDataModule
from model import ClassificationModel
from knn import KNearestNeighbor

def extract_features(model, dataloader):
    features = []
    labels = []
    model.eval()
    for idx, batch in enumerate(tqdm(dataloader)):
        (images, gts), _ = batch
        images = images.view(images.shape[0], -1)
        with torch.no_grad():
            feats = model.model(images)
        features.append(feats)
        labels.append(gts)
    features = torch.concat(features)
    labels = torch.concat(labels)
    return features, labels
        

def main(args):
    dm = FlowerDataModule(data_dir=args.data_dir)
    model = ClassificationModel().load_from_checkpoint(checkpoint_path=args.ckpt_path)
    model.model = nn.Sequential(*(list(model.model.children())[:-1]))

    train_data = extract_features(model, dm.train_dataloader())
    val_data = extract_features(model, dm.val_dataloader())

    metrics = Accuracy()
    nearest_neighbors = KNearestNeighbor()
    nearest_neighbors.train(train_data)

    val_pred = nearest_neighbors.predict(val_data[0], k=1)
    accuracy = metrics(val_pred, val_data[1])
    print(f"Accuracy with k=3: {accuracy}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/splitted_flowers")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="/root/AIVN-materials/image_analysis/logs/vanila_mlp/lightning_logs/version_0/Flower Classification/3jm88ltu/checkpoints/epoch=75-step=7219-val_loss=0.00.ckpt")
    args = parser.parse_args()

    main(args)