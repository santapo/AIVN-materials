import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl



class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32) 
        self.std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32) 

    @property
    def normalization(self):
        return transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())

    @property
    def train_transforms(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            self.normalization
        ])
    
    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalization
        ])

    def prepare_data(self):
        self.cifar_full = datasets.CIFAR100(root=self.data_dir, train=True, transform=self.train_transforms, download=True)
        self.cifar_test = datasets.CIFAR100(root=self.data_dir, train=False,transform=self.test_transforms , download=True)

    def setup(self, stage=None):
        self.cifar_train, self.cifar_val = random_split(self.cifar_full, [40000, 10000])

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    dm = CIFAR100DataModule()
    dm.prepare_data()
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch)
        break