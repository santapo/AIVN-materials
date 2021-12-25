import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(
        self,
        num_classes,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(1600, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


if __name__ == "__main__":
    import torch
    rand_inp = torch.rand(1, 3, 32, 32)
    model = LeNet(num_classes=100)
    res = model(rand_inp)