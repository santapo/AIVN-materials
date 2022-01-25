import torch.nn as nn
import torch.nn.functional as F


class Resnet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=0)
        self.res1 = ResidualBlock(input_dims=128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.res2 = ResidualBlock(input_dims=512)

        self.head = nn.Sequential(
            nn.Linear(512, 156),
            nn.ReLU(),
            nn.Linear(156, num_classes)
        )
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.res1(x)
        
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = self.res2(x)
        
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dims, input_dims, kernel_size=(3, 3), stride=1, padding=1)        
        self.conv2 = nn.Conv2d(input_dims, input_dims, kernel_size=(3, 3), stride=1, padding=1)        

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        return out + x

if __name__ == "__main__":
    import torch
    rand_inp = torch.rand(3, 3, 32, 32)
    model = Resnet()
    print(model(rand_inp).shape)