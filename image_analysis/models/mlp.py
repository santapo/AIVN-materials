import torch.nn as nn


def mlp_block(input_dim, expand_ratio=2, dropout=0., act_fn=nn.GELU):
    return nn.Sequential(
        nn.Linear(input_dim, int(input_dim*expand_ratio)),
        act_fn(),
        nn.Dropout(dropout))

class VanilaMLP(nn.Module):
    def __init__(self, input_dim:int = 3072, num_classes:int = 5):
        super().__init__()

        self.cfg = [
            # n, t, p
            [1, 0.02, 0.5],
            [1, 1.   , 0.2],
            [2, 0.8 , 0.2],
        ]

        layers = []
        for n, t, p in self.cfg:
            for _ in range(n):
                layers.append(mlp_block(input_dim, expand_ratio=t, dropout=p))
                input_dim = int(input_dim*t)

        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, num_classes)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, -1)
        x = self.layers(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    import torch
    rand_inp = torch.rand(3, 3, 32, 32)
    model = VanilaMLP(input_dim=32*32*3, num_classes=5)
    res = model(rand_inp)
    print(res.shape)