import torch.nn as nn


def mlp_block(input_dim, output_dim=1000, dropout=0., act_fn=nn.GELU):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        act_fn(),
        nn.BatchNorm1d(output_dim),
        nn.Dropout(dropout))

class VanilaMLP(nn.Module):
    def __init__(self, input_dim:int = 3072, num_classes:int = 5):
        super().__init__()

        self.cfg = [
            # n, out, p
            [1, 1000, 0. ],
            [1, 500 , 0.2],
            [2, 200 , 0.2],
        ]

        layers = []
        for n, out, p in self.cfg:
            for _ in range(n):
                layers.append(mlp_block(input_dim, output_dim=out, dropout=p))
                input_dim = out

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