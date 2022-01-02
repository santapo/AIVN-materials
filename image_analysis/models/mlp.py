import torch.nn as nn


def mlp_block(input_dim, expand_ratio=2, dropout=0., act_fn=nn.GELU):
    return nn.Sequential(
        nn.Linear(input_dim, int(input_dim*expand_ratio)),
        act_fn(),
        nn.Dropout(dropout))

class VanilaMLP(nn.Module):
    def __init__(self, input_dim:int, num_classes:int = 5):
        super().__init__()

        self.cfg = [
            # n, t, p
            [2, 1.2, 0. ],
            [2, 0.8, 0.1],
            [2, 0.7, 0. ]
        ]

        layers = []
        for n, t, p in self.cfg:
            for _ in range(n):
                layers.append(mlp_block(input_dim, expand_ratio=t, dropout=p))
                input_dim = int(input_dim*t)

        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, -1)
        x = self.layers(x)
        x = self.head(x)
        return x