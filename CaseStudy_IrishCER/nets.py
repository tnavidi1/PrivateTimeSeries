import torch
import torch.nn as nn



class Classifier(nn.Module):
    def __init__(self, z_dim=10, y_dim=0):
        super(Classifier, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 50),
            nn.ReLU(),
            # nn.ELU(),
            nn.Linear(50, 24),
            nn.ReLU(),
            # nn.ELU(),
            nn.Linear(24, y_dim)
            # nn.Linear(z_dim, y_dim),
        )

    def forward(self, x):
        o = self.net(x)
        return o



