import torch
from torch import nn

class Agent(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Agent, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, out_dim),
        )

    def forward(self, obs):
        return self.mlp(obs)
