import torch
import torch.nn as nn

import math


# SineEncoder from https://github.com/DSL-Lab/Specformer
class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(2) * div
        eeig = torch.cat((e.unsqueeze(2), torch.sin(pe), torch.cos(pe)), dim=2)

        return self.eig_w(eeig)
