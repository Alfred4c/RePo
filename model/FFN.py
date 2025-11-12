import torch
import torch.nn as nn

from timm.layers.helpers import to_2tuple


class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.ffn(x)
