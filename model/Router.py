import torch
import torch.nn as nn
import torch.nn.functional as F


## Mixture-of-Experts Dynamic Fusion(weights)
class Router(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Router, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # shared mlp
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, cnn_out, graph_out, cde_out):
        weight_cnn = self.mlp(cnn_out)  # (B, L, 1)
        weight_graph = self.mlp(graph_out)
        weight_cde = self.mlp(cde_out)

        weights = torch.cat([weight_cnn, weight_graph, weight_cde], dim=-1)  # (B, L, 3)

        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = F.softmax(weights + 1e-8, dim=-1)

        return weights
