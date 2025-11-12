import torch
import torch.nn as nn
import torch.nn.functional as F


##Correlation-aware encoding
class DAGG_NAPLGraph(nn.Module):
    def __init__(self, input_dim, output_dim, node_emb_dim=128):
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.output_dim = output_dim

        self.graph_embedding = nn.Linear(input_dim, node_emb_dim)
        self.Eg = nn.Linear(input_dim, node_emb_dim)
        self.Wg = nn.Parameter(torch.randn(node_emb_dim, output_dim))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, traj, traj_length):
        # traj: (B, L, D)
        B, L, D_in = traj.shape
        device = traj.device

        node_feats = traj  # (B, L, D_in)

        EA = self.graph_embedding(node_feats)  # (B, L, node_emb_dim) (Eq. 9 in paper)

        sim = torch.relu(
            torch.bmm(EA, EA.transpose(1, 2))
        )  # (B, L, L) (Eq. 10 in paper)

        mask = torch.arange(L, device=device)[None, :] < traj_length[:, None]  # (B, L)
        mask_float = mask.float()
        sim = sim * mask_float[:, None, :]

        A = F.softmax(sim, dim=-1)  # (B, L, L) (Eq. 10 in paper)

        Eg = self.Eg(node_feats)  # (B, L, node_emb_dim)
        theta = Eg @ self.Wg  # (B, L, output_dim)

        propagated = torch.bmm(A, theta)  # (B, L, output_dim)
        propagated = self.norm(propagated)  # (Eq. 11 in paper)

        propagated = propagated * mask_float.unsqueeze(-1)

        return propagated  # (B, L, output_dim)
