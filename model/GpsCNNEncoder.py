import torch
import torch.nn as nn
import torch.nn.functional as F


# Locality-aware encoding
class GPSCNNEncoder(nn.Module):
    def __init__(self, dim):
        super(GPSCNNEncoder, self).__init__()

        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, dilation=1, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)

        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, dilation=1, padding=1)
        self.norm2 = nn.GroupNorm(8, dim)

        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, dilation=1, padding=1)
        self.norm3 = nn.GroupNorm(8, dim)

        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x, traj_length):
        """
        x: (B, L, dim)
        traj_length: (B,)
        return: (B, L, dim)
        """

        x = x.permute(0, 2, 1)  # (B, C, L)

        max_len = x.size(-1)
        mask = (
            (torch.arange(max_len, device=x.device)[None, :] < traj_length[:, None])
            .unsqueeze(1)
            .float()
        )  # (B, 1, L)

        x = self.activation(self.norm1(self.conv1(x))) * mask
        x = self.activation(self.norm2(self.conv2(x))) * mask
        x = self.norm3(self.conv3(x)) * mask  # (Eq. 8 in paper)

        x = x.permute(0, 2, 1)  # (B, L, C)
        return x
