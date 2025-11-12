import torch
import torch.nn as nn


class GPSProjector(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.projector = nn.Linear(6, output_dim)

    def forward(self, gps_input):
        """
        gps_input: Tensor (batch_size, seq_len, input_dim)
        return: Tensor (batch_size, seq_len, output_dim)
        """
        return self.projector(gps_input)
