import torch
import torch.nn as nn


from model.EncoderBlock import EncoderBlock
from model.GpsProjector import GPSProjector
from model.ResProjector import ResProjector
from model.GridGraphEncoder import GridGraphEncoder


import pickle


class RePo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpsproj = GPSProjector(output_dim=config["dim"])
        self.grid_image_emb = ResProjector(config)
        self.grid_graph_emb = GridGraphEncoder(config)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config["num_layers"])]
        )
        self.tile = pickle.load(open(config["large_tile_path"], "rb"))

        if config["cls"] == 1:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config["dim"]))
            self.image_token = nn.Parameter(torch.zeros(1, 1, config["dim"]))
            self.graph_token = nn.Parameter(torch.zeros(1, 1, config["dim"]))
        else:
            self.cls_token = None
            self.image_token = None
            self.graph_token = None

    def forward(self, traj, traj_length, global_indices):
        assert not torch.isnan(traj).any(), "NaN in traj"

        gps_input = traj[:, :, :6]
        gps_embs = self.gpsproj(gps_input)  # Linear layer (Eq. 7 in paper)

        B, L, D = gps_embs.shape

        batch_tile_seqs = [
            self.tile[global_idx.item()] for global_idx in global_indices
        ]

        grid_image_emb, grid_image_emb_length = self.grid_image_emb(batch_tile_seqs)
        grid_graph_emb, grid_graph_emb_length = self.grid_graph_emb(batch_tile_seqs)

        grid_length = grid_graph_emb_length

        Bg, Lg, D = grid_image_emb.shape

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, 1, -1)  # [B, 1, D]
            gps_embs = torch.cat([cls_token, gps_embs], dim=1)  # [B, L+1, D]
            traj_length = traj_length + 1
            image_token = self.image_token.expand(Bg, 1, -1)
            grid_image_emb = torch.cat([image_token, grid_image_emb], dim=1)
            graph_token = self.graph_token.expand(Bg, 1, -1)
            grid_graph_emb = torch.cat([graph_token, grid_graph_emb], dim=1)
            grid_length = grid_length + 1

        fused = gps_embs
        raw_grid_image_emb = grid_image_emb
        raw_grid_graph_emb = grid_graph_emb
        grid_emb = None
        for i, encoder in enumerate(self.encoder_blocks):
            fused, grid_emb = encoder(
                fused, traj_length, grid_image_emb, grid_graph_emb, grid_length
            )

            grid_image_emb = grid_emb + raw_grid_image_emb
            grid_graph_emb = grid_emb + raw_grid_graph_emb

        if self.cls_token is None:
            mask = (
                torch.arange(grid_emb.shape[1], device=grid_emb.device)[None, :]
                < grid_length[:, None]
            )  # [B, L]
            mask = mask.float()  # [B, L]

            masked_sum = (grid_emb * mask.unsqueeze(-1)).sum(dim=1)  # [B, D]
            lengths = mask.sum(dim=1, keepdim=True) + 1e-6
            embeddings = masked_sum / lengths
        else:
            embeddings = grid_emb[:, 0]
        return embeddings
