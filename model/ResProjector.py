import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pickle
from collections import defaultdict


class ResProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.dim = config["dim"]

        with open(config["grid_emb_path"], "rb") as f:
            raw_grid_embs = pickle.load(f)

        self.known_tiles = list(raw_grid_embs.keys())
        emb_matrix = torch.stack(
            [
                torch.tensor(raw_grid_embs[tile], dtype=torch.float32)
                for tile in self.known_tiles
            ]
        )

        padding_embedding = torch.zeros((1, 2048), dtype=torch.float32)
        emb_matrix_with_padding = torch.cat([padding_embedding, emb_matrix], dim=0).to(
            self.device
        )

        self.tile_to_idx = {tile: idx + 1 for idx, tile in enumerate(self.known_tiles)}
        self.padding_idx = 0

        self.embedding = nn.Embedding.from_pretrained(
            embeddings=emb_matrix_with_padding,
            padding_idx=self.padding_idx,
            freeze=True,
        )

        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.dim),
        )

    def _tile_seq_to_indices(self, batch_tile_seqs):
        return [
            torch.tensor(
                [self.tile_to_idx.get(tile, self.padding_idx) for tile in seq],
                dtype=torch.long,
                device=self.device,
            )
            for seq in batch_tile_seqs
        ]

    def forward(self, batch_tile_seqs):
        indexed_seqs = self._tile_seq_to_indices(batch_tile_seqs)
        padded_indices = pad_sequence(
            indexed_seqs, batch_first=True, padding_value=self.padding_idx
        )
        raw_embs = self.embedding(padded_indices)
        compressed = self.projector(raw_embs)

        lengths = torch.tensor(
            [len(seq) for seq in batch_tile_seqs], device=self.device
        )

        return compressed, lengths
