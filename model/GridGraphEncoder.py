import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pickle


class GridGraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]

        # Load the raw graph embeddings
        with open(config["graph_emb_path"], "rb") as f:
            raw_embs = pickle.load(f)  # {tile_name: embedding (dim,)}

        self.known_tiles = list(raw_embs.keys())
        self.dim = config["dim"]

        # Create the embedding matrix from the raw embeddings
        emb_matrix = torch.stack(
            [
                torch.tensor(raw_embs[tile], dtype=torch.float32)
                for tile in self.known_tiles
            ]
        )
        padding_embedding = torch.zeros((1, 128), dtype=torch.float32)

        # Add padding to the embedding matrix
        emb_matrix_with_padding = torch.cat([padding_embedding, emb_matrix], dim=0).to(
            self.device
        )

        # Mapping of tiles to indices
        self.tile_to_idx = {tile: idx + 1 for idx, tile in enumerate(self.known_tiles)}
        self.padding_idx = 0

        # Create an embedding layer from the pre-trained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=emb_matrix_with_padding,
            padding_idx=self.padding_idx,
            freeze=True,  # Do not train the structure embeddings
        )

        # Add a projection layer to map original embeddings to the config's `dim`
        self.projection = nn.Linear(
            emb_matrix.shape[1], self.dim
        )  # Map to config['dim']

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
        # Convert tile sequences to indices
        indexed_seqs = self._tile_seq_to_indices(batch_tile_seqs)

        # Pad the sequences
        padded_indices = pad_sequence(
            indexed_seqs, batch_first=True, padding_value=self.padding_idx
        )

        # Get embeddings for the padded sequences
        embedding = self.embedding(padded_indices)

        # Project the embeddings to the desired dimension
        projected_embedding = self.projection(embedding)  # shape: [B, L, dim]

        # Calculate the lengths of the sequences
        lengths = torch.tensor(
            [len(seq) for seq in batch_tile_seqs], device=self.device
        )

        return projected_embedding, lengths  # shape: [B, L, dim]
