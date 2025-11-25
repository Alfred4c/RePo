import torch
import torch.nn as nn
import math
from model.GpsCNNEncoder import GPSCNNEncoder
from model.GpsAdaptiveGraph import DAGG_NAPLGraph
from model.GpsCDEEncoder import NeuralCDEEncoder
from model.Router import Router
from model.FFN import FFN


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cnnencoder = GPSCNNEncoder(dim=config["dim"])
        self.adaptivegraphencoder = DAGG_NAPLGraph(
            input_dim=config["dim"],
            output_dim=config["dim"],
            node_emb_dim=config["dim"],
        )
        self.cdeencoder = NeuralCDEEncoder(
            input_dim=config["dim"], hidden_dim=6, output_dim=config["dim"]
        )
        self.router = Router(input_dim=config["dim"])

        self.self_attention_image = nn.MultiheadAttention(
            embed_dim=config["dim"], num_heads=config["num_heads"], batch_first=True
        )
        self.self_attention_graph = nn.MultiheadAttention(
            embed_dim=config["dim"], num_heads=config["num_heads"], batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config["dim"], num_heads=config["num_heads"], batch_first=True
        )
        self.ffn = FFN(config["dim"], hidden_dim=config["dim"] // 2)
        self.norm = nn.LayerNorm(config["dim"])
        self.norm2 = nn.LayerNorm(config["dim"])

    def forward(
        self, gps_embs, gps_length, grid_image_emb, grid_graph_emb, grid_length
    ):

        L = gps_embs.shape[1]
        Lg = grid_image_emb.shape[1]
        grid_mask = (
            torch.arange(Lg, device=grid_image_emb.device)[None, :]
            >= grid_length[:, None]
        )
        image_attn_out, _ = self.self_attention_image(
            grid_image_emb, grid_image_emb, grid_image_emb, key_padding_mask=grid_mask
        )
        graph_attn_out, _ = self.self_attention_graph(
            grid_graph_emb, grid_graph_emb, grid_graph_emb, key_padding_mask=grid_mask
        )
        q = self.norm(image_attn_out + graph_attn_out)
        cnn_out = self.cnnencoder(gps_embs, gps_length)
        adaptivegraph_out = self.adaptivegraphencoder(gps_embs, gps_length)
        cde_out = self.cdeencoder(gps_embs, gps_length)
        weights = self.router(cnn_out, adaptivegraph_out, cde_out)
        w1 = weights[:, :, 0:1]
        w2 = weights[:, :, 1:2]
        w3 = weights[:, :, 2:3]
        fused = (
            w1 * cnn_out + w2 * adaptivegraph_out + w3 * cde_out
        )  # shape: (B, L, D) (Eq. 16 in paper)
        kv = fused + gps_embs
        mask = (
            torch.arange(L, device=self.config["device"])[None, :]
            >= gps_length[:, None]
        )  # shape: (B, L)
        cross_out, _ = self.cross_attention(
            q, kv, kv, key_padding_mask=mask
        )  # (Eq. 17 in paper)
        cross_out = cross_out + q
        ffn_input = self.norm2(cross_out + q)
        out = self.ffn(ffn_input) + cross_out  # (Eq. 18 in paper)
        return fused, out
