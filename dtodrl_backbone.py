"""
Backbone used by the paper-aligned DTODRL policy.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, global_mean_pool

from Graph_policy import build_graph_inputs_from_adj


class DTODRLGATEncoder(nn.Module):
    """Standard 3-layer GAT encoder with hidden size 128 and 3 heads."""

    def __init__(self, in_dim: int = 6, hidden_dim: int = 128, heads: int = 3, num_layers: int = 3):
        super().__init__()
        out_per_head = (hidden_dim + heads - 1) // heads
        mid_dim = out_per_head * heads
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, out_per_head, heads=heads, add_self_loops=True, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(mid_dim, out_per_head, heads=heads, add_self_loops=True, concat=True))
        self.convs.append(GATConv(mid_dim, hidden_dim, heads=1, add_self_loops=True, concat=False))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx < self.num_layers - 1:
                x = F.leaky_relu(x, 0.2)
        return x


class DTODRLLocationEncoder(nn.Module):
    """MLP over the 2D location profile (EFT, f)."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class DTODRLBackbone(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 6,
        location_feature_dim: int = 2,
        gat_hidden: int = 128,
        gat_heads: int = 3,
        gat_layers: int = 3,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.gat_encoder = DTODRLGATEncoder(
            in_dim=node_feature_dim,
            hidden_dim=gat_hidden,
            heads=gat_heads,
            num_layers=gat_layers,
        )
        self.location_encoder = DTODRLLocationEncoder(
            in_dim=location_feature_dim,
            hidden_dim=mlp_hidden,
            out_dim=gat_hidden,
        )

    def encode_nodes(self, obs: dict) -> Tuple[Tensor, Optional[Tensor]]:
        device = obs["nodes_C"].device
        node_features = torch.stack(
            [
                obs["nodes_C"].float(),
                obs["nodes_D"].float(),
                obs["nodes_in_degree"].float(),
                obs["nodes_out_degree"].float(),
                obs["nodes_loc"].float(),
                obs["nodes_ava"].float(),
            ],
            dim=-1,
        )

        adj = obs["adj"]
        edge_attr = obs["edge_attr"]
        edge_index, _, batch, _ = build_graph_inputs_from_adj(adj, edge_attr)
        edge_index = edge_index.to(device)

        if node_features.dim() == 3:
            batch_size, num_nodes, feat_dim = node_features.shape
            node_features = node_features.reshape(batch_size * num_nodes, feat_dim)

        node_embs = self.gat_encoder(node_features.to(device), edge_index, batch)
        return node_embs, batch

    def encode_locations(self, obs: dict) -> Tensor:
        loc_features = torch.stack(
            [
                obs["loc_min_processor_EAT"].float(),
                obs["loc_cpu_speed"].float(),
            ],
            dim=-1,
        )
        if loc_features.dim() == 3:
            batch_size, num_locations, _ = loc_features.shape
            loc_flat = loc_features.reshape(batch_size * num_locations, -1)
            return self.location_encoder(loc_flat).reshape(batch_size, num_locations, -1)
        return self.location_encoder(loc_features)

    def pool_nodes(self, node_embs: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        if batch is not None:
            return global_mean_pool(node_embs, batch)
        return node_embs.mean(dim=0, keepdim=True)

    def pool_locations(self, loc_embs: Tensor) -> Tensor:
        if loc_embs.dim() == 3:
            return loc_embs.mean(dim=1)
        return loc_embs.mean(dim=0, keepdim=True)
