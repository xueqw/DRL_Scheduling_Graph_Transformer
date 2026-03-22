"""
DTODRL 论文专用 Backbone

与原文完全一致:
- GAT: 标准 GATConv, 3 层, hidden 128, heads 3, LeakyReLU
- Location: 论文 (EFT, f) 两维, MLP hidden 256
- 节点特征: 6 维 (Ci, Di, in_degree, out_degree, loci, avai)
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import GATConv, global_mean_pool
from Graph_policy import build_graph_inputs_from_adj


class DTODRLGATEncoder(nn.Module):
    """论文标准 GAT: 3 层, hidden 128, heads 3, LeakyReLU"""
    def __init__(self, in_dim: int = 6, hidden_dim: int = 128, heads: int = 3, num_layers: int = 3):
        super().__init__()
        # 128 不能被 3 整除，用 129=43*3 作为中间维，最后投影到 128
        out_per_head = (hidden_dim + heads - 1) // heads  # 43
        mid_dim = out_per_head * heads  # 129
        self.hidden_dim = hidden_dim
        self.heads = heads
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
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.leaky_relu(x, 0.2)
        return x


class DTODRLLocationEncoder(nn.Module):
    """论文: olocations = (EFT, f) 两维, MLP hidden 256"""
    def __init__(self, in_dim: int = 2, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class DTODRLBackbone(nn.Module):
    """
    DTODRL 论文专用: GAT + Location MLP
    节点 6 维, 位置 2 维 (EFT, f)
    """
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
        self.gat_hidden = gat_hidden

    def encode_nodes(self, obs: dict) -> Tuple[Tensor, Optional[Tensor]]:
        device = obs["nodes_C"].device
        # 论文 6 维
        node_features = torch.stack([
            obs["nodes_C"].float(),
            obs["nodes_D"].float(),
            obs["nodes_in_degree"].float(),
            obs["nodes_out_degree"].float(),
            obs["nodes_loc"].float(),
            obs["nodes_ava"].float(),
        ], dim=-1)

        adj = obs["adj"]
        edge_attr = obs["edge_attr"]

        edge_index, _, batch, _ = build_graph_inputs_from_adj(adj, edge_attr)
        edge_index = edge_index.to(device)

        if node_features.dim() == 3:
            B, N, Fdim = node_features.shape
            node_features = node_features.reshape(B * N, Fdim)

        node_embs = self.gat_encoder(node_features.to(device), edge_index, batch)
        return node_embs, batch

    def encode_locations(self, obs: dict) -> Tensor:
        # 论文: (EFT, f) 两维
        loc_features = torch.stack([
            obs["loc_min_processor_EAT"].float(),
            obs["loc_cpu_speed"].float(),
        ], dim=-1)
        if loc_features.dim() == 3:
            B, L, _ = loc_features.shape
            loc_flat = loc_features.reshape(B * L, -1)
            loc_embs = self.location_encoder(loc_flat).reshape(B, L, -1)
        else:
            loc_embs = self.location_encoder(loc_features)
        return loc_embs

    def pool_nodes(self, node_embs: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        if batch is not None:
            return global_mean_pool(node_embs, batch)
        return node_embs.mean(dim=0, keepdim=True)

    def pool_locations(self, loc_embs: Tensor) -> Tensor:
        if loc_embs.dim() == 3:
            return loc_embs.mean(dim=1)
        return loc_embs.mean(dim=0, keepdim=True)
