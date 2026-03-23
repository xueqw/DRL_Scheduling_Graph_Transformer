# Graph_policy.py
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import TransformerConv

# ----------------------------
# Utils: adj -> edge_index
# ----------------------------
def build_graph_inputs_from_adj(
    adj: Tensor,
    edge_attr_matrix: Tensor,
) -> Tuple[Tensor, Tensor, Optional[Tensor], int]:
    """
    输入:
      adj: (N,N) 或 (B,N,N)
      edge_attr_matrix: (N,N) 或 (B,N,N)

    输出:
      edge_index: (2,E)
      edge_attr: (E,1)
      batch: (num_nodes,) or None
      num_nodes: int
    """
    if not torch.is_tensor(adj):
        adj = torch.as_tensor(adj)
    if not torch.is_tensor(edge_attr_matrix):
        edge_attr_matrix = torch.as_tensor(edge_attr_matrix)

    device = adj.device

    if adj.dim() == 2:
        N = adj.size(0)
        src, dst = (adj > 0).nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0).long().to(device)

        eattr = edge_attr_matrix[src, dst].float().unsqueeze(-1).to(device)

        num_nodes = N
        batch = None
        return edge_index, eattr, batch, num_nodes

    elif adj.dim() == 3:
        B, N, _ = adj.shape
        edge_indices = []
        edge_attrs = []

        for b in range(B):
            a = adj[b]
            ea = edge_attr_matrix[b]
            src, dst = (a > 0).nonzero(as_tuple=True)

            ei = torch.stack([src, dst], dim=0).long()
            ei = ei + b * N
            edge_indices.append(ei)

            eattr = ea[src, dst].float().unsqueeze(-1)
            edge_attrs.append(eattr)

        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=1).to(device)
            edge_attr = torch.cat(edge_attrs, dim=0).to(device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)

        num_nodes = B * N
        batch = torch.arange(B, device=device).repeat_interleave(N)
        return edge_index, edge_attr, batch, num_nodes

    else:
        raise ValueError(f"adj must be (N,N) or (B,N,N), got shape={tuple(adj.shape)}")

from torch_geometric.nn import TransformerConv, global_mean_pool

class NodeEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int,
        edge_dim: int = 1,
        num_layers: int = 3
    ):
        super().__init__()

        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        out_per_head = hidden_dim // heads

        self.layers = nn.ModuleList()

        # 第一层
        self.layers.append(
            TransformerConv(
                in_channels=in_dim,
                out_channels=out_per_head,
                heads=heads,
                dropout=0.1,
                edge_dim=edge_dim,
                beta=True,
                concat=True,
            )
        )

        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=out_per_head,
                    heads=heads,
                    dropout=0.1,
                    edge_dim=edge_dim,
                    beta=True,
                    concat=True,
                )
            )

        # 最后一层
        if num_layers > 1:
            self.layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=1,
                    dropout=0.1,
                    edge_dim=edge_dim,
                    beta=True,
                    concat=True,
                )
            )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Optional[Tensor] = None,
        pool: bool = True
    ) -> Tensor:
        x = node_features

        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.layers) - 1:
                x = F.relu(x)

        x = self.layer_norm(x)

        # 全局就上pool 走全局平均池化
        # two-stage 直接返回embedding
        if pool:
            if batch is not None:
                x = global_mean_pool(x, batch)
            else:
                x = x.mean(dim=0, keepdim=True)

        return x


class LocationEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()

        layers = []
        cur = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(cur, hidden_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
            cur = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, loc_features: Tensor) -> Tensor:
        x = self.mlp(loc_features)
        x = self.layer_norm(x)
        return x


class GraphDictFeaturesExtractor(BaseFeaturesExtractor):
    """
    只给baseline用，节点特征按论文 6 维
    """
    def __init__(
        self,
        observation_space,
        node_feature_dim: int = 6,
        location_feature_dim: int = 3,
        hidden_dim: int = 108,
        gat_heads: int = 4,
        gat_layers: int = 3,
        **kwargs
    ):
        super().__init__(observation_space, features_dim=hidden_dim)

        self.node_encoder = NodeEncoder(
            in_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            heads=gat_heads,
            num_layers=gat_layers,
        )

        self.location_encoder = LocationEncoder(
            in_dim=location_feature_dim,
            hidden_dim=hidden_dim
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, obs):
        device = obs["nodes_C"].device

        # 论文 6 维: Ci, Di, in_degree, out_degree, loci, avai
        node_features = torch.stack([
            obs["nodes_C"].float(),
            obs["nodes_D"].float(),
            obs["nodes_in_degree"].float(),
            obs["nodes_out_degree"].float(),
            obs["nodes_loc"].float(),
            obs["nodes_ava"].float(),
        ], dim=-1)

        adj = obs["adj"]
        edge_attr_matrix = obs["edge_attr"]

        edge_index, edge_attr, batch, num_nodes = build_graph_inputs_from_adj(
            adj, edge_attr_matrix
        )

        if node_features.dim() == 3:
            B, N, Fdim = node_features.shape
            node_features = node_features.reshape(B * N, Fdim)

        # 建议做个边特征归一化，避免 bits 数值太大
        edge_attr = torch.log1p(edge_attr)

        graph_emb = self.node_encoder(
            node_features=node_features.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            batch=batch,
            pool=True,
        )

        loc_features = torch.stack([
            obs["loc_cpu_speed"],
            obs["loc_min_processor_EAT"],
            obs["loc_num_processor"].float()
        ], dim=-1)

        if loc_features.dim() == 3:
            B, L, Fdim = loc_features.shape
            loc_flat = loc_features.reshape(B * L, Fdim)
            loc_encoded = self.location_encoder(loc_flat).reshape(B, L, -1).mean(dim=1)
            loc_emb = loc_encoded
        else:
            loc_emb = self.location_encoder(loc_features).mean(dim=0, keepdim=True)

        if graph_emb.shape[0] != loc_emb.shape[0]:
            if graph_emb.shape[0] == 1:
                graph_emb = graph_emb.expand(loc_emb.shape[0], -1)
            elif loc_emb.shape[0] == 1:
                loc_emb = loc_emb.expand(graph_emb.shape[0], -1)

        latent = self.fusion_layer(torch.cat([graph_emb, loc_emb], dim=-1))

        return latent

class GraphBackbone(nn.Module):
    """
    灵活化接口
    能直接抽节点的embedding
    节点特征按论文 6 维: Ci, Di, in_degree, out_degree, loci, avai
    """
    def __init__(
            self,
            node_feature_dim: int = 6,
            location_feature_dim: int = 3,
            hidden_dim: int = 108,
            gat_heads: int = 4,
            gat_layers: int = 3,
    ):
        super().__init__()

        self.node_encoder = NodeEncoder(
            in_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            heads=gat_heads,
            num_layers=gat_layers,
        )

        self.location_encoder = LocationEncoder(
            in_dim=location_feature_dim,
            hidden_dim=hidden_dim
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def encode_nodes(self, obs):
        device = obs["nodes_C"].device

        # 论文 6 维: Ci, Di, in_degree, out_degree, loci, avai
        node_features = torch.stack([
            obs["nodes_C"].float(),
            obs["nodes_D"].float(),
            obs["nodes_in_degree"].float(),
            obs["nodes_out_degree"].float(),
            obs["nodes_loc"].float(),
            obs["nodes_ava"].float(),
        ], dim=-1)

        adj = obs["adj"]
        edge_attr_matrix = obs["edge_attr"]

        edge_index, edge_attr, batch, num_nodes = build_graph_inputs_from_adj(
            adj, edge_attr_matrix
        )

        # 节点输入特征
        if node_features.dim() == 3:
            B, N, Fdim = node_features.shape
            node_features = node_features.reshape(B * N, Fdim)

        # 数据压缩
        # 边特征
        edge_attr = torch.log1p(edge_attr)

        # 保持节点embedding 不走平均池化 方便做出节点选择
        node_embs = self.node_encoder(
            node_features=node_features.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            batch=batch,
            pool=False,
        )

        return node_embs, batch

    def encode_locations(self, obs):
        loc_features = torch.stack([
            obs["loc_cpu_speed"],
            obs["loc_min_processor_EAT"],
            obs["loc_num_processor"].float(),
        ], dim=-1)

        if loc_features.dim() == 3:
            B, L, Fdim = loc_features.shape
            loc_flat = loc_features.reshape(B * L, Fdim)
            loc_embs = self.location_encoder(loc_flat).reshape(B, L, -1)
        else:
            loc_embs = self.location_encoder(loc_features)

        return loc_embs

    def pool_nodes(self, node_embs, batch=None):
        if batch is not None:
            return global_mean_pool(node_embs, batch)
        return node_embs.mean(dim=0, keepdim=True)

    def pool_locations(self, loc_embs):
        if loc_embs.dim() == 3:
            return loc_embs.mean(dim=1)
        return loc_embs.mean(dim=0, keepdim=True)

    def forward_all(self, obs):
        node_embs, batch = self.encode_nodes(obs)
        loc_embs = self.encode_locations(obs)

        graph_emb = self.pool_nodes(node_embs, batch)
        loc_global_emb = self.pool_locations(loc_embs)

        latent = self.fusion_layer(torch.cat([graph_emb, loc_global_emb], dim=-1))

        return {
            "node_embs": node_embs,
            "loc_embs": loc_embs,
            "graph_emb": graph_emb,
            "loc_global_emb": loc_global_emb,
            "latent": latent,
            "batch": batch,
        }


