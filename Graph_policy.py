# Graph_policy.py
import math
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
from torch_geometric.utils import softmax

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


class CPTransformerConv(TransformerConv):
    """
    在 attention 中加入 CP bias: α_ij = softmax((QK^T)/√d + λ·CP_norm(j))
    """

    def __init__(self, *args, cp_lambda: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cp_lambda = nn.Parameter(torch.tensor(cp_lambda, dtype=torch.float32))

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
        cp_norm: Optional[Tensor] = None,
        return_attention_weights=None,
    ):
        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x = (x, x)
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)
        kwargs = {"query": query, "key": key, "value": value, "edge_attr": edge_attr}
        if cp_norm is not None:
            kwargs["cp_norm"] = cp_norm
            kwargs["target_index"] = edge_index[1]
        out = self.propagate(edge_index, **kwargs)
        alpha = self._alpha
        self._alpha = None
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1)).sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r
        if isinstance(return_attention_weights, bool) and alpha is not None:
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
        return out

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: Optional[Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
        cp_norm: Optional[Tensor] = None,
        target_index: Optional[Tensor] = None,
    ) -> Tensor:
        if self.lin_edge is not None and edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        if cp_norm is not None and target_index is not None:
            cp_bias = self.cp_lambda * cp_norm[target_index]
            alpha = alpha + cp_bias
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = value_j
        if edge_attr is not None:
            out = out + edge_attr
        out = out * alpha.view(-1, self.heads, 1)
        return out


def _make_conv_layer(use_cp: bool, **kwargs) -> nn.Module:
    if use_cp:
        return CPTransformerConv(**kwargs, cp_lambda=1.0)
    return TransformerConv(**kwargs)


class NodeEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int,
        edge_dim: int = 1,
        num_layers: int = 3,
        use_cp: bool = False,
    ):
        super().__init__()
        self.use_cp = use_cp
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        out_per_head = hidden_dim // heads
        self.layers = nn.ModuleList()

        self.layers.append(
            _make_conv_layer(
                use_cp,
                in_channels=in_dim,
                out_channels=out_per_head,
                heads=heads,
                dropout=0.1,
                edge_dim=edge_dim,
                beta=True,
                concat=True,
            )
        )
        for _ in range(num_layers - 2):
            self.layers.append(
                _make_conv_layer(
                    use_cp,
                    in_channels=hidden_dim,
                    out_channels=out_per_head,
                    heads=heads,
                    dropout=0.1,
                    edge_dim=edge_dim,
                    beta=True,
                    concat=True,
                )
            )
        if num_layers > 1:
            self.layers.append(
                _make_conv_layer(
                    use_cp,
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
        pool: bool = True,
        cp_norm: Optional[Tensor] = None,
    ) -> Tensor:
        x = node_features
        for i, conv in enumerate(self.layers):
            if self.use_cp and cp_norm is not None and isinstance(conv, CPTransformerConv):
                x = conv(x, edge_index, edge_attr=edge_attr, cp_norm=cp_norm)
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        x = self.layer_norm(x)
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
    节点特征: 论文 6 维 + 可选 CP 维 (use_cp 时 7 维)
    """
    def __init__(
            self,
            node_feature_dim: int = 6,
            location_feature_dim: int = 3,
            hidden_dim: int = 108,
            gat_heads: int = 4,
            gat_layers: int = 3,
            use_cp: bool = False,
    ):
        super().__init__()
        self.use_cp = use_cp
        in_dim = node_feature_dim + (1 if use_cp else 0)

        self.node_encoder = NodeEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            heads=gat_heads,
            num_layers=gat_layers,
            use_cp=use_cp,
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

        # 论文 6 维: Ci, Di, in_degree, out_degree, loci, avai [+ nodes_cp]
        node_features = torch.stack([
            obs["nodes_C"].float(),
            obs["nodes_D"].float(),
            obs["nodes_in_degree"].float(),
            obs["nodes_out_degree"].float(),
            obs["nodes_loc"].float(),
            obs["nodes_ava"].float(),
        ], dim=-1)
        if self.use_cp and "nodes_cp" in obs:
            cp = obs["nodes_cp"].float()
            if cp.dim() == 1:
                cp = cp.unsqueeze(-1)
            node_features = torch.cat([node_features, cp], dim=-1)

        adj = obs["adj"]
        edge_attr_matrix = obs["edge_attr"]

        edge_index, edge_attr, batch, num_nodes = build_graph_inputs_from_adj(
            adj, edge_attr_matrix
        )

        if node_features.dim() == 3:
            B, N, Fdim = node_features.shape
            node_features = node_features.reshape(B * N, Fdim)

        edge_attr = torch.log1p(edge_attr)

        cp_norm = None
        if self.use_cp and "nodes_cp" in obs:
            cp_norm = obs["nodes_cp"].float().to(device)
            if cp_norm.dim() == 2:
                cp_norm = cp_norm.reshape(-1)

        node_embs = self.node_encoder(
            node_features=node_features.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            batch=batch,
            pool=False,
            cp_norm=cp_norm,
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


