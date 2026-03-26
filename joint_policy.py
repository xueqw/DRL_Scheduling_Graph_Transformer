import torch
import torch.nn as nn


class JointActor(nn.Module):
    def __init__(self, hidden_dim: int, raw_loc_feature_dim: int = 0):
        super().__init__()
        self.raw_loc_feature_dim = raw_loc_feature_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + raw_loc_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embs, loc_embs, loc_raw_features, graph_emb, loc_global_emb):
        """
        node_embs:
            single env: (N, d)
            batch:      (B, N, d)

        loc_embs:
            single env: (N, K, d)
            batch:      (B, N, K, d)

        loc_raw_features:
            single env: (N, K, f)
            batch:      (B, N, K, f)

        graph_emb:
            single env: (1, d)
            batch:      (B, d)

        loc_global_emb:
            single env: (1, d)
            batch:      (B, d)
        """
        squeeze_batch = False

        if node_embs.dim() == 2:
            node_embs = node_embs.unsqueeze(0)
            loc_embs = loc_embs.unsqueeze(0)
            loc_raw_features = loc_raw_features.unsqueeze(0)
            if graph_emb.dim() == 1:
                graph_emb = graph_emb.unsqueeze(0)
            if loc_global_emb.dim() == 1:
                loc_global_emb = loc_global_emb.unsqueeze(0)
            squeeze_batch = True

        if node_embs.dim() != 3:
            raise ValueError(f"node_embs should be (B,N,d), got {tuple(node_embs.shape)}")
        if loc_embs.dim() != 4:
            raise ValueError(f"loc_embs should be (B,N,K,d), got {tuple(loc_embs.shape)}")
        if loc_raw_features.dim() != 4:
            raise ValueError(
                f"loc_raw_features should be (B,N,K,f), got {tuple(loc_raw_features.shape)}"
            )
        if graph_emb.dim() != 2:
            raise ValueError(f"graph_emb should be (B,d), got {tuple(graph_emb.shape)}")
        if loc_global_emb.dim() != 2:
            raise ValueError(f"loc_global_emb should be (B,d), got {tuple(loc_global_emb.shape)}")

        bsz, num_nodes, hidden_dim = node_embs.shape
        bsz2, num_nodes2, num_locations, hidden_dim2 = loc_embs.shape
        bsz3, num_nodes3, num_locations2, raw_dim = loc_raw_features.shape
        bsz4, hidden_dim3 = graph_emb.shape
        bsz5, hidden_dim4 = loc_global_emb.shape

        if not (
            bsz == bsz2 == bsz3 == bsz4 == bsz5
            and num_nodes == num_nodes2 == num_nodes3
            and num_locations == num_locations2
            and hidden_dim == hidden_dim2 == hidden_dim3 == hidden_dim4
        ):
            raise ValueError(
                f"shape mismatch: node_embs={tuple(node_embs.shape)}, "
                f"loc_embs={tuple(loc_embs.shape)}, "
                f"loc_raw_features={tuple(loc_raw_features.shape)}, "
                f"graph_emb={tuple(graph_emb.shape)}, "
                f"loc_global_emb={tuple(loc_global_emb.shape)}"
            )
        if raw_dim != self.raw_loc_feature_dim:
            raise ValueError(
                f"loc_raw_features last dim mismatch: got {raw_dim}, expected {self.raw_loc_feature_dim}"
            )

        graph_ctx = graph_emb.unsqueeze(1).expand(bsz, num_nodes, hidden_dim)
        loc_ctx = loc_global_emb.unsqueeze(1).expand(bsz, num_nodes, hidden_dim)
        node_ctx = torch.cat([node_embs, graph_ctx, loc_ctx], dim=-1)

        node_ctx = node_ctx.unsqueeze(2).expand(bsz, num_nodes, num_locations, 3 * hidden_dim)
        pair_feat = torch.cat([node_ctx, loc_embs, loc_raw_features], dim=-1)

        scores = self.score_mlp(pair_feat).squeeze(-1)
        logits = scores.reshape(bsz, num_nodes * num_locations)

        if squeeze_batch:
            return logits.squeeze(0)
        return logits


class JointCritic(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_emb, loc_global_emb):
        if graph_emb.dim() != 2 or loc_global_emb.dim() != 2:
            raise ValueError(
                f"graph_emb and loc_global_emb must be 2D, got "
                f"{tuple(graph_emb.shape)} and {tuple(loc_global_emb.shape)}"
            )

        if graph_emb.shape != loc_global_emb.shape:
            raise ValueError(
                f"shape mismatch: graph_emb={tuple(graph_emb.shape)}, "
                f"loc_global_emb={tuple(loc_global_emb.shape)}"
            )

        x = torch.cat([graph_emb, loc_global_emb], dim=-1)
        value = self.value_net(x).squeeze(-1)
        return value
