import torch.nn as nn
import torch


class JointActor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embs, loc_embs, graph_emb, loc_global_emb):
        """
        node_embs:
            单环境: (N, d)
            batch:   (B, N, d)

        loc_embs:
            单环境: (N, K, d)
            batch:   (B, N, K, d)

        graph_emb:
            单环境: (1, d)
            batch:   (B, d)

        loc_global_emb:
            单环境: (1, d)
            batch:   (B, d)
        """
        squeeze_batch = False

        if node_embs.dim() == 2:
            node_embs = node_embs.unsqueeze(0)      # (1, N, d)
            loc_embs = loc_embs.unsqueeze(0)        # (1, N, K, d)
            if graph_emb.dim() == 2 and graph_emb.shape[0] == 1:
                pass
            elif graph_emb.dim() == 1:
                graph_emb = graph_emb.unsqueeze(0)
            if loc_global_emb.dim() == 2 and loc_global_emb.shape[0] == 1:
                pass
            elif loc_global_emb.dim() == 1:
                loc_global_emb = loc_global_emb.unsqueeze(0)
            squeeze_batch = True

        if node_embs.dim() != 3:
            raise ValueError(f"node_embs should be (B,N,d), got {tuple(node_embs.shape)}")
        if loc_embs.dim() != 4:
            raise ValueError(f"loc_embs should be (B,N,K,d), got {tuple(loc_embs.shape)}")
        if graph_emb.dim() != 2:
            raise ValueError(f"graph_emb should be (B,d), got {tuple(graph_emb.shape)}")
        if loc_global_emb.dim() != 2:
            raise ValueError(f"loc_global_emb should be (B,d), got {tuple(loc_global_emb.shape)}")

        B, N, d = node_embs.shape
        B2, N2, K, d2 = loc_embs.shape
        B3, d3 = graph_emb.shape
        B4, d4 = loc_global_emb.shape

        if not (B == B2 == B3 == B4 and N == N2 and d == d2 == d3 == d4):
            raise ValueError(
                f"shape mismatch: node_embs={tuple(node_embs.shape)}, "
                f"loc_embs={tuple(loc_embs.shape)}, "
                f"graph_emb={tuple(graph_emb.shape)}, "
                f"loc_global_emb={tuple(loc_global_emb.shape)}"
            )

        graph_ctx = graph_emb.unsqueeze(1).expand(B, N, d)
        loc_ctx = loc_global_emb.unsqueeze(1).expand(B, N, d)
        node_ctx = torch.cat([node_embs, graph_ctx, loc_ctx], dim=-1)   # (B,N,3d)

        node_ctx = node_ctx.unsqueeze(2).expand(B, N, K, 3 * d)         # (B,N,K,3d)
        pair_feat = torch.cat([node_ctx, loc_embs], dim=-1)             # (B,N,K,4d)

        scores = self.score_mlp(pair_feat).squeeze(-1)                  # (B,N,K)
        logits = scores.reshape(B, N * K)                               # (B, N*K)

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
        """
        支持：
            graph_emb:      (1, d) or (B, d)
            loc_global_emb: (1, d) or (B, d)
        输出：
            value:          (1,) or (B,)
        """
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

        x = torch.cat([graph_emb, loc_global_emb], dim=-1)   # (B, 2d)
        value = self.value_net(x).squeeze(-1)                # (B,)
        return value