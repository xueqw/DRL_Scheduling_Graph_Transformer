"""
Two-stage policy helpers.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStageActor(nn.Module):
    """
    Factorized policy:
    - stage 1 chooses a node
    - stage 2 chooses a location for that node
    """

    def __init__(self, hidden_dim: int, raw_loc_feature_dim: int = 0):
        super().__init__()
        self.raw_loc_feature_dim = raw_loc_feature_dim
        self.node_score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.loc_score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + raw_loc_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embs: torch.Tensor,
        loc_embs: torch.Tensor,
        loc_raw_features: torch.Tensor,
        graph_emb: torch.Tensor,
        loc_global_emb: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return flattened joint logits with
        log p(node, loc) = log p(node) + log p(loc | node).
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

        bsz, num_nodes, hidden_dim = node_embs.shape
        bsz2, num_nodes2, num_locations, hidden_dim2 = loc_embs.shape
        bsz3, num_nodes3, num_locations2, raw_dim = loc_raw_features.shape

        if not (
            bsz == bsz2 == bsz3
            and num_nodes == num_nodes2 == num_nodes3
            and num_locations == num_locations2
            and hidden_dim == hidden_dim2
        ):
            raise ValueError(
                f"shape mismatch: node_embs={tuple(node_embs.shape)}, "
                f"loc_embs={tuple(loc_embs.shape)}, "
                f"loc_raw_features={tuple(loc_raw_features.shape)}"
            )
        if raw_dim != self.raw_loc_feature_dim:
            raise ValueError(
                f"loc_raw_features last dim mismatch: got {raw_dim}, expected {self.raw_loc_feature_dim}"
            )

        graph_ctx = graph_emb.unsqueeze(1).expand(bsz, num_nodes, hidden_dim)
        loc_ctx = loc_global_emb.unsqueeze(1).expand(bsz, num_nodes, hidden_dim)
        node_ctx = torch.cat([node_embs, graph_ctx, loc_ctx], dim=-1)
        node_scores = self.node_score_mlp(node_ctx).squeeze(-1)

        node_ctx_exp = node_ctx.unsqueeze(2).expand(bsz, num_nodes, num_locations, 3 * hidden_dim)
        pair_feat = torch.cat([node_ctx_exp, loc_embs, loc_raw_features], dim=-1)
        loc_scores = self.loc_score_mlp(pair_feat).squeeze(-1)

        if action_masks is not None:
            if action_masks.dim() == 1:
                action_masks = action_masks.unsqueeze(0)
            pair_mask = action_masks.reshape(bsz, num_nodes, num_locations)
            node_mask = pair_mask.any(dim=2)
        else:
            pair_mask = torch.ones(
                bsz, num_nodes, num_locations, dtype=torch.bool, device=node_embs.device
            )
            node_mask = torch.ones(bsz, num_nodes, dtype=torch.bool, device=node_embs.device)

        node_scores_masked = node_scores.masked_fill(~node_mask, -1e9)
        log_pi1 = F.log_softmax(node_scores_masked, dim=1)

        loc_scores_masked = loc_scores.masked_fill(~pair_mask, -1e9)
        log_pi2 = F.log_softmax(loc_scores_masked, dim=2)

        log_joint = log_pi1.unsqueeze(2) + log_pi2
        logits = log_joint.reshape(bsz, num_nodes * num_locations)

        if action_masks is not None:
            flat_mask = action_masks.reshape(bsz, num_nodes * num_locations)
            logits = logits.masked_fill(~flat_mask, -1e9)

        if squeeze_batch:
            return logits.squeeze(0)
        return logits
