"""
DTODRL 论文原方法 Baseline

论文: Dependent Task Offloading in Edge Computing Using GNN and Deep Reinforcement Learning
(Cao & Deng, arXiv:2303.17100)

与论文完全一致:
- Node Head: 统一 state 向量 → 一次输出 N 维 logits
- Location Head: loc_head(state)，与 Node Head 共享同一 state
- 仅 node mask，无 pair mask
- State = concat(flatten(stask), slocations)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DTODRLActor(nn.Module):
    """
    论文 DTODRL Actor: 统一 state → Node Head(N 维) + Location Head(K 维)
    - Node Head: loc_head(state) → N logits
    - Location Head: loc_head(state) → K logits（与 Node Head 共享同一 state）
    - 仅 node mask，无 pair mask
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden: int = 256,
        max_N: int = 200,
        max_K: int = 16,
    ):
        super().__init__()
        self.max_N = max_N
        self.max_K = max_K
        # Node Head: 论文 统一 state → N 维, state = flatten(node_embs) + flatten(loc_K_embs)
        # in = max_N*d + max_K*d
        self.node_head = nn.Sequential(
            nn.Linear(max_N * hidden_dim + max_K * hidden_dim, mlp_hidden),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, max_N),
        )
        # Location Head: 论文 loc_logits = loc_head(state)，与 Node Head 共享同一 state
        # in = max_N*d + max_K*d（与 node_head 相同）
        self.loc_head = nn.Sequential(
            nn.Linear(max_N * hidden_dim + max_K * hidden_dim, mlp_hidden),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, max_K),
        )

    def forward(
        self,
        node_embs: torch.Tensor,
        loc_K_embs: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        node_embs: (B, N, d)
        loc_K_embs: (B, K, d) 论文 slocations 的 K 个位置 embedding
        """
        squeeze_batch = False
        if node_embs.dim() == 2:
            node_embs = node_embs.unsqueeze(0)
            loc_K_embs = loc_K_embs.unsqueeze(0)
            squeeze_batch = True

        B, N, d = node_embs.shape
        _, K, _ = loc_K_embs.shape

        if N > self.max_N or K > self.max_K:
            raise ValueError(f"N={N} K={K} exceeds max_N={self.max_N} max_K={self.max_K}")

        # ----- 论文: State = concat(flatten(stask), slocations) -----
        node_flat = node_embs.reshape(B, -1)  # (B, N*d)
        loc_flat = loc_K_embs.reshape(B, -1)   # (B, K*d)

        # Pad to max dims for fixed Linear
        if N < self.max_N:
            node_flat = F.pad(node_flat, (0, (self.max_N - N) * d), value=0)
        if K < self.max_K:
            loc_flat = F.pad(loc_flat, (0, (self.max_K - K) * d), value=0)

        state = torch.cat([node_flat, loc_flat], dim=-1)  # (B, max_N*d + max_K*d)

        # ----- Node Head: 统一 state → N 维 logits -----
        node_scores_full = self.node_head(state)  # (B, max_N)
        node_scores = node_scores_full[:, :N]     # (B, N)

        # ----- Location Head: 论文 loc_logits = loc_head(state) -----
        loc_scores_full = self.loc_head(state)  # (B, max_K)
        loc_scores = loc_scores_full[:, :K]        # (B, K)

        # ----- 论文仅 node mask，无 pair mask -----
        if action_masks is not None:
            if action_masks.dim() == 1:
                action_masks = action_masks.unsqueeze(0)
            pair_mask = action_masks.reshape(B, N, K)
            node_mask = pair_mask.any(dim=2)
        else:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=node_embs.device)

        # ----- π(anode | s)，仅对 node 做 mask -----
        node_scores_masked = node_scores.masked_fill(~node_mask, -1e9)
        log_pi_node = F.log_softmax(node_scores_masked, dim=1)

        # ----- π(alocation | s)，论文无 location mask -----
        log_pi_loc = F.log_softmax(loc_scores, dim=1)

        # ----- log π(a) = log π(anode) + log π(alocation)，invalid node 已通过 log_pi_node=-inf 排除 -----
        log_joint = log_pi_node.unsqueeze(2) + log_pi_loc.unsqueeze(1)
        logits = log_joint.reshape(B, N * K)

        if squeeze_batch:
            return logits.squeeze(0)
        return logits
