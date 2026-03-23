"""
DTODRL 论文原方法 Baseline

论文: Dependent Task Offloading in Edge Computing Using GNN and Deep Reinforcement Learning
(Cao & Deng, arXiv:2303.17100)

与论文完全一致:
- Node Head: 统一 state 向量 → 一次输出 N 维 logits
- Location Head: loc_head(state)，与 Node Head 共享同一 state
- 仅 node mask，无 pair mask
- State = flatten(concat(stask, slocations))，先 concat 再展平
- 两个独立 Categorical: dist_node, dist_loc，采样 node ~ dist_node, loc ~ dist_loc，action = node*K+loc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from sb3_contrib.common.maskable.distributions import MaskableDistribution
from torch.distributions import Categorical


class TwoHeadMaskableCategoricalDistribution(MaskableDistribution):
    """
    两个独立 Categorical: dist_node, dist_loc
    - dist_node = Categorical(logits=node_scores_masked)
    - dist_loc = Categorical(logits=loc_scores)
    - 采样: node ~ dist_node, loc ~ dist_loc, action = node*K + loc
    - log_prob(a) = log_prob_node(a_node) + log_prob_loc(a_loc)
    """

    def __init__(self):
        super().__init__()

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """不使用，logits 由 Actor 直接输出"""
        return nn.Identity()

    def proba_distribution(
        self,
        node_scores_masked: torch.Tensor,
        loc_scores: torch.Tensor,
    ) -> "TwoHeadMaskableCategoricalDistribution":
        if node_scores_masked.dim() == 1:
            node_scores_masked = node_scores_masked.unsqueeze(0)
            loc_scores = loc_scores.unsqueeze(0)
        self.dist_node = Categorical(logits=node_scores_masked)
        self.dist_loc = Categorical(logits=loc_scores)
        self._K = loc_scores.shape[-1]
        return self

    def sample(self) -> torch.Tensor:
        node = self.dist_node.sample()
        loc = self.dist_loc.sample()
        return node * self._K + loc

    def mode(self) -> torch.Tensor:
        node = self.dist_node.logits.argmax(dim=-1)
        loc = self.dist_loc.logits.argmax(dim=-1)
        return node * self._K + loc

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        node = (actions // self._K).long().clamp(0, self.dist_node.logits.shape[-1] - 1)
        loc = (actions % self._K).long().clamp(0, self.dist_loc.logits.shape[-1] - 1)
        return self.dist_node.log_prob(node) + self.dist_loc.log_prob(loc)

    def entropy(self) -> torch.Tensor:
        return self.dist_node.entropy() + self.dist_loc.entropy()

    def apply_masking(self, masks=None) -> None:
        """node mask 已在 Actor 中融入 node_scores_masked，此处无需再处理"""
        pass


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
        # Node Head: 论文 统一 state → N 维, state = flatten(concat(stask, slocations))
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        node_embs: (B, N, d)
        loc_K_embs: (B, K, d) 论文 slocations 的 K 个位置 embedding
        Returns:
            node_scores_masked: (B, N) 已做 node mask
            loc_scores: (B, K)
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

        # ----- 论文 logits 分离: h = f(s), state = flatten(concat(stask, slocations))，先 concat 再展平 -----
        # Pad to max dims
        if N < self.max_N:
            node_embs = F.pad(node_embs, (0, 0, 0, self.max_N - N), value=0)  # (B, max_N, d)
        if K < self.max_K:
            loc_K_embs = F.pad(loc_K_embs, (0, 0, 0, self.max_K - K), value=0)  # (B, max_K, d)
        # 先 concat 再 flatten
        state_seq = torch.cat([node_embs, loc_K_embs], dim=1)  # (B, max_N+max_K, d)
        state = state_seq.reshape(B, -1)  # (B, (max_N+max_K)*d) = (B, max_N*d + max_K*d)

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

        # ----- 返回两个独立分布的参数，不再返回 joint logits -----
        if squeeze_batch:
            return node_scores_masked.squeeze(0), loc_scores.squeeze(0)
        return node_scores_masked, loc_scores
