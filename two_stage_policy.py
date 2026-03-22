"""
Two-Stage 方法策略模块

将决策分解为两阶段：
  Stage 1: 选择任务 (Node)   π₁(i|s) = Softmax(f₁(s,i))
  Stage 2: 选择资源 (Location) π₂(j|i,r) = Softmax(f₂(i,r,j))
  联合概率: π(a) = π(i,j) = π₁(i|s) * π₂(j|i,r)
  log π(a) = log π₁(i) + log π₂(j|i)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TwoStageActor(nn.Module):
    """
    两阶段打分网络：
    - f₁(s,i): 任务选择得分 -> π₁(i|s)
    - f₂(i,r,j): 资源选择得分 -> π₂(j|i,r)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Stage 1: 任务选择 f₁(s,i), 输入 node_ctx (B,N,3d)
        self.node_score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        # Stage 2: 资源选择 f₂(i,r,j), 输入 (node_i, loc_j, resource_ctx)
        self.loc_score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embs: torch.Tensor,
        loc_embs: torch.Tensor,
        graph_emb: torch.Tensor,
        loc_global_emb: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        输出联合 logits (B, N*K)，满足 log π(a) = log π₁(i) + log π₂(j|i)

        action_masks: (N*K,) 或 (B, N*K)，True 表示 (node_i, loc_j) 合法
        """
        squeeze_batch = False
        if node_embs.dim() == 2:
            node_embs = node_embs.unsqueeze(0)
            loc_embs = loc_embs.unsqueeze(0)
            if graph_emb.dim() == 1:
                graph_emb = graph_emb.unsqueeze(0)
            if loc_global_emb.dim() == 1:
                loc_global_emb = loc_global_emb.unsqueeze(0)
            squeeze_batch = True

        B, N, d = node_embs.shape
        _, _, K, _ = loc_embs.shape

        # ----- Stage 1: f₁(s,i) 任务选择得分 -----
        graph_ctx = graph_emb.unsqueeze(1).expand(B, N, d)
        loc_ctx = loc_global_emb.unsqueeze(1).expand(B, N, d)
        node_ctx = torch.cat([node_embs, graph_ctx, loc_ctx], dim=-1)  # (B,N,3d)
        node_scores = self.node_score_mlp(node_ctx).squeeze(-1)  # (B,N)

        # ----- Stage 2: f₂(i,r,j) 资源选择得分 -----
        node_ctx_exp = node_ctx.unsqueeze(2).expand(B, N, K, 3 * d)
        pair_feat = torch.cat([node_ctx_exp, loc_embs], dim=-1)  # (B,N,K,4d)
        loc_scores = self.loc_score_mlp(pair_feat).squeeze(-1)  # (B,N,K)

        # ----- 解析 action mask -----
        if action_masks is not None:
            if action_masks.dim() == 1:
                action_masks = action_masks.unsqueeze(0)
            if not torch.is_tensor(action_masks):
                action_masks = torch.as_tensor(action_masks, device=node_embs.device)
            # pair_mask[b,i,j] = mask[b, i*K+j]，确保 bool 以支持 ~ 运算符
            pair_mask = action_masks.reshape(B, N, K).bool()
            node_mask = pair_mask.any(dim=2)  # (B,N), 至少有一个 loc 合法的 node
        else:
            pair_mask = torch.ones(B, N, K, dtype=torch.bool, device=node_embs.device)
            node_mask = torch.ones(B, N, dtype=torch.bool, device=node_embs.device)

        # ----- π₁(i|s) = Softmax(f₁) over valid nodes -----
        node_scores_masked = node_scores.masked_fill(~node_mask, -1e9)
        log_pi1 = F.log_softmax(node_scores_masked, dim=1)  # (B,N)

        # ----- π₂(j|i,r) = Softmax(f₂) over valid locs for each i -----
        loc_scores_masked = loc_scores.masked_fill(~pair_mask, -1e9)
        log_pi2 = F.log_softmax(loc_scores_masked, dim=2)  # (B,N,K)

        # ----- log π(a) = log π₁(i) + log π₂(j|i), a = i*K+j -----
        log_joint = log_pi1.unsqueeze(2) + log_pi2  # (B,N,K)
        logits = log_joint.reshape(B, N * K)

        # 无效动作置为极小值，后续 apply_masking 会再次处理
        if action_masks is not None:
            flat_mask = action_masks.reshape(B, N * K).bool()
            logits = logits.masked_fill(~flat_mask, -1e9)

        if squeeze_batch:
            return logits.squeeze(0)
        return logits
