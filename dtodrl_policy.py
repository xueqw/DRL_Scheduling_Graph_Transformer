"""
DTODRL paper-aligned actor/distribution helpers.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from torch.distributions import Categorical


class TwoHeadMaskableCategoricalDistribution(MaskableDistribution):
    """
    Two independent categorical distributions:
    - dist_node = Categorical(logits=node_scores_masked)
    - dist_loc = Categorical(logits=loc_scores)
    - sample action as node * K + loc
    """

    def __init__(self):
        super().__init__()

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
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
        # The DTODRL paper applies a mask only on the node head.
        pass

    def actions_from_params(
        self,
        node_scores_masked: torch.Tensor,
        loc_scores: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        self.proba_distribution(node_scores_masked=node_scores_masked, loc_scores=loc_scores)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        node_scores_masked: torch.Tensor,
        loc_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(node_scores_masked=node_scores_masked, loc_scores=loc_scores)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class DTODRLActor(nn.Module):
    """
    Paper-aligned DTODRL actor:
    state = concat(flatten(stask), flatten(slocations))
    Node Head and Location Head share the same state.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        num_locations: int,
        num_user_locations: int,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_locations = num_locations
        self.num_user_locations = num_user_locations
        self.num_action_locations = num_locations - num_user_locations + 1

        state_dim = (num_nodes + num_locations) * hidden_dim

        self.node_head = nn.Sequential(
            nn.Linear(state_dim, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, num_nodes),
        )
        self.loc_head = nn.Sequential(
            nn.Linear(state_dim, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, self.num_action_locations),
        )

    def forward(
        self,
        node_embs: torch.Tensor,
        loc_all_embs: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_batch = False
        if node_embs.dim() == 2:
            node_embs = node_embs.unsqueeze(0)
            loc_all_embs = loc_all_embs.unsqueeze(0)
            squeeze_batch = True

        bsz, num_nodes, _ = node_embs.shape
        _, num_locations, _ = loc_all_embs.shape
        num_action_locations = self.num_action_locations

        if num_nodes != self.num_nodes or num_locations != self.num_locations:
            raise ValueError(
                f"Unexpected DTODRL state shape: nodes={num_nodes}/{self.num_nodes}, "
                f"locations={num_locations}/{self.num_locations}"
            )

        state = torch.cat(
            [node_embs.reshape(bsz, -1), loc_all_embs.reshape(bsz, -1)],
            dim=1,
        )

        node_scores = self.node_head(state)
        loc_scores = self.loc_head(state)

        if action_masks is not None:
            if action_masks.dim() == 1:
                action_masks = action_masks.unsqueeze(0)
            pair_mask = action_masks.reshape(bsz, num_nodes, num_action_locations)
            node_mask = pair_mask.any(dim=2)
        else:
            node_mask = torch.ones(bsz, num_nodes, dtype=torch.bool, device=node_embs.device)

        node_scores_masked = node_scores.masked_fill(~node_mask, -1e9)

        if squeeze_batch:
            return node_scores_masked.squeeze(0), loc_scores.squeeze(0)
        return node_scores_masked, loc_scores


class DTODRLCritic(nn.Module):
    """Use Tanh to match the paper's PPO activation setting."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_emb: torch.Tensor, loc_global_emb: torch.Tensor) -> torch.Tensor:
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
        return self.value_net(x).squeeze(-1)
