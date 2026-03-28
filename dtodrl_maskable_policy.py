"""
DTODRL paper-aligned maskable PPO policy.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from dtodrl_backbone import DTODRLBackbone
from Graph_policy import GraphBackbone
from dtodrl_policy import DTODRLActor, DTODRLCritic, TwoHeadMaskableCategoricalDistribution


class _IdentityExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__()
        self.observation_space = observation_space
        self.features_dim = 1

    def forward(self, obs):
        return torch.zeros((1, 1), device=next(self.parameters(), torch.tensor(0)).device)


class _DummyMlpExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim_pi = 1
        self.latent_dim_vf = 1

    def forward(self, features):
        return features, features

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return features


class DTODRLMaskablePolicy(MaskableActorCriticPolicy):
    """
    DTODRL with two independent heads:
    P(a_node, a_location) = P(a_node) * P(a_location)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        gat_hidden: int = 128,
        gat_heads: int = 3,
        gat_layers: int = 3,
        mlp_hidden: int = 256,
        pretrained_gat_path: Optional[str] = None,
        freeze_pretrained_gat: bool = True,
        use_transformer_backbone: bool = False,
        hidden_dim: int = 108,
        **kwargs,
    ):
        _use_cp = kwargs.pop("use_cp", False)
        _tf_gat_heads = kwargs.pop("gat_heads", 4)
        _tf_gat_layers = kwargs.pop("gat_layers", 3)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=_IdentityExtractor,
            features_extractor_kwargs={},
            **kwargs,
        )

        self.action_dist = TwoHeadMaskableCategoricalDistribution()
        self.features_extractor = _IdentityExtractor(observation_space)
        self.mlp_extractor = _DummyMlpExtractor()

        if use_transformer_backbone:
            self.backbone = GraphBackbone(
                node_feature_dim=6,
                location_feature_dim=3,
                hidden_dim=hidden_dim,
                gat_heads=_tf_gat_heads,
                gat_layers=_tf_gat_layers,
                use_cp=_use_cp,
            )
            actor_hidden_dim = hidden_dim
        else:
            self.backbone = DTODRLBackbone(
                node_feature_dim=6,
                location_feature_dim=2,
                gat_hidden=gat_hidden,
                gat_heads=gat_heads,
                gat_layers=gat_layers,
                mlp_hidden=mlp_hidden,
            )
            actor_hidden_dim = gat_hidden

        self.actor = DTODRLActor(
            hidden_dim=actor_hidden_dim,
            num_nodes=observation_space["nodes_C"].shape[0],
            num_locations=observation_space["loc_cpu_speed"].shape[0],
            num_user_locations=observation_space["ue_upload_EAT"].shape[0],
            mlp_hidden=mlp_hidden,
        )
        self.critic = DTODRLCritic(hidden_dim=actor_hidden_dim)

        if pretrained_gat_path and not use_transformer_backbone:
            state = torch.load(pretrained_gat_path, map_location="cpu")
            if isinstance(state, dict) and "encoder" in state:
                state = state["encoder"]
            self.backbone.gat_encoder.load_state_dict(state, strict=False)
            if freeze_pretrained_gat:
                for param in self.backbone.gat_encoder.parameters():
                    param.requires_grad = False

        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1.0),
            **self.optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = _DummyMlpExtractor()

    def _to_tensor_obs(self, obs: PyTorchObs) -> Dict[str, torch.Tensor]:
        if not isinstance(obs, dict):
            raise ValueError(f"Policy expects dict obs, got {type(obs)}")
        out = {}
        for key, value in obs.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, device=self.device)
            else:
                value = value.to(self.device)
            out[key] = value
        return out

    def _reshape_node_embs(
        self,
        node_embs: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if obs["nodes_C"].dim() == 1:
            return node_embs.unsqueeze(0)
        if obs["nodes_C"].dim() == 2:
            batch_size, num_nodes = obs["nodes_C"].shape
            flat_nodes, hidden_dim = node_embs.shape
            if flat_nodes != batch_size * num_nodes:
                raise ValueError(f"node_embs first dim mismatch: {flat_nodes} vs {batch_size * num_nodes}")
            return node_embs.reshape(batch_size, num_nodes, hidden_dim)
        raise ValueError(f"Unexpected obs['nodes_C'] shape: {tuple(obs['nodes_C'].shape)}")

    def _reshape_location_embs(
        self,
        loc_embs: torch.Tensor,
        obs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if obs["loc_cpu_speed"].dim() == 1:
            return loc_embs.unsqueeze(0)
        if obs["loc_cpu_speed"].dim() == 2:
            return loc_embs
        raise ValueError(f"Unexpected obs['loc_cpu_speed'] shape: {tuple(obs['loc_cpu_speed'].shape)}")

    def _extract_joint_latents(
        self, obs: PyTorchObs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self._to_tensor_obs(obs)
        node_embs_raw, batch = self.backbone.encode_nodes(obs)
        loc_embs_raw = self.backbone.encode_locations(obs)
        graph_emb = self.backbone.pool_nodes(node_embs_raw, batch=batch)
        loc_global_emb = self.backbone.pool_locations(loc_embs_raw)

        node_embs = self._reshape_node_embs(node_embs_raw, obs, batch=batch)
        loc_embs = self._reshape_location_embs(loc_embs_raw, obs)

        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if loc_global_emb.dim() == 1:
            loc_global_emb = loc_global_emb.unsqueeze(0)

        return node_embs, loc_embs, graph_emb, loc_global_emb

    def _get_action_logits(
        self, obs: PyTorchObs, action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embs, loc_embs, _, _ = self._extract_joint_latents(obs)
        if action_masks is not None and not torch.is_tensor(action_masks):
            action_masks = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)

        return self.actor(
            node_embs=node_embs,
            loc_all_embs=loc_embs,
            action_masks=action_masks,
        )

    def _get_values(self, obs: PyTorchObs) -> torch.Tensor:
        _, _, graph_emb, loc_global_emb = self._extract_joint_latents(obs)
        return self.critic(graph_emb, loc_global_emb)

    def forward(
        self,
        obs: PyTorchObs,
        deterministic: bool = False,
        action_masks: Optional[torch.Tensor] = None,
    ):
        node_scores_masked, loc_scores = self._get_action_logits(obs, action_masks)
        values = self._get_values(obs)

        distribution = self.action_dist.proba_distribution(
            node_scores_masked=node_scores_masked,
            loc_scores=loc_scores,
        )
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ):
        node_scores_masked, loc_scores = self._get_action_logits(obs, action_masks)
        values = self._get_values(obs)

        distribution = self.action_dist.proba_distribution(
            node_scores_masked=node_scores_masked,
            loc_scores=loc_scores,
        )
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(
        self,
        obs: PyTorchObs,
        action_masks: Optional[torch.Tensor] = None,
    ):
        node_scores_masked, loc_scores = self._get_action_logits(obs, action_masks)
        distribution = self.action_dist.proba_distribution(
            node_scores_masked=node_scores_masked,
            loc_scores=loc_scores,
        )
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        return self._get_values(obs)
