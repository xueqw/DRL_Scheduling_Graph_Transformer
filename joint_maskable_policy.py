from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from Graph_policy import GraphBackbone
from joint_policy import JointActor, JointCritic


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


class JointMaskablePolicy(MaskableActorCriticPolicy):
    """
    直接用 GraphBackbone + JointActor + JointCritic
    输出联合动作 logits:
        单环境: (N*K,)
        batch: (B, N*K)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        hidden_dim: int = 108,
        gat_heads: int = 4,
        gat_layers: int = 3,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=_IdentityExtractor,
            features_extractor_kwargs={},
            **kwargs,
        )

        self.features_extractor = _IdentityExtractor(observation_space)
        self.mlp_extractor = _DummyMlpExtractor()

        self.backbone = GraphBackbone(
            node_feature_dim=6,
            location_feature_dim=3,
            hidden_dim=hidden_dim,
            gat_heads=gat_heads,
            gat_layers=gat_layers,
        )

        self.actor = JointActor(hidden_dim=hidden_dim)
        self.critic = JointCritic(hidden_dim=hidden_dim)

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
            raise ValueError(f"JointMaskablePolicy expects dict obs, got {type(obs)}")

        out = {}
        for k, v in obs.items():
            if not torch.is_tensor(v):
                v = torch.as_tensor(v, device=self.device)
            else:
                v = v.to(self.device)
            out[k] = v
        return out

    def _reshape_node_embs(
        self,
        node_embs: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        将 GraphBackbone.encode_nodes() 的输出统一整理成:
            (B, N, d)
        """
        # 单环境：obs["nodes_C"] shape = (N,)
        if obs["nodes_C"].dim() == 1:
            if node_embs.dim() != 2:
                raise ValueError(f"Single-env node_embs should be 2D, got {tuple(node_embs.shape)}")
            return node_embs.unsqueeze(0)  # (1, N, d)

        # batch 环境：obs["nodes_C"] shape = (B, N)
        elif obs["nodes_C"].dim() == 2:
            B, N = obs["nodes_C"].shape
            if node_embs.dim() != 2:
                raise ValueError(f"Batched node_embs should be flattened 2D (B*N,d), got {tuple(node_embs.shape)}")

            BN, d = node_embs.shape
            if BN != B * N:
                raise ValueError(f"node_embs first dim mismatch: got {BN}, expected {B*N}")

            return node_embs.reshape(B, N, d)  # 因为 build_graph_inputs_from_adj 按图顺序拼接

        else:
            raise ValueError(f"Unexpected obs['nodes_C'] shape: {tuple(obs['nodes_C'].shape)}")

    def _extract_joint_latents(
        self, obs: PyTorchObs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        统一返回 batch 版：
            node_embs:      (B, N, d)
            loc_embs:       (B, K, d)
            graph_emb:      (B, d)
            loc_global_emb: (B, d)
        """
        obs = self._to_tensor_obs(obs)

        node_embs_raw, batch = self.backbone.encode_nodes(obs)
        loc_embs_raw = self.backbone.encode_locations(obs)
        graph_emb = self.backbone.pool_nodes(node_embs_raw, batch=batch)
        loc_global_emb = self.backbone.pool_locations(loc_embs_raw)

        node_embs = self._reshape_node_embs(node_embs_raw, obs, batch=batch)
        candidate_loc_embs = self._build_candidate_loc_embs(obs, loc_embs_raw)

        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if loc_global_emb.dim() == 1:
            loc_global_emb = loc_global_emb.unsqueeze(0)

        return node_embs, candidate_loc_embs, graph_emb, loc_global_emb

    def _get_action_logits(self, obs: PyTorchObs) -> torch.Tensor:
        node_embs, loc_embs, graph_emb, loc_global_emb = self._extract_joint_latents(obs)

        logits = self.actor(
            node_embs=node_embs,
            loc_embs=loc_embs,
            graph_emb=graph_emb,
            loc_global_emb=loc_global_emb,
        )
        return logits

    def _get_values(self, obs: PyTorchObs) -> torch.Tensor:
        _, _, graph_emb, loc_global_emb = self._extract_joint_latents(obs)
        values = self.critic(graph_emb, loc_global_emb)
        return values

    def forward(
        self,
        obs: PyTorchObs,
        deterministic: bool = False,
        action_masks: Optional[torch.Tensor] = None,
    ):
        logits = self._get_action_logits(obs)
        values = self._get_values(obs)

        distribution = self.action_dist.proba_distribution(action_logits=logits)

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
        logits = self._get_action_logits(obs)
        values = self._get_values(obs)

        distribution = self.action_dist.proba_distribution(action_logits=logits)

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
        logits = self._get_action_logits(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)

        if action_masks is not None:
            distribution.apply_masking(action_masks)

        return distribution

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        return self._get_values(obs)

    def _build_candidate_loc_embs(
            self,
            obs: Dict[str, torch.Tensor],
            loc_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入:
            obs["nodes_ue_id"]:
                单环境: (N,)
                batch:   (B, N)

            loc_embs:
                单环境: (L, d)
                batch:   (B, L, d)

        输出:
            candidate_loc_embs:
                单环境: (N, K, d)
                batch:   (B, N, K, d)

        其中:
            K = es_numbers + 1
            第 0 维是该 node 自己所属 UE 的 local
            第 1..M 维是各个 ES
        """
        nodes_ue_id = obs["nodes_ue_id"].long()

        # 单环境
        if loc_embs.dim() == 2:
            # loc 顺序来自 build_env_from_dag_case:
            # [UE0 local, UE1 local, ..., UE(U-1) local, ES1, ES2, ...]
            N = nodes_ue_id.shape[0]
            L, d = loc_embs.shape
            ue_num = obs["ue_upload_EAT"].shape[0]
            es_num = L - ue_num
            K = es_num + 1

            out = []
            es_embs = loc_embs[ue_num:]  # (M, d)

            for i in range(N):
                ue_id = int(nodes_ue_id[i].item())
                local_emb = loc_embs[ue_id].unsqueeze(0)  # (1, d)
                cand_i = torch.cat([local_emb, es_embs], dim=0)  # (K, d)
                out.append(cand_i)

            return torch.stack(out, dim=0)  # (N, K, d)

        # batch
        elif loc_embs.dim() == 3:
            B, L, d = loc_embs.shape
            _, N = nodes_ue_id.shape

            ue_num = obs["ue_upload_EAT"].shape[-1]
            es_num = L - ue_num
            K = es_num + 1

            out = []
            for b in range(B):
                es_embs = loc_embs[b, ue_num:]  # (M, d)
                per_env = []
                for i in range(N):
                    ue_id = int(nodes_ue_id[b, i].item())
                    local_emb = loc_embs[b, ue_id].unsqueeze(0)  # (1, d)
                    cand_i = torch.cat([local_emb, es_embs], dim=0)  # (K, d)
                    per_env.append(cand_i)
                out.append(torch.stack(per_env, dim=0))  # (N, K, d)

            return torch.stack(out, dim=0)  # (B, N, K, d)

        else:
            raise ValueError(f"Unexpected loc_embs shape: {tuple(loc_embs.shape)}")