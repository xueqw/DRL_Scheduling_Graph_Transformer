"""
DTODRL 论文原方法 Maskable Policy

论文方法: Node Head + Location Head 独立双头，P(a)=P(node)×P(loc)
与原文完全一致: GAT(128,3头), Location(EFT,f)两维, MLP 256, Tanh
"""
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from dtodrl_backbone import DTODRLBackbone
from Graph_policy import GraphBackbone
from joint_policy import JointCritic
from dtodrl_policy import DTODRLActor, TwoHeadMaskableCategoricalDistribution


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
    DTODRL 论文原方法: 双头独立 (Node Head + Location Head)
    P(anode, alocation) = P(anode) × P(alocation)
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
        max_K: int = 16,
        max_N: int = 200,
        pretrained_gat_path: Optional[str] = None,
        freeze_pretrained_gat: bool = True,
        use_transformer_backbone: bool = False,
        **kwargs,
    ):
        self.dtodrl_max_N = max_N
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
            # 横向对比: 与 Joint/Two-Stage 共用 TransformerConv + 3D location
            hidden = kwargs.get("hidden_dim", 108)
            self.backbone = GraphBackbone(
                node_feature_dim=6,
                location_feature_dim=3,
                hidden_dim=hidden,
                gat_heads=kwargs.get("gat_heads", 4),
                gat_layers=kwargs.get("gat_layers", 3),
            )
            backbone_hidden = hidden
            pretrained_gat_path = None
        else:
            # 论文原方法: GAT + 2D location
            self.backbone = DTODRLBackbone(
                node_feature_dim=6,
                location_feature_dim=2,
                gat_hidden=gat_hidden,
                gat_heads=gat_heads,
                gat_layers=gat_layers,
                mlp_hidden=mlp_hidden,
            )
            backbone_hidden = gat_hidden

        self.actor = DTODRLActor(
            hidden_dim=backbone_hidden,
            mlp_hidden=mlp_hidden,
            max_N=getattr(self, "dtodrl_max_N", 200),
            max_K=max_K,
        )
        self.critic = JointCritic(hidden_dim=backbone_hidden)

        if pretrained_gat_path and not use_transformer_backbone:
            state = torch.load(pretrained_gat_path, map_location="cpu")
            if isinstance(state, dict) and "encoder" in state:
                state = state["encoder"]
            self.backbone.gat_encoder.load_state_dict(state, strict=False)
            if freeze_pretrained_gat:
                for p in self.backbone.gat_encoder.parameters():
                    p.requires_grad = False

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
        if obs["nodes_C"].dim() == 1:
            return node_embs.unsqueeze(0)
        elif obs["nodes_C"].dim() == 2:
            B, N = obs["nodes_C"].shape
            BN, d = node_embs.shape
            if BN != B * N:
                raise ValueError(f"node_embs first dim mismatch: {BN} vs {B*N}")
            return node_embs.reshape(B, N, d)
        else:
            raise ValueError(f"Unexpected obs['nodes_C'] shape: {tuple(obs['nodes_C'].shape)}")

    def _extract_joint_latents(
        self, obs: PyTorchObs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self._to_tensor_obs(obs)
        node_embs_raw, batch = self.backbone.encode_nodes(obs)
        loc_embs_raw = self.backbone.encode_locations(obs)
        graph_emb = self.backbone.pool_nodes(node_embs_raw, batch=batch)
        loc_global_emb = self.backbone.pool_locations(loc_embs_raw)

        node_embs = self._reshape_node_embs(node_embs_raw, obs, batch=batch)
        candidate_loc_embs = self._build_candidate_loc_embs(obs, loc_embs_raw)
        loc_K_embs = self._build_loc_K_embs_for_dtodrl(obs, loc_embs_raw)

        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if loc_global_emb.dim() == 1:
            loc_global_emb = loc_global_emb.unsqueeze(0)

        return node_embs, candidate_loc_embs, graph_emb, loc_global_emb, loc_K_embs

    def _get_action_logits(
        self, obs: PyTorchObs, action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (node_scores_masked, loc_scores) 供双分布使用"""
        node_embs, _, graph_emb, loc_global_emb, loc_K_embs = self._extract_joint_latents(obs)

        if action_masks is not None and not torch.is_tensor(action_masks):
            action_masks = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)

        return self.actor(
            node_embs=node_embs,
            loc_K_embs=loc_K_embs,
            action_masks=action_masks,
        )

    def _get_values(self, obs: PyTorchObs) -> torch.Tensor:
        _, _, graph_emb, loc_global_emb, _ = self._extract_joint_latents(obs)
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
            node_scores_masked=node_scores_masked, loc_scores=loc_scores
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
            node_scores_masked=node_scores_masked, loc_scores=loc_scores
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
            node_scores_masked=node_scores_masked, loc_scores=loc_scores
        )
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        return self._get_values(obs)

    def _build_loc_K_embs_for_dtodrl(
        self,
        obs: Dict[str, torch.Tensor],
        loc_embs: torch.Tensor,
    ) -> torch.Tensor:
        """论文 slocations: K 个候选位置的 embedding, [mean(UE_loc), ES1, ES2, ...]"""
        ue_num = obs["ue_upload_EAT"].shape[-1] if obs["ue_upload_EAT"].dim() > 0 else obs["ue_upload_EAT"].shape[0]
        if loc_embs.dim() == 2:
            local_mean = loc_embs[:ue_num].mean(dim=0, keepdim=True)
            es_embs = loc_embs[ue_num:]
            loc_K = torch.cat([local_mean, es_embs], dim=0)  # (K, d)
            return loc_K.unsqueeze(0)
        else:
            B, L, d = loc_embs.shape
            local_mean = loc_embs[:, :ue_num].mean(dim=1, keepdim=True)  # (B, 1, d)
            es_embs = loc_embs[:, ue_num:]  # (B, M, d)
            loc_K = torch.cat([local_mean, es_embs], dim=1)  # (B, K, d)
            return loc_K

    def _build_candidate_loc_embs(
        self,
        obs: Dict[str, torch.Tensor],
        loc_embs: torch.Tensor,
    ) -> torch.Tensor:
        nodes_ue_id = obs["nodes_ue_id"].long()
        if loc_embs.dim() == 2:
            N = nodes_ue_id.shape[0]
            L, d = loc_embs.shape
            ue_num = obs["ue_upload_EAT"].shape[0]
            es_embs = loc_embs[ue_num:]
            out = []
            for i in range(N):
                ue_id = int(nodes_ue_id[i].item())
                local_emb = loc_embs[ue_id].unsqueeze(0)
                cand_i = torch.cat([local_emb, es_embs], dim=0)
                out.append(cand_i)
            return torch.stack(out, dim=0)

        elif loc_embs.dim() == 3:
            B, L, d = loc_embs.shape
            _, N = nodes_ue_id.shape
            ue_num = obs["ue_upload_EAT"].shape[-1]
            out = []
            for b in range(B):
                es_embs = loc_embs[b, ue_num:]
                per_env = []
                for i in range(N):
                    ue_id = int(nodes_ue_id[b, i].item())
                    local_emb = loc_embs[b, ue_id].unsqueeze(0)
                    cand_i = torch.cat([local_emb, es_embs], dim=0)
                    per_env.append(cand_i)
                out.append(torch.stack(per_env, dim=0))
            return torch.stack(out, dim=0)
        else:
            raise ValueError(f"Unexpected loc_embs shape: {tuple(loc_embs.shape)}")
