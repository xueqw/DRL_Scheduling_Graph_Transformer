import copy
import os
import time
import json
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Tuple, List

import gymnasium
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import tensorboard
import torch
from gymnasium import Env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from DTO_env import DTOEnv

from DTO_scheduler import (
    DTOScheduler,
    Location,
    Processor,
    ExecutionModel,
    UploadModel,
    TransmissionModel,
    Node
)

from dag_generator import generate_multi_ue_dag, DAGCase, make_dag_case

from Graph_policy import GraphDictFeaturesExtractor

from joint_maskable_policy import JointMaskablePolicy
from two_stage_maskable_policy import TwoStageMaskablePolicy
from dtodrl_maskable_policy import DTODRLMaskablePolicy


def _debug_log(hypothesis_id: str, location: str, message: str, data: Dict) -> None:
    # region agent log
    open("/opt/cursor/logs/debug.log", "a", encoding="utf-8").write(
        json.dumps(
            {
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            },
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )
    # endregion


class PrintEpisodeReturnCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_r = info["episode"]["r"]
                ep_l = info["episode"]["l"]
                print(f"[EP END] return={ep_r:.6f}, len={ep_l}")
        return True


@dataclass
class TrainConfig:
    runtime: str = "DTO_DRL"
    log_dir: str = "./runs"
    seed: int = 0
    n_envs: int = 1 # 同时推进的环境数量
    use_subproc: bool = False   # 多线程/单线程训练

    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 512
    gamma: float = 0.99
    # 限制新旧策略差距
    clip_range: float = 0.2
    # 训练策略的随机性（统一三种方法）
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 记录GAT, MLP处理过后的latent维度
    features_dim: int = 108

    # DTODRL 专用: 预训练 GAT 路径；若加载预训练则默认冻结 encoder
    dtodrl_pretrained_gat: Optional[str] = None
    dtodrl_freeze_pretrained_gat: bool = True

    # Critical Path 特征: GraphBackbone 使用 CP 归一化 + attention bias
    use_cp: bool = False
    use_vecnormalize: bool = False

@dataclass
class DAGConfig:
    """
    仿真环境配置
    """
    ue_numbers = 3
    es_numbers = 2
    f_ue = 1e9  # 1 GHz
    f_es = 10e9  # 10 GHz
    tr_ue_es = 50e6 # ue_es传输速度
    tr_es_es = 200e6    # es_es传输速度
    # 奖励函数: baseline (local oracle, no scale) | improved (greedy oracle + scale)
    reward_oracle: str = "greedy"   # "local" | "greedy"
    reward_scale: bool = True

class DTODRLTrainer:
    def __init__(self,
                 config: TrainConfig,
                 env_controller: Callable[[], "DTOEnv"],
                 eval_env_controller: Optional[Callable[[], "DTOEnv"]] = None,
                 ):

        self.config = config
        self.env_controller = env_controller
        self.eval_env_controller = eval_env_controller

        # 给文件一个适合用来分别的时间戳
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_dic = os.path.join(
            config.log_dir,
            f"{config.runtime}_{ts}"
        )
        os.makedirs(self.run_dic, exist_ok=True)

        self.vec_env: Optional[VecEnv] = None
        self.evaluation_vec_env: Optional[VecEnv] = None
        self.model: Optional[MaskablePPO] = None

    # use_subprocess选择开启多线程
    def _make_vec_env(self, controller, n_envs: int, seed: int, use_subprocess: bool):
        # 为一个batch建立一个矩阵 同时推进step 储存多个environment

        def make_thunk(rank: int):
            def _init():
                # 调用时创建DAG 创建scheduler, env实例
                env = controller()
                # 兼容gym和gymnasium的seed写法
                try:
                    env.reset(seed=seed + rank)
                except TypeError:
                    if hasattr(env, "seed"):
                        env.seed(seed + rank)

                env = ActionMasker(env, lambda e: e.action_masks())
                env = Monitor(env)

                return env

            return _init

        if n_envs == 1:
            return DummyVecEnv([make_thunk(0)])

        thunks = [make_thunk(i) for i in range(n_envs)]

        if use_subprocess:
            # 开多线程
            return SubprocVecEnv(thunks)
        else:
            # 单线程
            return DummyVecEnv(thunks)

    def build_vec_env(self):
        self.vec_env = self._make_vec_env(
            controller=self.env_controller,
            n_envs=self.config.n_envs,
            seed=self.config.seed,
            use_subprocess=self.config.use_subproc,
        )

        # 用于评估的环境 不更新 仅前向
        self.evaluation_vec_env = self._make_vec_env(
            controller=self.env_controller,
            n_envs=1,
            seed=self.config.seed + 10000,
            use_subprocess=False,
        )

        if self.config.use_vecnormalize:
            self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            self.evaluation_vec_env = VecNormalize(
                self.evaluation_vec_env,
                training=False,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
            )

    def build_model(self, model_type: str = "joint"):
        """
        build model可控
        走SB3或者手写都ok
        """
        if self.vec_env is None:
            raise RuntimeError("先调用build_vec_env()")

        if model_type == "joint":
            policy_kwargs = dict(
                hidden_dim=self.config.features_dim,
                gat_heads=4,
                gat_layers=3,
                use_cp=self.config.use_cp,
            )
            self.model = MaskablePPO(
                policy=JointMaskablePolicy,
                env=self.vec_env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=policy_kwargs,
                tensorboard_log=self.run_dic,
                verbose=1,
                seed=self.config.seed,
                device=self.config.device,
            )

        elif model_type == "dtodrl":
            # 论文 DTODRL 原方法: GAT + 2D location
            policy_kwargs = dict(
                gat_hidden=128,
                gat_heads=3,
                gat_layers=3,
                mlp_hidden=256,
                pretrained_gat_path=self.config.dtodrl_pretrained_gat,
                freeze_pretrained_gat=self.config.dtodrl_freeze_pretrained_gat,
            )
            self.model = MaskablePPO(
                policy=DTODRLMaskablePolicy,
                env=self.vec_env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=policy_kwargs,
                tensorboard_log=self.run_dic,
                verbose=1,
                seed=self.config.seed,
                device=self.config.device,
            )

        elif model_type == "dtodrl_tf":
            # 横向对比: DTODRL 双头 + TransformerConv（与 Joint/Two-Stage 共用图编码）
            policy_kwargs = dict(
                use_transformer_backbone=True,
                hidden_dim=self.config.features_dim,
                gat_heads=4,
                gat_layers=3,
                mlp_hidden=256,
                use_cp=self.config.use_cp,
            )
            self.model = MaskablePPO(
                policy=DTODRLMaskablePolicy,
                env=self.vec_env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=policy_kwargs,
                tensorboard_log=self.run_dic,
                verbose=1,
                seed=self.config.seed,
                device=self.config.device,
            )

        elif model_type == "two_stage":
            # Two-Stage 方法: Stage1 选 Node, Stage2 选 Location
            # π(a) = π₁(i|s) * π₂(j|i,r)
            policy_kwargs = dict(
                hidden_dim=self.config.features_dim,
                gat_heads=4,
                gat_layers=3,
                use_cp=self.config.use_cp,
            )
            self.model = MaskablePPO(
                policy=TwoStageMaskablePolicy,
                env=self.vec_env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=policy_kwargs,
                tensorboard_log=self.run_dic,
                verbose=1,
                seed=self.config.seed,
                device=self.config.device,
            )

        elif model_type == "baseline":
            policy_kwargs = dict(
                features_extractor_class=GraphDictFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=self.config.features_dim),
                net_arch=dict(
                    pi=[256, 256],
                    vf=[256, 256],
                )
            )
            self.model = MaskablePPO(
                policy="MultiInputPolicy",
                env=self.vec_env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=policy_kwargs,
                tensorboard_log=self.run_dic,
                verbose=1,
                seed=self.config.seed,
                device=self.config.device,
            )

        else:
            raise ValueError(
                f"Unknown model_type={model_type}. Use 'joint' / 'two_stage' / 'dtodrl' / 'dtodrl_tf' / 'baseline'"
            )

    def train(self, model_type: str = "joint") -> str:
        if self.vec_env is None or self.evaluation_vec_env is None:
            self.build_vec_env()

        if self.model is None:
            self.build_model(model_type=model_type)

        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=True,
            tb_log_name="tb",
            # callback=PrintEpisodeReturnCallback(),
        )

        save_path = os.path.join(self.run_dic, "final_model.zip")
        self.model.save(save_path)
        # 保存 VecNormalize，评估时需加载以保持 obs 归一化一致
        norm_path = os.path.join(self.run_dic, "vecnormalize.pkl")
        if isinstance(self.vec_env, VecNormalize):
            self.vec_env.save(norm_path)

        return save_path

def build_env_from_dag_case(
        case: DAGCase,
        ue_number: int,
        es_number: int,
        f_ue: float,
        f_es: float,
        es_processors: int,
        tr_ue_es: float,
        tr_es_es: float,
        *,
        reward_oracle: str = "greedy",
        reward_scale: bool = True,
):
    """
    独立建立env
    """

    # 预防污染aft eat deepcopy一下
    nodes = copy.deepcopy(case.nodes)
    edges_data = copy.deepcopy(case.edges_data)

    locations = []
    # UE本地
    for ue_id in range(ue_number):
        locations.append(
            Location(
                id=0,
                ue_id=ue_id,
                cpu_speed=f_ue,
                processors=[Processor(id=0)]
            )
        )
    # ES
    for es_id in range(1, es_number + 1):
        locations.append(
            Location(
                id=es_id,
                ue_id=None,
                cpu_speed=f_es,
                processors=[Processor(id=p) for p in range(es_processors)]
            )
        )

    # Model
    exec_model = ExecutionModel()
    upload_model = UploadModel(transmission_rate=tr_ue_es, local=0)
    trans_model = TransmissionModel(
        ue_es_transmission_rate=tr_ue_es,
        es_es_transmission_rate=tr_es_es,
    )

    # Scheduler
    scheduler = DTOScheduler(
        nodes=nodes,
        edges_data=edges_data,
        locations=locations,
        exec_model=exec_model,
        upload_model=upload_model,
        trans_model=trans_model,
        end_nodes=case.end_nodes,
        download_nodes=case.download_nodes,
        ue_number=ue_number,
        es_number=es_number,
    )

    # Env
    return DTOEnv(scheduler, reward_oracle=reward_oracle, reward_scale=reward_scale)

def make_dto_env_controller(
        ue_number: int,
        es_number: int,
        n_compute_nodes_per_ue: int,
        start_count_max: int,
        f_ue: float,
        f_es: float,
        es_processors: int,
        tr_ue_es: float,
        tr_es_es: float,
        seed0: int = 0,
        *,
        reward_oracle: str = "greedy",
        reward_scale: bool = True,
):
    # 以seed0为基准 创建 (seed0+i) 的controller
    counter = {"i": 0}

    def _controller():
        index = counter["i"]
        counter["i"] += 1

        case = make_dag_case(
            ue_number=ue_number,
            n_compute_nodes_per_ue=n_compute_nodes_per_ue,
            start_count_max=start_count_max,
            seed=seed0 + index,
        )

        return build_env_from_dag_case(
            case=case,
            ue_number=ue_number,
            es_number=es_number,
            f_ue=f_ue,
            f_es=f_es,
            es_processors=es_processors,
            tr_ue_es=tr_ue_es,
            tr_es_es=tr_es_es,
            reward_oracle=reward_oracle,
            reward_scale=reward_scale,
        )
    return _controller

def run_rl_episode(env, model: MaskablePPO, deterministic: bool = True):
    """
    评估网络（单 env，obs 来自 env.reset/step）
    """
    obs, _ = env.reset()
    steps = 0
    step_infos = []
    policy_step_stats = []
    base_env = _get_base_env(env)

    while not env.done():
        masks = env.action_masks()
        policy_stats = _collect_policy_step_stats(model, obs, masks, base_env)
        if policy_stats:
            policy_step_stats.append(policy_stats)
        action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)

        if isinstance(action, np.ndarray):
            action = int(action.item())  # 或 action[0]

        obs, reward, terminated, truncated, info = env.step(action)
        step_info = info.get("step_info")
        if step_info is not None:
            step_infos.append(step_info)

        steps += 1

    ue_finish = list(base_env.scheduler.download_EAT.values())
    mean_aft = float(np.mean(ue_finish))
    makespan = float(np.max(ue_finish))
    episode_stats = _summarize_offload(step_infos)
    episode_stats.update(_summarize_policy_step_stats(policy_step_stats))
    return mean_aft, makespan, steps, episode_stats


def _get_base_env(env):
    """Unwrap to get DTOEnv (has scheduler)"""
    while hasattr(env, "env"):
        env = env.env
    return env


def _summarize_offload(step_infos) -> Dict[str, float]:
    if not step_infos:
        return {"es_ratio": 0.0, "local_ratio": 0.0}

    total = len(step_infos)
    local_count = sum(
        1
        for step_info in step_infos
        if getattr(step_info, "chosen_loc_ue_id", None) is not None
        or getattr(step_info, "chosen_loc_id", None) == 0
    )
    es_count = total - local_count
    return {
        "es_ratio": float(es_count) / float(total),
        "local_ratio": float(local_count) / float(total),
    }


def _extract_distribution_probs(distribution) -> Optional[torch.Tensor]:
    base_distribution = getattr(distribution, "distribution", None)
    if base_distribution is not None and hasattr(base_distribution, "probs"):
        return base_distribution.probs
    if hasattr(distribution, "probs"):
        return distribution.probs
    return None


def _collect_policy_step_stats(
    model: MaskablePPO,
    obs,
    action_masks,
    base_env,
) -> Dict[str, float]:
    try:
        policy = model.policy
        mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=policy.device)
        distribution = policy.get_distribution(obs, action_masks=mask_tensor)
    except Exception:
        return {}

    ready_count = len(base_env.ready_nodes())
    if ready_count == 0:
        return {}

    if hasattr(distribution, "dist_loc"):
        loc_probs = distribution.dist_loc.probs
        if loc_probs.dim() > 1:
            loc_probs = loc_probs[0]
        local_mass = float(loc_probs[0].item())
        es_mass = float(loc_probs[1:].sum().item())
        local_argmax = float(torch.argmax(loc_probs).item() == 0)
        max_loc_prob = torch.max(loc_probs)
        loc_tie = float(((loc_probs - max_loc_prob).abs() <= 1e-6).sum().item() > 1)
        return {
            "policy_local_mass": local_mass,
            "policy_es_mass": es_mass,
            "policy_local_pref": local_mass,
            "policy_es_pref": es_mass,
            "policy_local_argmax_share": local_argmax,
            "policy_es_argmax_share": 1.0 - local_argmax,
            "policy_loc_tie_share": loc_tie,
        }

    probs = _extract_distribution_probs(distribution)
    if probs is None:
        return {}
    if probs.dim() > 1:
        probs = probs[0]

    num_locations = base_env.es_numbers + 1
    pair_probs = probs.reshape(base_env.N, num_locations)
    pair_mask = mask_tensor.reshape(base_env.N, num_locations)
    ready_rows = pair_mask.any(dim=1)
    if not ready_rows.any():
        return {}

    ready_pair_probs = pair_probs[ready_rows]
    row_mass = ready_pair_probs.sum(dim=1).clamp_min(1e-9)
    local_pref = ready_pair_probs[:, 0] / row_mass
    local_argmax = (ready_pair_probs.argmax(dim=1) == 0).float()
    row_max = ready_pair_probs.max(dim=1, keepdim=True).values
    tie_rows = ((ready_pair_probs - row_max).abs() <= 1e-6).sum(dim=1) > 1

    local_mass = float(ready_pair_probs[:, 0].sum().item())
    es_mass = float(ready_pair_probs[:, 1:].sum().item())
    local_pref_avg = float(local_pref.mean().item())
    local_argmax_share = float(local_argmax.mean().item())
    return {
        "policy_local_mass": local_mass,
        "policy_es_mass": es_mass,
        "policy_local_pref": local_pref_avg,
        "policy_es_pref": 1.0 - local_pref_avg,
        "policy_local_argmax_share": local_argmax_share,
        "policy_es_argmax_share": 1.0 - local_argmax_share,
        "policy_loc_tie_share": float(tie_rows.float().mean().item()),
    }


def _append_metric_lists(metric_lists: Dict[str, List[float]], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        metric_lists.setdefault(key, []).append(float(value))


def _summarize_metric_lists(metric_lists: Dict[str, List[float]]) -> Dict[str, float]:
    summary = {}
    for key, values in metric_lists.items():
        summary[f"{key}_avg"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def _summarize_policy_step_stats(policy_step_stats: List[Dict[str, float]]) -> Dict[str, float]:
    if not policy_step_stats:
        return {}

    metric_lists: Dict[str, List[float]] = {}
    for metrics in policy_step_stats:
        _append_metric_lists(metric_lists, metrics)
    return _summarize_metric_lists(metric_lists)


def run_rl_episode_vec(venv: VecEnv, model: MaskablePPO, deterministic: bool = True):
    """
    评估网络（VecEnv，支持 VecNormalize 归一化）
    """
    obs = venv.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    steps = 0
    reward_sum = 0.0
    dones = [False]
    step_infos = []
    policy_step_stats = []
    last_info = None
    base_env = _get_base_env(venv.envs[0])
    # region agent log
    _debug_log(
        "A",
        "final_training.py:run_rl_episode_vec:entry",
        "RL episode vec entry",
        {
            "deterministic": bool(deterministic),
            "ready_count": len(base_env.ready_nodes()),
            "vec_type": type(venv).__name__,
            "has_vecnormalize": bool(hasattr(venv, "obs_rms")),
        },
    )
    # endregion

    while not dones[0]:
        masks = venv.envs[0].action_masks()
        policy_stats = _collect_policy_step_stats(model, obs, masks, base_env)
        if policy_stats:
            policy_step_stats.append(policy_stats)
        action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
        if isinstance(action, np.ndarray):
            action = action.reshape(-1)
        if steps < 3:
            # region agent log
            _debug_log(
                "A",
                "final_training.py:run_rl_episode_vec:pre_step",
                "RL step action/mask snapshot",
                {
                    "step": steps,
                    "valid_action_count": int(np.sum(np.asarray(masks, dtype=np.int32))),
                    "mask_size": int(len(masks)),
                    "action": np.asarray(action).astype(int).tolist(),
                },
            )
            # endregion
        obs, rewards, dones, infos = venv.step(action)
        reward_sum += float(np.asarray(rewards).reshape(-1)[0])
        if infos:
            last_info = infos[0]
            step_info = last_info.get("step_info")
            if step_info is not None:
                step_infos.append(step_info)
        if isinstance(obs, tuple):
            obs = obs[0]
        steps += 1

    if last_info and "step_info" in last_info:
        ue_finish = list(last_info["step_info"].makespan_by_ue.values())
    else:
        base_env = _get_base_env(venv.envs[0])
        ue_finish = list(base_env.scheduler.download_EAT.values())  # DTOEnv.scheduler
    mean_aft = float(np.mean(ue_finish))
    makespan = float(np.max(ue_finish))
    episode_stats = _summarize_offload(step_infos)
    episode_stats.update(_summarize_policy_step_stats(policy_step_stats))
    # region agent log
    _debug_log(
        "C",
        "final_training.py:run_rl_episode_vec:exit",
        "RL episode vec exit",
        {
            "steps": steps,
            "reward_sum": reward_sum,
            "mean_aft": mean_aft,
            "makespan": makespan,
            "es_ratio": episode_stats.get("es_ratio"),
            "local_ratio": episode_stats.get("local_ratio"),
        },
    )
    # endregion
    return mean_aft, makespan, steps, episode_stats

def run_baseline_episode(env, node_rule: "str"):
    """
        node_rule: "topo" | "spt" | "lpt"
        - topo: ready里选最小id
        - sjf: ready里选C最小（短作业优先）
        - ljf: ready里选C最大（长作业优先）
        """
    obs, _ = env.reset()
    steps = 0
    node_id = None
    # region agent log
    _debug_log(
        "E",
        "final_training.py:run_baseline_episode:entry",
        "Baseline episode entry",
        {
            "node_rule": node_rule,
            "ready_count": len(env.ready_nodes()),
        },
    )
    # endregion

    while not env.done():
        ready = env.ready_nodes()
        if len(ready) == 0:
            raise RuntimeError("ready_list为空，但episode未结束，依赖更新可能有问题")

        if node_rule == "topo":
            node_id = min(ready)
        elif node_rule == "sjf":
            node_id = min(ready, key=lambda nid: env.scheduler.nodes[nid].C)
        elif node_rule == "ljf":
            node_id = max(ready, key=lambda nid: env.scheduler.nodes[nid].C)

        if steps < 3:
            # region agent log
            _debug_log(
                "E",
                "final_training.py:run_baseline_episode:pre_step",
                "Baseline step snapshot",
                {
                    "step": steps,
                    "node_rule": node_rule,
                    "chosen_node_id": int(node_id),
                    "ready_count": len(ready),
                },
            )
            # endregion
        env.step_greedy(node_id)
        steps += 1

    ue_finish = list(env.scheduler.download_EAT.values())
    mean_aft = float(np.mean(ue_finish))
    makespan = float(np.max(ue_finish))
    # region agent log
    _debug_log(
        "E",
        "final_training.py:run_baseline_episode:exit",
        "Baseline episode exit",
        {
            "node_rule": node_rule,
            "steps": steps,
            "mean_aft": mean_aft,
            "makespan": makespan,
        },
    )
    # endregion
    return mean_aft, makespan, steps


def _build_eval_vec_env(case: DAGCase, dag_cfg: DAGConfig, norm_path: str):
    env = build_env_from_dag_case(
        case=case,
        ue_number=dag_cfg.ue_numbers,
        es_number=dag_cfg.es_numbers,
        f_ue=dag_cfg.f_ue,
        f_es=dag_cfg.f_es,
        es_processors=4,
        tr_ue_es=dag_cfg.tr_ue_es,
        tr_es_es=dag_cfg.tr_es_es,
        reward_oracle=getattr(dag_cfg, "reward_oracle", "greedy"),
        reward_scale=getattr(dag_cfg, "reward_scale", True),
    )
    env = ActionMasker(env, lambda e: e.action_masks())
    venv = DummyVecEnv([lambda e=env: e])
    if os.path.exists(norm_path):
        venv = VecNormalize.load(norm_path, venv)
        venv.training = False
        # region agent log
        _debug_log(
            "B",
            "final_training.py:_build_eval_vec_env",
            "Loaded VecNormalize for eval",
            {"norm_path": norm_path, "exists": True},
        )
        # endregion
    else:
        # region agent log
        _debug_log(
            "B",
            "final_training.py:_build_eval_vec_env",
            "VecNormalize missing for eval",
            {"norm_path": norm_path, "exists": False},
        )
        # endregion
    return venv


def evaluate_trained_model(
    model: MaskablePPO,
    dag_cfg: DAGConfig,
    norm_path: str,
    *,
    n_eval_seeds: int = 30,
    deterministic: bool = True,
    seed_base: int = 1000,
) -> Dict[str, float]:
    mean_aft_list: List[float] = []
    makespan_list: List[float] = []
    metric_lists: Dict[str, List[float]] = {}

    for s in range(n_eval_seeds):
        case = make_dag_case(
            ue_number=dag_cfg.ue_numbers,
            n_compute_nodes_per_ue=20,
            start_count_max=3,
            seed=seed_base + s,
        )
        venv = _build_eval_vec_env(case, dag_cfg, norm_path)
        mean_aft, makespan, _, episode_stats = run_rl_episode_vec(
            venv,
            model,
            deterministic=deterministic,
        )
        mean_aft_list.append(mean_aft)
        makespan_list.append(makespan)
        _append_metric_lists(metric_lists, episode_stats)
        if hasattr(venv, "close"):
            venv.close()

    summary = {
        "mean_AFT_avg": float(np.mean(mean_aft_list)),
        "makespan_avg": float(np.mean(makespan_list)),
        "mean_AFT_std": float(np.std(mean_aft_list)),
        "makespan_std": float(np.std(makespan_list)),
    }
    summary.update(_summarize_metric_lists(metric_lists))
    return summary


def run_comparison_experiment(
    train_cfg: TrainConfig,
    dag_cfg: DAGConfig,
    n_eval_episodes: int = 30,
    n_eval_seeds: int = 50,
    methods: Optional[List[str]] = None,
    results_prefix: str = "comparison",
):
    """
    多方法对比实验:
    - dtodrl: 论文 DTODRL 原方法 (Node+Location 独立双头)
    - joint: 单阶段 (node,loc) 联合打分
    - two_stage: 两阶段 先选 Node 再选 Location
    """
    if methods is None:
        methods = ["dtodrl", "joint", "two_stage"]
    rl_train_controller = make_dto_env_controller(
        ue_number=dag_cfg.ue_numbers,
        es_number=dag_cfg.es_numbers,
        n_compute_nodes_per_ue=20,
        start_count_max=3,
        f_ue=dag_cfg.f_ue,
        f_es=dag_cfg.f_es,
        es_processors=4,
        tr_ue_es=dag_cfg.tr_ue_es,
        tr_es_es=dag_cfg.tr_es_es,
        reward_oracle=getattr(dag_cfg, "reward_oracle", "greedy"),
        reward_scale=getattr(dag_cfg, "reward_scale", True),
    )

    results = {}

    for model_type in methods:
        print(f"\n{'='*60}")
        print(f"训练 {model_type.upper()} 方法...")
        print(f"{'='*60}")

        cfg = copy.deepcopy(train_cfg)
        cfg.runtime = f"DTO_DRL_{model_type}"
        # 横向对比：所有图编码器端到端训练，不冻结
        if model_type in ("dtodrl", "dtodrl_tf"):
            cfg.dtodrl_pretrained_gat = None
            cfg.dtodrl_freeze_pretrained_gat = False
        trainer = DTODRLTrainer(config=cfg, env_controller=rl_train_controller)
        trainer.build_vec_env()
        trainer.build_model(model_type=model_type)
        save_path = trainer.train(model_type=model_type)
        model = trainer.model
        run_dic = trainer.run_dic
        norm_path = os.path.join(run_dic, "vecnormalize.pkl")
        print(f"  -> 模型已保存: {save_path}")

        # 评估（使用 VecNormalize 保持 obs 归一化与训练一致）
        det_summary = evaluate_trained_model(
            model,
            dag_cfg,
            norm_path,
            n_eval_seeds=n_eval_seeds,
            deterministic=True,
        )
        stochastic_summary = evaluate_trained_model(
            model,
            dag_cfg,
            norm_path,
            n_eval_seeds=n_eval_seeds,
            deterministic=False,
        )

        results[model_type] = dict(det_summary)
        results[model_type].update(
            {f"stochastic_{key}": value for key, value in stochastic_summary.items()}
        )
        print(f"  -> meanAFT(avg)={results[model_type]['mean_AFT_avg']:.6f}, "
              f"makespan(avg)={results[model_type]['makespan_avg']:.6f}, "
              f"es_ratio(avg)={results[model_type]['es_ratio_avg']:.3f}, "
              f"local_ratio(avg)={results[model_type]['local_ratio_avg']:.3f}")
        print(f"     stochastic es/local = "
              f"{results[model_type]['stochastic_es_ratio_avg']:.3f}/"
              f"{results[model_type]['stochastic_local_ratio_avg']:.3f}")
        print(f"     policy mass(es/local) = "
              f"{results[model_type].get('policy_es_mass_avg', 0.0):.3f}/"
              f"{results[model_type].get('policy_local_mass_avg', 0.0):.3f}")
        print(f"     per-ready pref(es/local) = "
              f"{results[model_type].get('policy_es_pref_avg', 0.0):.3f}/"
              f"{results[model_type].get('policy_local_pref_avg', 0.0):.3f}")
        print(f"     loc tie share = "
              f"{results[model_type].get('policy_loc_tie_share_avg', 0.0):.3f}")

    # 打印对比表
    title = "横向对比 (TransformerConv)" if "dtodrl_tf" in methods else "DTODRL vs Joint vs Two-Stage 对比结果"
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"{'方法':<12} | {'meanAFT(avg)':>12} ± {'std':>8} | {'makespan(avg)':>12} ± {'std':>8}")
    print("-" * 70)
    for name in methods:
        r = results[name]
        print(f"{name:<12} | {r['mean_AFT_avg']:>12.4f} ± {r['mean_AFT_std']:>8.4f} | "
              f"{r['makespan_avg']:>12.4f} ± {r['makespan_std']:>8.4f}")
        print(f"{'':12} | det es/local = {r['es_ratio_avg']:.3f}/{r['local_ratio_avg']:.3f}")
        print(f"{'':12} | stoch es/local = {r['stochastic_es_ratio_avg']:.3f}/{r['stochastic_local_ratio_avg']:.3f}")
        print(f"{'':12} | policy mass(es/local) = "
              f"{r.get('policy_es_mass_avg', 0.0):.3f}/{r.get('policy_local_mass_avg', 0.0):.3f}")
        print(f"{'':12} | per-ready pref(es/local) = "
              f"{r.get('policy_es_pref_avg', 0.0):.3f}/{r.get('policy_local_pref_avg', 0.0):.3f}")
        print(f"{'':12} | loc tie share = {r.get('policy_loc_tie_share_avg', 0.0):.3f}")
    print(f"{'='*70}\n")

    # 保存结果到 JSON
    import json
    os.makedirs(train_cfg.log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    results_path = os.path.join(train_cfg.log_dir, f"{results_prefix}_results_{ts}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到 {results_path}\n")

    return results


def run_comparison_transformer_experiment(
    train_cfg: TrainConfig,
    dag_cfg: DAGConfig,
    n_eval_episodes: int = 10,
    n_eval_seeds: int = 30,
):
    """
    横向对比实验: 三种方法统一使用 TransformerConv 图编码，全部端到端训练不冻结
    - dtodrl_tf: DTODRL 双头 + TransformerConv（不冻结）
    - joint: 联合打分 + TransformerConv（不冻结）
    - two_stage: 两阶段 + TransformerConv（不冻结）
    """
    print("运行横向对比实验 (统一 TransformerConv，图编码器全部端到端训练不冻结)...")
    return run_comparison_experiment(
        train_cfg,
        dag_cfg,
        n_eval_episodes=n_eval_episodes,
        n_eval_seeds=n_eval_seeds,
        methods=["dtodrl_tf", "joint", "two_stage"],
        results_prefix="comparison_transformer",
    )


def run_reward_comparison_experiment(
    train_cfg: TrainConfig,
    dag_cfg: DAGConfig,
    model_type: str = "joint",
    n_eval_seeds: int = 30,
):
    """
    奖励函数对比实验：baseline (local oracle, no scale) vs improved (greedy oracle + scale)
    默认用 joint 方法做对比，可指定 model_type
    """
    reward_configs = [
        ("baseline", "local", False),
        ("improved", "greedy", True),
    ]

    results = {}
    for name, oracle, scale in reward_configs:
        print(f"\n{'='*60}")
        print(f"奖励配置: {name} (oracle={oracle}, scale={scale})")
        print(f"训练 {model_type.upper()} ...")
        print(f"{'='*60}")

        cfg = copy.deepcopy(dag_cfg)
        cfg.reward_oracle = oracle
        cfg.reward_scale = scale

        rl_train_controller = make_dto_env_controller(
            ue_number=cfg.ue_numbers,
            es_number=cfg.es_numbers,
            n_compute_nodes_per_ue=20,
            start_count_max=3,
            f_ue=cfg.f_ue,
            f_es=cfg.f_es,
            es_processors=4,
            tr_ue_es=cfg.tr_ue_es,
            tr_es_es=cfg.tr_es_es,
            reward_oracle=oracle,
            reward_scale=scale,
        )

        train_cfg_copy = copy.deepcopy(train_cfg)
        train_cfg_copy.runtime = f"DTO_DRL_{model_type}_reward_{name}"
        trainer = DTODRLTrainer(config=train_cfg_copy, env_controller=rl_train_controller)
        trainer.build_vec_env()
        trainer.build_model(model_type=model_type)
        save_path = trainer.train(model_type=model_type)
        model = trainer.model
        norm_path = os.path.join(trainer.run_dic, "vecnormalize.pkl")
        print(f"  -> 模型已保存: {save_path}")

        det_summary = evaluate_trained_model(
            model,
            cfg,
            norm_path,
            n_eval_seeds=n_eval_seeds,
            deterministic=True,
        )
        stochastic_summary = evaluate_trained_model(
            model,
            cfg,
            norm_path,
            n_eval_seeds=n_eval_seeds,
            deterministic=False,
        )

        results[name] = dict(det_summary)
        results[name].update(
            {f"stochastic_{key}": value for key, value in stochastic_summary.items()}
        )
        print(f"  -> meanAFT(avg)={results[name]['mean_AFT_avg']:.6f}, "
              f"makespan(avg)={results[name]['makespan_avg']:.6f}, "
              f"es_ratio(avg)={results[name]['es_ratio_avg']:.3f}, "
              f"local_ratio(avg)={results[name]['local_ratio_avg']:.3f}")
        print(f"     stochastic es/local = "
              f"{results[name]['stochastic_es_ratio_avg']:.3f}/"
              f"{results[name]['stochastic_local_ratio_avg']:.3f}")
        print(f"     policy mass(es/local) = "
              f"{results[name].get('policy_es_mass_avg', 0.0):.3f}/"
              f"{results[name].get('policy_local_mass_avg', 0.0):.3f}")
        print(f"     loc tie share = "
              f"{results[name].get('policy_loc_tie_share_avg', 0.0):.3f}")

    print(f"\n{'='*70}")
    print("奖励函数对比结果 (baseline vs improved)")
    print(f"{'='*70}")
    print(f"{'奖励配置':<12} | {'meanAFT(avg)':>12} ± {'std':>8} | {'makespan(avg)':>12} ± {'std':>8}")
    print("-" * 70)
    for name in ["baseline", "improved"]:
        r = results[name]
        print(f"{'':12} | det es/local = {r['es_ratio_avg']:.3f}/{r['local_ratio_avg']:.3f}")
        print(f"{'':12} | stoch es/local = {r['stochastic_es_ratio_avg']:.3f}/{r['stochastic_local_ratio_avg']:.3f}")
        print(f"{'':12} | policy mass(es/local) = "
              f"{r.get('policy_es_mass_avg', 0.0):.3f}/{r.get('policy_local_mass_avg', 0.0):.3f}")
        print(f"{name:<12} | {r['mean_AFT_avg']:>12.4f} ± {r['mean_AFT_std']:>8.4f} | "
              f"{r['makespan_avg']:>12.4f} ± {r['makespan_std']:>8.4f}")
    print(f"{'='*70}\n")

    os.makedirs(train_cfg.log_dir, exist_ok=True)
    import json
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(train_cfg.log_dir, f"reward_comparison_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到 {path}\n")
    return results


def log_baselines_to_tb(env_controller, log_dir, n_episodes, seed):
    writer = SummaryWriter(log_dir=log_dir)

    for rule in ["topo", "sjf", "ljf"]:
        mean_aft_list, makespan_list = [], []

        for ep in range(n_episodes):
            env = env_controller()

            mean_aft, makespan, steps = run_baseline_episode(env, rule)

            mean_aft_list.append(mean_aft)
            makespan_list.append(makespan)

            writer.add_scalar(f"baseline/{rule}/mean_AFT", mean_aft, ep)
            writer.add_scalar(f"baseline/{rule}/makespan", makespan, ep)

        writer.add_scalar(f"baseline/{rule}/mean_AFT_avg",
                          float(np.mean(mean_aft_list)),
                          n_episodes)

        writer.add_scalar(f"baseline/{rule}/makespan_avg",
                          float(np.mean(makespan_list)),
                          n_episodes)

    writer.close()


if __name__ == "__main__":
    import sys

    train_cfg = TrainConfig(
        seed=0,
        n_envs=8,
        use_subproc=False,
        total_timesteps=50000,
    )

    dag_cfg = DAGConfig()

    # 通过命令行参数选择: joint | two_stage | dtodrl | dtodrl_tf | comparison | comparison_transformer | reward_comparison
    mode = sys.argv[1] if len(sys.argv) > 1 else "joint"

    if mode == "comparison":
        # DTODRL vs Joint vs Two-Stage 对比实验（各自原图编码）
        print("运行 DTODRL vs Joint vs Two-Stage 对比实验...")
        run_comparison_experiment(
            train_cfg, dag_cfg,
            n_eval_episodes=10, n_eval_seeds=30,
            methods=["dtodrl", "joint", "two_stage"],
        )
    elif mode == "comparison_transformer":
        # 横向对比: 三种方法统一 TransformerConv 图编码
        run_comparison_transformer_experiment(
            train_cfg, dag_cfg,
            n_eval_episodes=10, n_eval_seeds=30,
        )
    elif mode == "reward_comparison":
        # 奖励函数对比: baseline (local) vs improved (greedy + scale)
        print("运行奖励函数对比实验 (baseline vs improved)...")
        run_reward_comparison_experiment(
            train_cfg, dag_cfg,
            model_type=sys.argv[2] if len(sys.argv) > 2 else "joint",
            n_eval_seeds=30,
        )
    else:
        # 单方法训练
        model_type = mode if mode in ("joint", "two_stage", "dtodrl", "dtodrl_tf") else "joint"
        rl_train_controller = make_dto_env_controller(
            ue_number=dag_cfg.ue_numbers,
            es_number=dag_cfg.es_numbers,
            n_compute_nodes_per_ue=20,
            start_count_max=3,
            f_ue=dag_cfg.f_ue,
            f_es=dag_cfg.f_es,
            es_processors=4,
            tr_ue_es=dag_cfg.tr_ue_es,
            tr_es_es=dag_cfg.tr_es_es,
            reward_oracle=getattr(dag_cfg, "reward_oracle", "greedy"),
            reward_scale=getattr(dag_cfg, "reward_scale", True),
        )

        trainer = DTODRLTrainer(config=train_cfg, env_controller=rl_train_controller)
        model_path = trainer.train(model_type=model_type)
        print(f"训练完成，模型已保存到: {model_path}")

        # 拿到训练完的模型 (可用于后续评估)
        model = trainer.model
        norm_path = os.path.join(trainer.run_dic, "vecnormalize.pkl")
        det_summary = evaluate_trained_model(
            model,
            dag_cfg,
            norm_path,
            n_eval_seeds=10,
            deterministic=True,
        )
        stochastic_summary = evaluate_trained_model(
            model,
            dag_cfg,
            norm_path,
            n_eval_seeds=10,
            deterministic=False,
        )
        print(f"[DIAG] det es/local = {det_summary.get('es_ratio_avg', 0.0):.3f}/"
              f"{det_summary.get('local_ratio_avg', 0.0):.3f}")
        print(f"[DIAG] stoch es/local = {stochastic_summary.get('es_ratio_avg', 0.0):.3f}/"
              f"{stochastic_summary.get('local_ratio_avg', 0.0):.3f}")
        print(f"[DIAG] policy mass(es/local) = {det_summary.get('policy_es_mass_avg', 0.0):.3f}/"
              f"{det_summary.get('policy_local_mass_avg', 0.0):.3f}")
        print(f"[DIAG] per-ready pref(es/local) = {det_summary.get('policy_es_pref_avg', 0.0):.3f}/"
              f"{det_summary.get('policy_local_pref_avg', 0.0):.3f}")
        print(f"[DIAG] loc tie share = {det_summary.get('policy_loc_tie_share_avg', 0.0):.3f}")

        import json
        ts = time.strftime("%Y%m%d-%H%M%S")
        diag_path = os.path.join(trainer.run_dic, f"policy_diagnosis_{ts}.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "deterministic": det_summary,
                    "stochastic": stochastic_summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[DIAG] saved to {diag_path}")

    # ========== baseline vs. PPO+GAT 评估模块 (取消注释后启用) ==========

    # seeds = list(range(50))
    # # 存储baseline的结果
    # baseline_result = {r: [] for r in ["topo", "sjf", "ljf"]}
    # # 存储rl的结果
    # rl_result = []
    #
    # for i, s in enumerate(seeds):
    #     case = make_dag_case(
    #         ue_number=dag_cfg.ue_numbers,
    #         n_compute_nodes_per_ue=50,
    #         start_count_max=3,
    #         seed=s,
    #     )
    #
    #     # ---------- baselines ----------
    #     for rule in ["topo", "sjf", "ljf"]:
    #         env_base = build_env_from_dag_case(
    #             case=case,
    #             ue_number=dag_cfg.ue_numbers,
    #             es_number=dag_cfg.es_numbers,
    #             f_ue=dag_cfg.f_ue,
    #             f_es=dag_cfg.f_es,
    #             es_processors=4,
    #             tr_ue_es=dag_cfg.tr_ue_es,
    #             tr_es_es=dag_cfg.tr_es_es,
    #         )
    #         mean_aft, makespan, _ = run_baseline_episode(env_base, rule)
    #         baseline_result[rule].append((mean_aft, makespan))
    #
    #     # ---------- RL eval ----------
    #     env_rl = build_env_from_dag_case(
    #         case=case,
    #         ue_number=dag_cfg.ue_numbers,
    #         es_number=dag_cfg.es_numbers,
    #         f_ue=dag_cfg.f_ue,
    #         f_es=dag_cfg.f_es,
    #         es_processors=4,
    #         tr_ue_es=dag_cfg.tr_ue_es,
    #         tr_es_es=dag_cfg.tr_es_es,
    #     )
    #     env_rl = Monitor(env_rl)
    #     env_rl = ActionMasker(env_rl, lambda e: e.action_masks())
    #
    #     mean_aft, makespan, steps = run_rl_episode(env_rl, model, deterministic=True)
    #     rl_result.append((mean_aft, makespan))
    #
    #     if (i + 1) % 10 == 0:
    #         print(f"[EVAL] seed={s}  RL meanAFT={mean_aft:.6f}  makespan={makespan:.6f}  steps={steps}")
    #
    # def summarize(name, arr):
    #     mean_aft = np.mean([x[0] for x in arr])
    #     makespan = np.mean([x[1] for x in arr])
    #     print(f"{name:12s} | meanAFT(avg)={mean_aft:.6f} | makespan(avg)={makespan:.6f}")
    #
    # summarize("RL(PPO+GAT)", rl_result)
    # for rule in ["topo", "sjf", "ljf"]:
    #     summarize(f"baseline-{rule}", baseline_result[rule])
