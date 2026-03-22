import copy
import os
import time
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
    batch_size: int = 256
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
    reward_oracle: str = "local"   # "local" | "greedy"
    reward_scale: bool = False

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

        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        self.evaluation_vec_env = VecNormalize(self.evaluation_vec_env, training=False, norm_obs=True,
                                               norm_reward=False, clip_obs=10.0)

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
                max_K=16,
                max_N=200,
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
                max_K=16,
                max_N=200,
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
        reward_oracle: str = "local",
        reward_scale: bool = False,
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
        reward_oracle: str = "local",
        reward_scale: bool = False,
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
    支持 wrapped env，通过 _get_base_env 获取 DTOEnv 的 done/scheduler
    """
    base_env = _get_base_env(env)
    obs, _ = env.reset()
    steps = 0
    last_info = None

    while not base_env.done():
        masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)

        if isinstance(action, np.ndarray):
            action = int(action.item())  # 或 action[0]

        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        steps += 1

    # 优先从 step_info 获取，避免 auto-reset 导致 download_EAT 被清零
    if last_info and "step_info" in last_info:
        ue_finish = list(last_info["step_info"].makespan_by_ue.values())
    else:
        ue_finish = list(base_env.scheduler.download_EAT.values())
    mean_aft = float(np.mean(ue_finish))
    makespan = float(np.max(ue_finish))
    return mean_aft, makespan, steps


def _get_base_env(env):
    """Unwrap to get DTOEnv (has scheduler)"""
    while hasattr(env, "env"):
        env = env.env
    return env


def run_rl_episode_vec(venv: VecEnv, model: MaskablePPO, deterministic: bool = True):
    """
    评估网络（VecEnv，支持 VecNormalize 归一化）
    DummyVecEnv 在 done=True 时会自动 reset，download_EAT 被清零，
    因此从 step 返回的 infos["step_info"].makespan_by_ue 获取结果。
    """
    obs = venv.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    steps = 0
    dones = [False]
    last_info = None

    while not dones[0]:
        masks = venv.envs[0].action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
        if isinstance(action, np.ndarray):
            action = action.reshape(-1)
        obs, rewards, dones, infos = venv.step(action)
        last_info = infos[0] if infos else None
        if isinstance(obs, tuple):
            obs = obs[0]
        steps += 1

    if last_info and "step_info" in last_info:
        ue_finish = list(last_info["step_info"].makespan_by_ue.values())
    else:
        base_env = _get_base_env(venv.envs[0])
        ue_finish = list(base_env.scheduler.download_EAT.values())
    mean_aft = float(np.mean(ue_finish))
    makespan = float(np.max(ue_finish))
    return mean_aft, makespan, steps

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

        env.step_greedy(node_id)
        steps += 1

    ue_finish = list(env.scheduler.download_EAT.values())
    mean_aft = float(np.mean(ue_finish))
    makespan = float(np.max(ue_finish))
    return mean_aft, makespan, steps

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
        reward_oracle=getattr(dag_cfg, "reward_oracle", "local"),
        reward_scale=getattr(dag_cfg, "reward_scale", False),
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
        # 评估 DAG 规模需与训练一致(n_compute_nodes_per_ue=20)，否则 VecNormalize 观测形状不匹配
        mean_aft_list, makespan_list = [], []
        use_vecnorm = os.path.exists(norm_path)
        vecnorm_fallback = False  # 若 VecNormalize 评估失败则回退到无归一化

        for s in range(n_eval_seeds):
            case = make_dag_case(
                ue_number=dag_cfg.ue_numbers,
                n_compute_nodes_per_ue=20,  # 与训练一致，否则 VecNormalize 观测形状不匹配
                start_count_max=3,
                seed=1000 + s,
            )
            env = build_env_from_dag_case(
                case=case,
                ue_number=dag_cfg.ue_numbers,
                es_number=dag_cfg.es_numbers,
                f_ue=dag_cfg.f_ue,
                f_es=dag_cfg.f_es,
                es_processors=4,
                tr_ue_es=dag_cfg.tr_ue_es,
                tr_es_es=dag_cfg.tr_es_es,
                reward_oracle=getattr(dag_cfg, "reward_oracle", "local"),
                reward_scale=getattr(dag_cfg, "reward_scale", False),
            )
            env = ActionMasker(env, lambda e: e.action_masks())
            venv = DummyVecEnv([lambda e=env: e])
            try:
                if use_vecnorm and not vecnorm_fallback:
                    venv = VecNormalize.load(norm_path, venv)
                    venv.training = False
                mean_aft, makespan, steps = run_rl_episode_vec(venv, model, deterministic=True)
                if not np.isfinite(mean_aft) or not np.isfinite(makespan):
                    raise ValueError(f"NaN/Inf in eval: mean_aft={mean_aft}, makespan={makespan}")
            except Exception as ex:
                if use_vecnorm and not vecnorm_fallback:
                    vecnorm_fallback = True
                    print(f"  [WARN] VecNormalize 评估异常 ({ex})，回退到无归一化评估")
                    env_fb = build_env_from_dag_case(
                        case=case,
                        ue_number=dag_cfg.ue_numbers,
                        es_number=dag_cfg.es_numbers,
                        f_ue=dag_cfg.f_ue,
                        f_es=dag_cfg.f_es,
                        es_processors=4,
                        tr_ue_es=dag_cfg.tr_ue_es,
                        tr_es_es=dag_cfg.tr_es_es,
                        reward_oracle=getattr(dag_cfg, "reward_oracle", "local"),
                        reward_scale=getattr(dag_cfg, "reward_scale", False),
                    )
                    env_fb = ActionMasker(env_fb, lambda e: e.action_masks())
                    venv = DummyVecEnv([lambda e=env_fb: e])
                    mean_aft, makespan, steps = run_rl_episode_vec(venv, model, deterministic=True)
                else:
                    raise
            mean_aft_list.append(mean_aft)
            makespan_list.append(makespan)

        results[model_type] = {
            "mean_AFT_avg": float(np.mean(mean_aft_list)),
            "makespan_avg": float(np.mean(makespan_list)),
            "mean_AFT_std": float(np.std(mean_aft_list)),
            "makespan_std": float(np.std(makespan_list)),
        }
        print(f"  -> meanAFT(avg)={results[model_type]['mean_AFT_avg']:.6f}, "
              f"makespan(avg)={results[model_type]['makespan_avg']:.6f}")

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

        mean_aft_list, makespan_list = [], []
        use_vecnorm = os.path.exists(norm_path)
        vecnorm_fallback = False
        for s in range(n_eval_seeds):
            case = make_dag_case(
                ue_number=cfg.ue_numbers,
                n_compute_nodes_per_ue=20,  # 与训练一致
                start_count_max=3,
                seed=1000 + s,
            )
            env = build_env_from_dag_case(
                case=case,
                ue_number=cfg.ue_numbers,
                es_number=cfg.es_numbers,
                f_ue=cfg.f_ue,
                f_es=cfg.f_es,
                es_processors=4,
                tr_ue_es=cfg.tr_ue_es,
                tr_es_es=cfg.tr_es_es,
                reward_oracle=oracle,
                reward_scale=scale,
            )
            env = ActionMasker(env, lambda e: e.action_masks())
            venv = DummyVecEnv([lambda e=env: e])
            try:
                if use_vecnorm and not vecnorm_fallback:
                    venv = VecNormalize.load(norm_path, venv)
                    venv.training = False
                mean_aft, makespan, _ = run_rl_episode_vec(venv, model, deterministic=True)
                if not np.isfinite(mean_aft) or not np.isfinite(makespan):
                    raise ValueError(f"NaN/Inf in eval: mean_aft={mean_aft}, makespan={makespan}")
            except Exception as ex:
                if use_vecnorm and not vecnorm_fallback:
                    vecnorm_fallback = True
                    print(f"  [WARN] VecNormalize 评估异常 ({ex})，回退到无归一化评估")
                    env_fb = build_env_from_dag_case(
                        case=case,
                        ue_number=cfg.ue_numbers,
                        es_number=cfg.es_numbers,
                        f_ue=cfg.f_ue,
                        f_es=cfg.f_es,
                        es_processors=4,
                        tr_ue_es=cfg.tr_ue_es,
                        tr_es_es=cfg.tr_es_es,
                        reward_oracle=oracle,
                        reward_scale=scale,
                    )
                    env_fb = ActionMasker(env_fb, lambda e: e.action_masks())
                    venv = DummyVecEnv([lambda e=env_fb: e])
                    mean_aft, makespan, _ = run_rl_episode_vec(venv, model, deterministic=True)
                else:
                    raise
            mean_aft_list.append(mean_aft)
            makespan_list.append(makespan)

        results[name] = {
            "mean_AFT_avg": float(np.mean(mean_aft_list)),
            "makespan_avg": float(np.mean(makespan_list)),
            "mean_AFT_std": float(np.std(mean_aft_list)),
            "makespan_std": float(np.std(makespan_list)),
        }
        print(f"  -> meanAFT(avg)={results[name]['mean_AFT_avg']:.6f}, "
              f"makespan(avg)={results[name]['makespan_avg']:.6f}")

    print(f"\n{'='*70}")
    print("奖励函数对比结果 (baseline vs improved)")
    print(f"{'='*70}")
    print(f"{'奖励配置':<12} | {'meanAFT(avg)':>12} ± {'std':>8} | {'makespan(avg)':>12} ± {'std':>8}")
    print("-" * 70)
    for name in ["baseline", "improved"]:
        r = results[name]
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
            reward_oracle=getattr(dag_cfg, "reward_oracle", "local"),
            reward_scale=getattr(dag_cfg, "reward_scale", False),
        )

        trainer = DTODRLTrainer(config=train_cfg, env_controller=rl_train_controller)
        model_path = trainer.train(model_type=model_type)
        print(f"训练完成，模型已保存到: {model_path}")

        # 拿到训练完的模型 (可用于后续评估)
        model = trainer.model

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


