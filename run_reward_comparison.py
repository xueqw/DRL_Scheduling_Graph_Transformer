"""
奖励函数对比实验: baseline (local oracle) vs improved (greedy oracle + scale)

用法: python run_reward_comparison.py [joint|two_stage|dtodrl]

默认用 joint 方法，可指定其他方法。
"""
import sys
from final_training import (
    TrainConfig,
    DAGConfig,
    run_reward_comparison_experiment,
)

if __name__ == "__main__":
    train_cfg = TrainConfig(
        seed=0,
        n_envs=8,
        use_subproc=False,
        total_timesteps=50_000,
    )

    dag_cfg = DAGConfig()
    model_type = sys.argv[1] if len(sys.argv) > 1 else "joint"

    print("运行奖励函数对比实验 (baseline vs improved)...")
    print(f"  - 方法: {model_type}")
    print(f"  - 每个奖励配置训练 {train_cfg.total_timesteps} steps")
    print(f"  - 评估 30 个随机 DAG 种子")
    print()

    run_reward_comparison_experiment(
        train_cfg,
        dag_cfg,
        model_type=model_type,
        n_eval_seeds=30,
    )
