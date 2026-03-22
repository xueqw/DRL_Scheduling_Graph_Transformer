"""
横向对比实验: 三种方法统一使用 TransformerConv 图编码

用法: python run_comparison_transformer.py

对比 dtodrl_tf / joint / two_stage，仅决策头不同，图编码均为 TransformerConv。
"""
from final_training import (
    TrainConfig,
    DAGConfig,
    run_comparison_transformer_experiment,
)

if __name__ == "__main__":
    train_cfg = TrainConfig(
        seed=0,
        n_envs=8,
        use_subproc=False,
        total_timesteps=50_000,
    )

    dag_cfg = DAGConfig()

    print("运行横向对比实验 (统一 TransformerConv 图编码，全部端到端训练不冻结)...")
    print(f"  - 方法: dtodrl_tf, joint, two_stage")
    print(f"  - 每个方法训练 {train_cfg.total_timesteps} steps")
    print(f"  - 评估 30 个随机 DAG 种子")
    print()

    run_comparison_transformer_experiment(
        train_cfg,
        dag_cfg,
        n_eval_episodes=10,
        n_eval_seeds=30,
    )
