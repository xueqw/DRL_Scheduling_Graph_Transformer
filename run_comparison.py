"""
DTODRL vs Joint vs Two-Stage 对比实验

用法: python run_comparison.py

依次训练三种方法，在相同 DAG 测试集上评估，输出 meanAFT 和 makespan 对比表。
"""
from final_training import (
    TrainConfig,
    DAGConfig,
    run_comparison_experiment,
)

if __name__ == "__main__":
    train_cfg = TrainConfig(
        seed=0,
        n_envs=8,
        use_subproc=False,
        total_timesteps=50_000,
    )

    dag_cfg = DAGConfig()

    print("运行 DTODRL vs Joint vs Two-Stage 对比实验...")
    print(f"  - 每个方法训练 {train_cfg.total_timesteps} steps")
    print(f"  - 评估 {30} 个随机 DAG 种子")
    print()

    run_comparison_experiment(
        train_cfg,
        dag_cfg,
        n_eval_episodes=10,
        n_eval_seeds=30,
        methods=["dtodrl", "joint", "two_stage"],
    )
