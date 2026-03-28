"""
最小复现实验：在同一组 DAG seeds 上比较 ML(policy) 与 greedy baseline。

用法示例：
python repro_ml_vs_greedy.py --method joint --train-steps 6000 --eval-seeds 8 --nodes-per-ue 8 --seed-base 2100
"""
import argparse
import copy
import json
import os
import statistics
import time
from typing import Dict, List

import numpy as np

from final_training import (
    TrainConfig,
    DAGConfig,
    DTODRLTrainer,
    build_env_from_dag_case,
    make_dto_env_controller,
    run_baseline_episode,
    run_rl_episode_vec,
    _build_eval_vec_env,
)
from dag_generator import make_dag_case


def evaluate_same_seed(
    *,
    model,
    dag_cfg: DAGConfig,
    norm_path: str,
    case,
) -> Dict:
    # RL：与训练路径保持一致（含可选 VecNormalize）
    venv = _build_eval_vec_env(case, dag_cfg, norm_path)
    rl_mean_aft, rl_makespan, rl_steps, rl_stats = run_rl_episode_vec(
        venv, model, deterministic=True
    )
    if hasattr(venv, "close"):
        venv.close()

    # Baseline：同一个 DAG case，分别评估 topo/sjf/ljf
    base_results = {}
    for rule in ("topo", "sjf", "ljf"):
        env_base = build_env_from_dag_case(
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
        b_mean_aft, b_makespan, b_steps = run_baseline_episode(env_base, rule)
        base_results[rule] = {
            "mean_aft": float(b_mean_aft),
            "makespan": float(b_makespan),
            "steps": int(b_steps),
        }

    best_rule = min(base_results.keys(), key=lambda r: base_results[r]["mean_aft"])
    return {
        "rl": {
            "mean_aft": float(rl_mean_aft),
            "makespan": float(rl_makespan),
            "steps": int(rl_steps),
            "stats": rl_stats,
        },
        "baseline_by_rule": base_results,
        "baseline_best_rule": best_rule,
        "baseline_best": copy.deepcopy(base_results[best_rule]),
    }


def summarize(records: List[Dict]) -> Dict:
    rl_mean_aft = [x["rl"]["mean_aft"] for x in records]
    rl_makespan = [x["rl"]["makespan"] for x in records]
    base_mean_aft = [x["baseline_best"]["mean_aft"] for x in records]
    base_makespan = [x["baseline_best"]["makespan"] for x in records]
    gap_mean_aft = [r - b for r, b in zip(rl_mean_aft, base_mean_aft)]
    gap_makespan = [r - b for r, b in zip(rl_makespan, base_makespan)]

    better_cnt = int(sum(g <= 0 for g in gap_mean_aft))
    return {
        "rl_mean_aft_avg": float(np.mean(rl_mean_aft)),
        "rl_makespan_avg": float(np.mean(rl_makespan)),
        "baseline_mean_aft_avg": float(np.mean(base_mean_aft)),
        "baseline_makespan_avg": float(np.mean(base_makespan)),
        "gap_mean_aft_avg_rl_minus_baseline": float(np.mean(gap_mean_aft)),
        "gap_makespan_avg_rl_minus_baseline": float(np.mean(gap_makespan)),
        "rl_not_worse_ratio_by_seed": float(better_cnt / len(records)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["joint", "dtodrl"], default="joint")
    parser.add_argument("--train-steps", type=int, default=6000)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=2100)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--nodes-per-ue", type=int, default=8)
    parser.add_argument("--start-count-max", type=int, default=2)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--use-vecnormalize", action="store_true")
    args = parser.parse_args()

    dag_cfg = DAGConfig()
    train_cfg = TrainConfig(
        seed=args.train_seed,
        n_envs=args.n_envs,
        use_subproc=False,
        total_timesteps=args.train_steps,
        n_steps=256,
        batch_size=128,
        use_vecnormalize=args.use_vecnormalize,
        runtime=f"DTO_DEBUG_{args.method}",
    )

    controller = make_dto_env_controller(
        ue_number=dag_cfg.ue_numbers,
        es_number=dag_cfg.es_numbers,
        n_compute_nodes_per_ue=args.nodes_per_ue,
        start_count_max=args.start_count_max,
        f_ue=dag_cfg.f_ue,
        f_es=dag_cfg.f_es,
        es_processors=4,
        tr_ue_es=dag_cfg.tr_ue_es,
        tr_es_es=dag_cfg.tr_es_es,
        reward_oracle=getattr(dag_cfg, "reward_oracle", "greedy"),
        reward_scale=getattr(dag_cfg, "reward_scale", True),
    )

    trainer = DTODRLTrainer(config=train_cfg, env_controller=controller)
    trainer.build_vec_env()
    trainer.build_model(model_type=args.method)
    trainer.train(model_type=args.method)
    model = trainer.model
    norm_path = os.path.join(trainer.run_dic, "vecnormalize.pkl")

    records: List[Dict] = []
    for idx in range(args.eval_seeds):
        seed = args.seed_base + idx
        case = make_dag_case(
            ue_number=dag_cfg.ue_numbers,
            n_compute_nodes_per_ue=args.nodes_per_ue,
            start_count_max=args.start_count_max,
            seed=seed,
        )
        one = evaluate_same_seed(model=model, dag_cfg=dag_cfg, norm_path=norm_path, case=case)
        one["seed"] = seed
        records.append(one)

    summary = summarize(records)
    output = {
        "method": args.method,
        "train_steps": args.train_steps,
        "eval_seed_count": args.eval_seeds,
        "nodes_per_ue": args.nodes_per_ue,
        "start_count_max": args.start_count_max,
        "summary": summary,
        "records": records,
    }

    os.makedirs("runs", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join("runs", f"debug_ml_vs_greedy_{args.method}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[DEBUG] result_json={out_path}")
    print(
        f"[DEBUG] RL meanAFT={summary['rl_mean_aft_avg']:.6f}, "
        f"BestGreedy meanAFT={summary['baseline_mean_aft_avg']:.6f}, "
        f"gap={summary['gap_mean_aft_avg_rl_minus_baseline']:.6f}, "
        f"not_worse_ratio={summary['rl_not_worse_ratio_by_seed']:.3f}"
    )

    for r in records:
        print(
            f"[SEED {r['seed']}] rl.meanAFT={r['rl']['mean_aft']:.6f}, "
            f"greedy_best={r['baseline_best']['mean_aft']:.6f} ({r['baseline_best_rule']})"
        )


if __name__ == "__main__":
    main()
