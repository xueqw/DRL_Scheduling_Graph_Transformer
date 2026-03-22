#!/usr/bin/env python3
"""
读取 runs/comparison_results_*.json，绘制 meanAFT 与 makespan 对比柱状图。
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def find_latest_results(log_dir: str = "./runs") -> str:
    """自动查找最新的 comparison 结果文件"""
    patterns = [
        os.path.join(log_dir, "comparison_results_*.json"),
        os.path.join(log_dir, "comparison_transformer_results_*.json"),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        raise FileNotFoundError(f"未找到 {log_dir}/comparison*_results_*.json")
    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="对比实验可视化")
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="结果 JSON 路径，默认自动找最新",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="仅保存图片，不弹窗显示",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片路径，默认与 JSON 同目录",
    )
    args = parser.parse_args()

    if args.results:
        path = args.results
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    else:
        path = find_latest_results()
        print(f"使用结果文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    methods = list(results.keys())
    mean_aft = [results[m]["mean_AFT_avg"] for m in methods]
    mean_aft_std = [results[m]["mean_AFT_std"] for m in methods]
    makespan = [results[m]["makespan_avg"] for m in methods]
    makespan_std = [results[m]["makespan_std"] for m in methods]

    x = np.arange(len(methods))
    width = 0.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.bar(x, mean_aft, width, yerr=mean_aft_std, label="meanAFT", capsize=3)
    ax1.set_ylabel("meanAFT")
    ax1.set_title("meanAFT 对比")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()

    ax2.bar(x, makespan, width, yerr=makespan_std, label="makespan", capsize=3)
    ax2.set_ylabel("makespan")
    ax2.set_title("makespan 对比")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()

    plt.tight_layout()

    out_path = args.output or path.replace(".json", ".png")
    plt.savefig(out_path, dpi=150)
    print(f"图片已保存: {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
