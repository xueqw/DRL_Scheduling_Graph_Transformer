# DRL_Scheduling

Dependent Task Offloading (DTO) with Deep Reinforcement Learning. 实现 DTODRL（论文基线）、Joint、Two-Stage 三种方法对比。

## 方法

- **DTODRL**: 论文 *Dependent Task Offloading in Edge Computing Using GNN and Deep Reinforcement Learning* 原方法，Node Head + Location Head 独立双头
- **Joint**: 单阶段 (node, loc) 联合打分
- **Two-Stage**: 两阶段，先选 Node 再选 Location

## 运行

```bash
# 单方法训练
python final_training.py joint      # 或 two_stage / dtodrl

# 方法对比实验 (DTODRL vs Joint vs Two-Stage，各自原图编码)
python final_training.py comparison
# 或 python run_comparison.py

# 横向对比 (统一 TransformerConv 图编码)
python final_training.py comparison_transformer
# 或 python run_comparison_transformer.py

# 奖励函数对比实验 (baseline vs improved)
python final_training.py reward_comparison [joint]
# 或 python run_reward_comparison.py [joint]
```

## 奖励函数

- **baseline** (默认): 论文 oracle，剩余节点全部本地执行预估 mean_eft，无缩放
- **improved**: greedy oracle（每步选 EFT 最小 loc）+ 相对奖励缩放
- 可通过 `DAGConfig(reward_oracle="greedy", reward_scale=True)` 使用改进版

## 依赖

- Python 3.8+
- PyTorch, torch-geometric
- stable-baselines3, sb3-contrib
- gymnasium
