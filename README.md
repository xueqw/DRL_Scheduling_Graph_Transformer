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

# 对比实验
python run_comparison.py
# 或
python final_training.py comparison
```

## 依赖

- Python 3.8+
- PyTorch, torch-geometric
- stable-baselines3, sb3-contrib
- gymnasium
