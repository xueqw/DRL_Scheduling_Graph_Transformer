# DTODRL 论文与当前实现方法对比

**论文**: *Dependent Task Offloading in Edge Computing Using GNN and Deep Reinforcement Learning* (Cao & Deng, arXiv:2303.17100)

---

## 1. 论文 DTODRL 方法概要

### 1.1 核心设计
- **GAT + PPO** 解决 DTO（Dependent Task Offloading）问题
- **多离散动作空间**: 每步同时选择 (节点, 位置) 对
- **GAT 预训练**: 在云端无监督预训练，与 DRL 解耦

### 1.2 动作空间（论文核心）
```
A = { at | at = (anode, alocation) }
```
- **Node Head**: 输出 N 维 logits，在合法节点上做 Softmax（node mask）
- **Location Head**: 输出 (M+1) 维 logits，表示 [本地 UE, ES1, ..., ESM]
- **两头共享同一 state**，**相互独立**：
  ```
  P(anode, alocation | s) = P(anode | s) × P(alocation | s)
  ```
- 维度 0 表示「该节点所属 UE 的本地」，1..M 表示 ES

### 1.3 状态表示
- **stask** = GAT(otask)，节点特征经过 GAT 编码
- **slocations** = MLP(olocations)，位置特征经过 MLP
- **State** = concat(stask_flatten, slocations)

### 1.4 观测特征
- **节点**: Ci, Di, in-degree, out-degree, loci, avai（6 维）
- **位置**: (EFT, f) 每位置 2 维

### 1.5 GAT 预训练
- GAT 自编码器：Encoder + Decoder
- 损失：Feature Loss (MSE) + Structure Loss (邻接相似度)
- 仅保留 Encoder，在云端训练，下发到边缘

### 1.6 奖励
```
r_t = (1/K) × (Σ EFT_{A1:t-1} - Σ EFT_{A1:t})
```
即每步平均完成时间的**减少量**。

---

## 2. 当前实现方法对比

### 2.1 三种方法对比表

| 维度 | 论文 DTODRL | 当前 Joint（单阶段） | 当前 Two-Stage |
|------|-------------|----------------------|----------------|
| **动作分解** | (anode, alocation) 两个独立头 | 单一 (node, loc) 联合打分 | 先选 node，再选 loc |
| **概率形式** | P(anode)×P(alocation) 独立 | P(node,loc) 联合 | π₁(node)×π₂(loc\|node) 条件 |
| **是否利用「选 node 后再选 loc」的依赖** | ❌ 否 | ✅ 是 | ✅ 是 |
| **Location 维度** | 固定 M+1，共享 | 每 node 有 N×K 候选 | 每 node 有 K 候选，依赖 node |
| **GAT 训练** | 云端无监督预训练 | 端到端与 PPO 同训 | 端到端与 PPO 同训 |
| **打分输入** | state → Node Head / Location Head | (node_ctx, loc_emb) 联合 → score_mlp | node_ctx→f₁; (node_i, loc_j)→f₂ |

### 2.2 动作空间细节对比

```
论文 DTODRL:
  Node Head:    logits ∈ R^N    → Softmax(masked) → anode
  Location Head: logits ∈ R^{M+1} → Softmax        → alocation
  联合: P(a) = P(anode) × P(alocation)   [假设独立]

当前 Joint:
  score_mlp(node_ctx ⊕ loc_emb) → scores[n,k] → logits = reshape to (N×K)
  P(a) = Softmax(logits)[a]   [显式建模 (node,loc) 联合]

当前 Two-Stage:
  f₁(s,i) → π₁(i|s)
  f₂(i,r,j) → π₂(j|i,r)
  P(a) = π₁(node) × π₂(loc|node)   [loc 依赖 node]
```

### 2.3 观测 / 状态差异

| 项目 | 论文 | DTODRL baseline | Joint / Two-Stage |
|------|------|-----------------|-------------------|
| 节点特征 | Ci, Di, in-deg, out-deg, loci, avai | **已统一** 6 维 | **已统一** 6 维 |
| 位置特征 | (EFT, f) | **(EFT, f)** 两维 | (cpu_speed, min_EAT, num_processor) |
| GNN | GAT（标准） | **标准 GATConv** | TransformerConv |
| GAT 激活 | LeakyReLU | **LeakyReLU** | ReLU |
| MLP 激活 | Tanh | **Tanh** | ReLU |

### 2.4 奖励函数

| 方法 | 奖励形式 | 说明 |
|------|----------|------|
| 论文 | r = (1/K)(Σ EFT_old - Σ EFT_new) | 平均 AFT 的减少 |
| 当前 | r = prev_mean_eft - mean_eft | 与论文一致（K=1 时等价） |

---

## 3. 方法优缺点简要分析

### 3.1 论文 DTODRL
- **优点**: 结构简单；Location Head 维度固定，易扩展；GAT 预训练可减轻 DRL 训练负担
- **缺点**: Node 与 Location **假设独立**，忽略「选好 node 后选 loc」的耦合，可能次优

### 3.2 当前 Joint
- **优点**: 联合建模 (node, loc)，充分利用二者依赖
- **缺点**: 参数量与 N×K 相关，大图时计算更贵

### 3.3 当前 Two-Stage
- **优点**: 分步决策，计算更高效；π₂(j|i) 显式建模 node–loc 依赖
- **缺点**: 两阶段序贯，存在信息不完整（第一阶段看不到第二阶段结果）

---

## 4. 实现与实验

### 4.1 已实现模块
- **DTODRL baseline**（与论文一致）：`dtodrl_policy.py`、`dtodrl_maskable_policy.py`、`dtodrl_backbone.py`
- **对比文档**：`docs/DTODRL_Paper_Comparison.md`

### 4.2 DTODRL 与论文对齐项
- Node Head: 统一 state = concat(flatten(node_embs), slocations) → 一次输出 N 维
- Location Head: slocations 逐位置 embedding → K 维
- GAT: 标准 GATConv, 3 层, hidden 128, heads 3, LeakyReLU
- GAT 预训练: `python gat_pretrain.py` 或 `run_pretrain_and_save()`, 通过 `config.dtodrl_pretrained_gat` 加载; `freeze_pretrained_gat=True` 时冻结 encoder
- 位置特征: (EFT, f) 两维
- MLP: hidden 256, Tanh
- PPO: γ=0.99, ent_coef=0.01（论文 Table 2）

### 4.3 运行方式
```bash
# 单方法训练
python final_training.py joint      # 联合打分
python final_training.py two_stage  # 两阶段
python final_training.py dtodrl     # 论文 DTODRL baseline

# 多方法对比实验 (DTODRL vs Joint vs Two-Stage)
python final_training.py comparison
```

### 4.4 建议对比维度
- mean AFT、makespan
- 训练稳定性与收敛速度
- 不同 DAG 规模（节点数、UE 数、ES 数）下的表现

### 4.5 可选扩展
- 增加 GAT 无监督预训练，对比「预训练 GAT + DRL」与「端到端联合训练」
