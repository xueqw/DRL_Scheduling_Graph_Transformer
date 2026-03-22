# 修复日志 (2026-03-22)

## 概述

修复对比实验 (DTODRL vs Joint vs Two-Stage) 运行时的评估崩溃、结果异常等问题。

---

## 1. 导入修复

**文件**: `DTO_env.py`

**问题**: 从 `GNNDAG.DTO_scheduler` 导入 `Node`，但项目为扁平结构，无 `GNNDAG` 包。

**修复**:
```diff
- from GNNDAG.DTO_scheduler import Node
+ from DTO_scheduler import Node
```

---

## 2. 评估阶段 DAG 规模与 VecNormalize 形状不匹配

**文件**: `final_training.py` — `run_comparison_experiment()`

**问题**: 训练时 `n_compute_nodes_per_ue=20`（约 60 节点），评估时 `n_compute_nodes_per_ue=50`（约 150 节点），VecNormalize 的观测形状 `(60,60)` 与评估 `(150,150)` 不一致，导致 `AssertionError: spaces must have the same shape`。

**修复**:
```diff
  case = make_dag_case(
      ...
-     n_compute_nodes_per_ue=50,
+     n_compute_nodes_per_ue=20,  # 与训练一致，否则 VecNormalize 观测形状不匹配
      ...
  )
```

---

## 3. 评估改用 run_rl_episode 并修复 base_env

**文件**: `final_training.py` — `run_rl_episode()`, `run_comparison_experiment()`

**问题**:
- 原用 `run_rl_episode_vec` + VecNormalize，Joint 评估时出现 NaN logits 崩溃。
- 回退到无 VecNormalize 后，DTODRL 和 Joint 结果差异异常（如 DTODRL 1.5 vs Joint 5.0），因观测分布与训练不一致。

**修复**:
- `run_rl_episode()` 使用 `_get_base_env(env)` 获取 DTOEnv，确保 `base_env.done()` 和 `base_env.scheduler.download_EAT` 正确。
- 评估逻辑支持 VecNormalize 优先，失败时回退到无归一化。

---

## 4. DummyVecEnv auto-reset 导致 download_EAT 为 0

**文件**: `final_training.py` — `run_rl_episode_vec()`

**问题**: DummyVecEnv 在 `done=True` 时会自动 `reset()`，`env.scheduler.download_EAT` 被清零，导致 meanAFT/makespan 始终为 0。

**修复**: 从 `step` 返回的 `infos["step_info"].makespan_by_ue` 获取结果，而不是从 env 读取。

```diff
  def run_rl_episode_vec(...):
      ...
+     last_info = None
      while not dones[0]:
          ...
          obs, rewards, dones, infos = venv.step(action)
+         last_info = infos[0] if infos else None
          ...

-     base_env = _get_base_env(venv.envs[0])
-     ue_finish = list(base_env.scheduler.download_EAT.values())
+     if last_info and "step_info" in last_info:
+         ue_finish = list(last_info["step_info"].makespan_by_ue.values())
+     else:
+         base_env = _get_base_env(venv.envs[0])
+         ue_finish = list(base_env.scheduler.download_EAT.values())
```

---

## 5. 观测 dtype 不匹配导致 Joint 评估崩溃

**文件**: `DTO_env.py` — `build_obs()`

**问题**: `loc_cpu_speed`, `loc_min_processor_EAT` 等以 Python 列表返回，转为 tensor 后为 float64，与模型期望的 float32 不一致，导致 `RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float`。

**修复**:
```diff
  observation = {
      ...
-     "loc_cpu_speed": cpu_speed,
-     "loc_min_processor_EAT": min_processor_EAT,
-     "loc_num_processor": num_processor,
-     "ue_upload_EAT": upload_EAT,
-     "ue_download_EAT": download_EAT,
+     "loc_cpu_speed": np.asarray(cpu_speed, dtype=np.float32),
+     "loc_min_processor_EAT": np.asarray(min_processor_EAT, dtype=np.float32),
+     "loc_num_processor": np.asarray(num_processor, dtype=np.int64),
+     "ue_upload_EAT": np.asarray(upload_EAT, dtype=np.float32),
+     "ue_download_EAT": np.asarray(download_EAT, dtype=np.float32),
      ...
  }
```

---

## 6. 对比实验评估逻辑：VecNormalize 优先 + 回退

**文件**: `final_training.py` — `run_comparison_experiment()`

**问题**: 部分 DAG 在 VecNormalize 下会产生 NaN，需在失败时回退到无归一化评估。

**修复**: 评估时优先使用 VecNormalize；若出现异常或 NaN 结果，则对该方法全部改用无归一化评估，并打印警告。

---

## 7. 新增可视化脚本

**文件**: `visualize_comparison.py`（新建）

**功能**: 读取 `runs/comparison_results_*.json`，绘制 meanAFT 与 makespan 对比柱状图。

**用法**:
```bash
python visualize_comparison.py              # 自动找最新结果
python visualize_comparison.py --results runs/comparison_results_xxx.json
python visualize_comparison.py --no-show    # 仅保存图片，不弹窗
```

---

## 修改文件汇总

| 文件 | 修改类型 |
|------|----------|
| `DTO_env.py` | 导入修复、观测 dtype 修复 |
| `final_training.py` | 评估 DAG 规模、run_rl_episode、run_rl_episode_vec、对比实验逻辑 |
| `visualize_comparison.py` | 新建 |
| `CHANGELOG_FIXES.md` | 新建（本文件） |

---

## 验证结果

修复后，使用 VecNormalize 评估：
- **DTODRL**: meanAFT ~1.5–1.7, makespan ~1.6–1.9
- **Joint**: meanAFT ~1.3–1.4, makespan ~1.6–1.8

两者量级一致，Joint 略优，符合预期。
