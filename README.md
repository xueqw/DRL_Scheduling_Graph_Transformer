# DRL Scheduling Graph Transformer

## 1. 环境

```bash
conda create -n drl_sched python=3.10 -y
conda activate drl_sched
pip install torch torchvision torchaudio
pip install torch-geometric
pip install stable-baselines3 sb3-contrib gymnasium tensorboard numpy gensim matplotlib networkx
```

## 2. 拉代码

```bash
git clone https://github.com/xueqw/DRL_Scheduling_Graph_Transformer.git
cd DRL_Scheduling_Graph_Transformer
git pull origin main
```

## 3. 先测一下

```bash
python smoke_test.py
```

## 4. 单模型训练

```bash
python final_training.py joint
python final_training.py two_stage
python final_training.py dtodrl
python final_training.py dtodrl_tf
```

## 5. 对比实验

```bash
python final_training.py comparison
python final_training.py comparison_transformer
python final_training.py reward_comparison
python final_training.py reward_comparison joint
python final_training.py reward_comparison two_stage
python final_training.py reward_comparison dtodrl
```

也可以跑封装脚本：

```bash
python run_comparison.py
python run_comparison_transformer.py
python run_reward_comparison.py joint
```

## 6. 看日志

```bash
tensorboard --logdir runs
```

## 7. 重点看这些文件

单模型目录下：

- `runs/DTO_DRL_<method>_<time>/final_model.zip`
- `runs/DTO_DRL_<method>_<time>/policy_diagnosis_*.json`

对比实验目录下：

- `runs/comparison_results_*.json`
- `runs/comparison_transformer_results_*.json`
- `runs/reward_comparison_*.json`

## 8. 复查塌缩时只跑这个

```bash
python final_training.py joint
python final_training.py two_stage
python final_training.py dtodrl
```

跑完重点看：

- `deterministic.es_ratio_avg`
- `stochastic.es_ratio_avg`
- `policy_es_mass_avg`
- `policy_local_mass_avg`
- `policy_loc_tie_share_avg`
