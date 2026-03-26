# DRL Scheduling Graph Transformer

This repository implements dependent task offloading (DTO) in edge computing with several reinforcement-learning policies:

- `joint`: single-stage joint scoring over `(node, location)`
- `two_stage`: select node first, then select location
- `dtodrl`: paper-style DTODRL baseline with separate node/location heads
- `dtodrl_tf`: DTODRL heads on the shared Transformer-based backbone

The main training and evaluation entry point is [`final_training.py`](./final_training.py).

## 1. Environment

Recommended:

- Python 3.10 or 3.11
- PyTorch with CUDA if you want GPU training
- Linux or Windows

Core Python dependencies used by this repo:

- `torch`
- `torch-geometric`
- `stable-baselines3`
- `sb3-contrib`
- `gymnasium`
- `tensorboard`
- `numpy`
- `gensim`

Optional utilities:

- `matplotlib`
- `networkx`

## 2. Installation

Create and activate a clean environment first.

### Conda example

```bash
conda create -n drl_sched python=3.10 -y
conda activate drl_sched
```

### Install PyTorch

Install a PyTorch build that matches your machine first.

Examples:

```bash
# CUDA example
pip install torch torchvision torchaudio

# CPU-only example
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Install PyTorch Geometric

Install `torch-geometric` with wheels that match your installed PyTorch version and CUDA/CPU setup.

Typical command:

```bash
pip install torch-geometric
```

If your machine needs the version-matched wheels from PyG, install them according to the official PyTorch Geometric instructions for your Torch version.

### Install the remaining packages

```bash
pip install stable-baselines3 sb3-contrib gymnasium tensorboard numpy gensim matplotlib networkx
```

## 3. Quick Start

Clone the repository and enter the project directory:

```bash
git clone https://github.com/xueqw/DRL_Scheduling_Graph_Transformer.git
cd DRL_Scheduling_Graph_Transformer
```

Run a quick local smoke test first:

```bash
python smoke_test.py
```

Then train one policy:

```bash
python final_training.py joint
```

Other supported single-method modes:

```bash
python final_training.py two_stage
python final_training.py dtodrl
python final_training.py dtodrl_tf
```

## 4. Comparison Experiments

Run the three-method comparison:

```bash
python final_training.py comparison
```

Equivalent wrapper:

```bash
python run_comparison.py
```

Run the shared-transformer comparison:

```bash
python final_training.py comparison_transformer
```

Equivalent wrapper:

```bash
python run_comparison_transformer.py
```

Run the reward comparison:

```bash
python final_training.py reward_comparison
```

Or specify the policy explicitly:

```bash
python final_training.py reward_comparison joint
python final_training.py reward_comparison two_stage
python final_training.py reward_comparison dtodrl
```

Equivalent wrapper:

```bash
python run_reward_comparison.py joint
```

## 5. Optional GAT Pretraining

If you want to pretrain the DTODRL GAT encoder first:

```bash
python gat_pretrain.py
```

This saves a pretrained encoder state dict. To actually use that checkpoint in `dtodrl`, set the following fields in [`final_training.py`](./final_training.py):

- `TrainConfig.dtodrl_pretrained_gat`
- `TrainConfig.dtodrl_freeze_pretrained_gat`

## 6. Outputs

By default, outputs are written under `./runs`.

Single-policy training creates a timestamped run directory such as:

```text
runs/DTO_DRL_20260326-123456/
```

Important files inside a run directory:

- `final_model.zip`
- `vecnormalize.pkl` when `use_vecnormalize=True`
- TensorBoard event files
- `policy_diagnosis_*.json` after single-method training runs

Comparison scripts additionally save JSON summaries directly under `runs/`, for example:

- `comparison_results_*.json`
- `comparison_transformer_results_*.json`
- `reward_comparison_*.json`

Current comparison outputs also record:

- `es_ratio_avg`
- `local_ratio_avg`
- `es_ratio_std`
- `local_ratio_std`

These metrics help detect whether a policy collapses to all-local or all-ES assignment.

Recent diagnostics also include deterministic vs stochastic ratios, policy-side local/ES probability mass, and `policy_loc_tie_share_avg`.

- If `policy_loc_tie_share_avg` is close to `1.0`, the location head is producing ties for many ready nodes.
- In that case, deterministic evaluation will often fall back to location index `0` first, which is the local option in this environment.
- This is different from a true local preference. Check `policy_local_pref_avg` / `policy_es_pref_avg` together with `policy_loc_tie_share_avg`.

## 7. TensorBoard

Launch TensorBoard from the repo root:

```bash
tensorboard --logdir runs
```

Then open the local TensorBoard URL shown in the terminal.

## 8. Default Training Settings

The default command-line entry in [`final_training.py`](./final_training.py) currently uses:

- `n_envs = 8`
- `total_timesteps = 50000`
- `seed = 0`
- `ue_numbers = 3`
- `es_numbers = 2`
- `n_compute_nodes_per_ue = 20`
- `reward_oracle = "greedy"`
- `reward_scale = True`

If you want to change the training budget or DAG size, edit:

- `TrainConfig`
- `DAGConfig`
- the `if __name__ == "__main__":` block in [`final_training.py`](./final_training.py)

The wrapper scripts use the same defaults.

## 9. Notes

- Device selection is automatic: the code uses CUDA when available, otherwise CPU.
- The environment import in [`DTO_env.py`](./DTO_env.py) is expected to resolve `Node` from the local [`DTO_scheduler.py`](./DTO_scheduler.py).
- Some auxiliary visualization scripts require `matplotlib` and `networkx`.
- This repo does not currently ship a pinned `requirements.txt`, so if you are reproducing results on a new machine, install the packages above in a clean virtual environment.

## 10. Minimal Run Checklist

If you just want the shortest path to a training run:

```bash
conda create -n drl_sched python=3.10 -y
conda activate drl_sched
pip install torch torchvision torchaudio
pip install torch-geometric
pip install stable-baselines3 sb3-contrib gymnasium tensorboard numpy gensim matplotlib networkx
git clone https://github.com/xueqw/DRL_Scheduling_Graph_Transformer.git
cd DRL_Scheduling_Graph_Transformer
python smoke_test.py
python final_training.py joint
tensorboard --logdir runs
```

## 11. Reproducing The Latest Collapse Check

If you are validating the latest `main`, do not start with the full comparison run.

Recommended order:

```bash
git pull origin main
python final_training.py joint
python final_training.py two_stage
python final_training.py dtodrl
```

After each single-method run, keep these files:

- `runs/DTO_DRL_<method>_<timestamp>/final_model.zip`
- `runs/DTO_DRL_<method>_<timestamp>/policy_diagnosis_*.json`
- TensorBoard event files under the same run directory

The most useful fields in `policy_diagnosis_*.json` are:

- `deterministic.es_ratio_avg` / `deterministic.local_ratio_avg`
- `stochastic.es_ratio_avg` / `stochastic.local_ratio_avg`
- `deterministic.policy_es_mass_avg` / `deterministic.policy_local_mass_avg`
- `deterministic.policy_es_pref_avg` / `deterministic.policy_local_pref_avg`
- `deterministic.policy_loc_tie_share_avg`

How to read them quickly:

- `deterministic` all-local or all-ES but `stochastic` is not:
  greedy action selection is amplifying a mild bias.
- `policy_loc_tie_share_avg` close to `1.0`:
  the location logits are tying, so deterministic decoding is not trustworthy by itself.
- `policy_es_mass_avg` close to `1.0` and `policy_loc_tie_share_avg` low:
  the policy distribution itself is really collapsing to ES.
