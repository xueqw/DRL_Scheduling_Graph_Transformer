# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

This is a Python-based Deep Reinforcement Learning research project for Dependent Task Offloading (DTO) in Mobile Edge Computing. All `.py` source files live at the repo root — there is no monorepo or subpackage structure.

### Key commands

See `README.md` for the full list. Quick reference:

- **Smoke test**: `python smoke_test.py` (note: `test_for_reset()` has an API mismatch with the current `DTO_env.step()` signature; the `main()` function works)
- **Single model training**: `python final_training.py joint` (also `two_stage`, `dtodrl`, `dtodrl_tf`)
- **Comparison experiments**: `python final_training.py comparison`
- **TensorBoard**: `tensorboard --logdir runs`

### Dependency caveats

- `stable-baselines3` and `sb3-contrib` must be pinned to `2.4.0`. The codebase's custom policy classes pass extra kwargs (`use_cp`, etc.) through `**kwargs` to `super().__init__()`; version 2.4.0 is the tested compatible version.
- `numpy` must be < 2.0 (required by `stable-baselines3==2.4.0`).
- PyTorch CPU-only is sufficient for development; GPU/CUDA is optional. Use `--index-url https://download.pytorch.org/whl/cpu` when installing torch for faster downloads in CI/cloud environments.
- `rich` is required at runtime for the SB3 progress bar (used by `final_training.py`).

### Known issues

- `DTODRLMaskablePolicy` cannot be instantiated standalone due to abstract methods in `TwoHeadMaskableCategoricalDistribution`; it requires running through the full `final_training.py dtodrl` pipeline.
- `smoke_test.py`'s `test_for_reset()` function passes tuples to `env.step()` but the current `DTO_env.step()` expects an integer action. The smoke test's `main()` function does work correctly.

### Training performance

- Training is CPU-only in the cloud VM. `final_training.py joint` with default settings (50k timesteps, 8 envs) takes ~8 minutes per rollout iteration at ~100 fps. Reduce `total_timesteps` for quicker iterations.
