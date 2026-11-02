# Training locally

Training runs in the selected model's own environment. LocoManiLab constructs the native command, validates the dataset contract where possible, and records process output. It does not provision a cluster, upload data, or download a checkpoint as part of training.

1. Select a model in the [support matrix](models.md) and initialize its source profile.
2. Provision the pinned repository's environment and acquire its base model under the relevant terms.
3. Set local checkout, converted-data, and checkpoint paths in workspace configuration.
4. Validate/convert the dataset, inspect `--dry-run`, then run a short model-specific check before a full training run.

The synthetic CPU dataset checks data tooling only and does not satisfy the model adapters' robot-data contracts. Use representative validated demonstrations with the selected controller's state/action layout.

## ACT and DP example

ACT and DP share a flat-action converter and the LeRobot environment. Provision that environment in the pinned `repos.lerobot` checkout using its own instructions; LocoManiLab invokes it with `uv run --no-sync`. Merge these settings into your generated workspace `env.local.yaml`, replacing the checkout path with your own:

```yaml
repos:
  lerobot: /path/to/LocoManiLab/external/lerobot
paths:
  lerobot_data_root: datasets/lerobot
  lerobot_outputs: checkpoints/lerobot
```

For schema v2 workspaces, the relative data/output paths are workspace-relative. The source demonstrations below are a placeholder for real robot data, with the action columns expected by the selected adapter; they are not the CPU sample.

```bash
lab doctor --model act --stage train
lab validate /path/to/robot-demonstrations
lab convert --model act --dataset /path/to/robot-demonstrations --dry-run
# Inspect the plan, then create the converted dataset:
lab convert --model act --dataset /path/to/robot-demonstrations
lab train --model act --dataset /path/to/robot-demonstrations --gpus 1 --dry-run
```

Keep passing the same source dataset to `train`: the adapter selects its converted copy under `paths.lerobot_data_root`. `--dry-run` does not perform conversion or train a model. Run a short training check only after conversion succeeds and the emitted command matches your data/environment.

## Other adapters

GR00T and starVLA read the shared format directly. Other adapters may require a real conversion or additional metadata; see their manifests and model notes.

PRTS currently accepts only the checked SONIC absolute state[43]/action[78] contract. Its pinned serving integration cannot decode HeFT/AMO actions, so the adapter refuses those datasets before conversion/training even though older native recipes exist.

`--set key=value` is native-configuration passthrough, not a portable cross-model option. Inspect the emitted command and the selected native entry point. Keep output paths local and explicit. For model families that require action-head widening, use the documented integration's checkpoint preparation tool before training; changing a dimension flag alone does not make released weights compatible.

The supported pi0.5 integration uses its JAX path. Select a literal native recipe whose controller and `data.repo_id` match your dataset with `--config-name`; `convert` computes dataset-bound normalization statistics before training. Use that same recipe for `deploy`, `eval`, and `infer-check`. Its alternate PyTorch base-weight path is not established here. Legacy online RL is unavailable because the recorded simulator commit lacks the expected training entry point.
