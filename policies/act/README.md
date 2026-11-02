# ACT

ACT uses the shared [LeRobot adapter](../../src/locomanilab/models/lerobot_baseline.py), [ACT manifest](../../src/locomanilab/configs/act.yaml), and [packaged conversion/server helpers](../../src/locomanilab/runtime/lerobot). This directory contains instructions and a workspace example; `lab` remains the execution entry point. [DP](../dp/README.md) uses the same source checkout, environment, and converted-data format.

The selected source is [Hugging Face LeRobot](https://github.com/huggingface/lerobot) (upstream, unmodified), at **v0.3.3 / `b883328e6c95681ca90a18b102e4ae5e1f91e2bf`**. The main repository's gitlink is authoritative. This revision has the v2.1 dataset loader and policy normalization APIs used here; newer LeRobot versions require a separate integration review. See the pinned [installation instructions](https://github.com/huggingface/lerobot/blob/b883328e6c95681ca90a18b102e4ae5e1f91e2bf/README.md#installation) and [license](https://github.com/huggingface/lerobot/blob/b883328e6c95681ca90a18b102e4ae5e1f91e2bf/LICENSE).

## CPU data walkthrough

Run from the **LocoManiLab checkout root**, after installing the main package with its `[data]` extra and activating that environment as described in the [project quickstart](../../README.md#start-on-cpu). No model source checkout, weights, simulator, or GPU is needed. This block requires a new `workspaces/act` directory and refuses an existing workspace or symlink.

<!-- cpu-smoke:start -->
```bash
(
set -eu
test ! -e workspaces/act && test ! -L workspaces/act || {
  echo "workspaces/act already exists; choose a fresh workspace before running this example." >&2
  exit 1
}
mkdir -p workspaces/act
cp policies/act/workspace.example.yaml workspaces/act/env.local.yaml
lab --config workspaces/act/env.local.yaml doctor --stage data
lab --config workspaces/act/env.local.yaml sample-data --out datasets/cpu-example
lab --config workspaces/act/env.local.yaml validate workspaces/act/datasets/cpu-example
lab --config workspaces/act/env.local.yaml prepare workspaces/act/datasets/cpu-example --merge-only --out datasets/cpu-prepared
lab --config workspaces/act/env.local.yaml validate workspaces/act/datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

This exercises data generation, validation, and preparation. The synthetic sample has state[4]/action[2]; it is **not** an ACT robot-training example. A model `--dry-run` prints a command and does not prove the backend environment, data contract, or checkpoint works.

The [workspace example](workspace.example.yaml) assumes it is copied to `workspaces/act/env.local.yaml`. Under schema v2, its `../../external/lerobot` resolves to this checkout's `external/lerobot`, while `datasets/`, `outputs/`, and `runs/` stay inside `workspaces/act`. Override `repos.*` for checkouts elsewhere. CLI dataset/checkpoint inputs are relative to your current directory; these examples always run from the project root.

## Install the shared LeRobot environment

These are **optional, explicit installation steps**. They can download Python/packages, including PyTorch, and are separate from the CPU walkthrough. Install `uv` first; follow the pinned upstream instructions for platform/CUDA and FFmpeg/PyAV requirements. ACT and DP use one environment at `external/lerobot/.venv`; do not install it into the main CLI environment.

```bash
(
set -eu
# First command only prints the source plan; --apply explicitly initializes it.
python3 scripts/setup/profiles.py act
python3 scripts/setup/profiles.py act --apply
lerobot_revision=$(git -C external/lerobot rev-parse HEAD)
test "$lerobot_revision" = b883328e6c95681ca90a18b102e4ae5e1f91e2bf || {
  echo "LeRobot revision differs from this recipe; review it with Git before installing. Nothing was reset." >&2
  exit 1
}
if test -e external/lerobot/.venv || test -L external/lerobot/.venv; then
  echo "Keeping existing external/lerobot/.venv; inspect its dependencies separately. No packages installed."
else
  uv venv --python 3.10 external/lerobot/.venv
  uv pip install --python external/lerobot/.venv/bin/python -e external/lerobot pyarrow pandas av pyzmq msgpack msgpack-numpy
fi
)
```

Python 3.10 or 3.11 can be selected for this backend; the pinned project declares Python >=3.10. The upstream dependency constraints apply to both ACT and DP. `pyzmq`, `msgpack`, and `msgpack-numpy` provide the compatible server transport; `pyarrow`, pandas, and PyAV support conversion. `lab` uses `uv run --no-sync` in `repos.lerobot`, so execution does not install missing dependencies for you. The block creates a new environment only; an existing `.venv`, including a symlink, is preserved without package changes and must be checked separately.

The profile helper preserves existing source checkouts. If the revision guard fails, inspect `git -C external/lerobot status` and `git -C external/lerobot diff`. Preserve or commit your intentional changes yourself. Once the checkout is clean and you have reviewed the change, `git -C external/lerobot switch --detach b883328e6c95681ca90a18b102e4ae5e1f91e2bf` selects this exact integration revision. No automatic reset, clean, stash, or floating-branch update is part of this recipe. `repos.lerobot` can instead point to a separate compatible checkout; provision that checkout's `.venv` and adjust the installation commands accordingly.

For the ACT workspace created above, check the selected paths and prerequisites:

```bash
lab --config workspaces/act/env.local.yaml doctor --model act --stage train
```

`doctor` does not load a trained model or validate CUDA operation. ACT's default ImageNet ResNet18 backbone may require an initial weight download/cache during real training. A full pretrained ACT policy is not required by this training recipe; deployment does require a locally saved policy checkpoint.

## Robot-data contract and training

This recipe covers **G1 with SONIC**, not every G1 controller. Provide prepared LeRobot **v2.1** demonstrations with:

- RGB video `observation.images.ego_view` and `observation.state[43]` in the collector's joint/hand order.
- `action.motion_token[64]`, `teleop.left_hand_joints[7]`, and `teleop.right_hand_joints[7]`; conversion concatenates them, in that order, into `action[78]`.
- Matching episode/task metadata, statistics, and the provenance of the SONIC checkpoint variant that produced the tokens.

ACT is not language-conditioned; accepted prompts are ignored. Equal vector widths do not make another controller or SONIC checkpoint variant interchangeable. The source dataset and its videos must remain available: conversion links videos into the converted copy.

Place your real prepared dataset at `workspaces/act/datasets/robot/prepared`, or replace that path below. Once the backend environment is provisioned:

```bash
lab --config workspaces/act/env.local.yaml validate workspaces/act/datasets/robot/prepared
lab --config workspaces/act/env.local.yaml convert --model act --dataset workspaces/act/datasets/robot/prepared --dry-run
# Inspect the command, then create the converted copy:
lab --config workspaces/act/env.local.yaml convert --model act --dataset workspaces/act/datasets/robot/prepared
lab --config workspaces/act/env.local.yaml train --model act --dataset workspaces/act/datasets/robot/prepared --gpus 1 --gpu-ids 0 --task first-run --set steps=100 --set save_freq=100 --dry-run
# Explicit short GPU training run; inspect data/command before executing:
lab --config workspaces/act/env.local.yaml train --model act --dataset workspaces/act/datasets/robot/prepared --gpus 1 --gpu-ids 0 --task first-run --set steps=100 --set save_freq=100
```

Keep passing the **same source path** to `train`. The adapter selects `workspaces/act/datasets/lerobot/prepared`; the example run writes `workspaces/act/outputs/lerobot/act/first-run`. The adapter supports one GPU per training process. Use a new `--task` for a new run; existing output directories are protected. A short run is a local integration check, not a task-success benchmark.

Converted data is bound to its source path and metadata. A different source with the same directory name, changed source metadata, or an old conversion without a source record is refused rather than silently reused. Select a fresh `paths.lerobot_data_root` (for example `datasets/lerobot-v2`) and convert again; keep the original demonstrations unchanged. The adapter does not automatically overwrite an existing conversion.

Use `--set` for native training hyperparameters. `dataset`, `dataset.root`, and
`dataset.repo_id` are reserved so an override cannot select unverified converted
data; choose the original source through `--dataset` instead.

## Local server and optional simulation

After training has produced a checkpoint, inspect and explicitly start the model server:

```bash
lab --config workspaces/act/env.local.yaml deploy --model act --model-path workspaces/act/outputs/lerobot/act/first-run --dry-run
lab --config workspaces/act/env.local.yaml deploy --model act --model-path workspaces/act/outputs/lerobot/act/first-run
```

This starts a policy server on port 5550; it does not connect a robot. A run directory selects its newest saved `pretrained_model`; pass an explicit saved checkpoint directory when you need a fixed revision. Stop your server before running `eval`, which starts its own server.

Simulation additionally needs the `sim` source profile, the simulator's separately provisioned environment, controller weights, and rendering dependencies; see [collection/runtime setup](../../docs/data_collection.md) and [deployment boundaries](../../docs/deployment.md). Do not run profile `--apply` or download assets as part of the CPU walkthrough. For data recorded with SONIC `v1_1`, an illustrative plan is:

```bash
lab --config workspaces/act/env.local.yaml eval --model act --sim --model-path workspaces/act/outputs/lerobot/act/first-run --backend sonic --sonic-ckpt v1_1 --env EmptyEnv --episodes 1 --max-steps 60 --video-dir runs/eval/act --dry-run
```

Choose the SONIC variant matching your data. Remove `--dry-run` only after separately checking the simulator/runtime and checkpoint. `EmptyEnv` checks wiring; choose a provisioned task for an actual task evaluation. Hardware collection/control belongs to `g1_backend`; this page does not establish real-robot readiness.

## Validation and common issues

The CPU data walkthrough, configuration/path resolution, converter behavior, and command construction are checked independently of the model environment. The pinned v0.3.3 source/API compatibility was inspected. Fresh GPU training, learned-policy inference, task success, and hardware operation have **not** been validated for this pilot.

- `repository:lerobot` or backend-environment failure: initialize the source and provision its `.venv`, or correct `repos.lerobot`; installing the main package is insufficient.
- Missing SONIC columns: use matching real demonstrations; the generic CPU sample and another controller's actions are not substitutes.
- Wrong or stale converted copy: keep the same source path for conversion and training; source changes require a fresh converted-data root. For ACT/DP sharing, see the [DP note](../dp/README.md#data-and-training).
- Existing data/run output: retain it and choose a new destination/task. Resume requires a compatible native saved training configuration; `--set resume=true` alone is not a complete resume recipe.
- Hub/network failure: the adapter defaults to offline Hub access; provision required assets/caches explicitly, or deliberately set `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0` for an authorized online run.

Defaults and behavior remain in the [manifest](../../src/locomanilab/configs/act.yaml), [adapter](../../src/locomanilab/models/lerobot_baseline.py), and [model status matrix](../../docs/models.md); this directory adds no training implementation.
