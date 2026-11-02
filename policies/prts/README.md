# PRTS

Use `--model prts`. This directory is a guide and workspace example; execution stays in the shared `lab` CLI, [prts adapter](../../src/locomanilab/models/prts.py), and [manifest](../../src/locomanilab/configs/prts.yaml). It adds no training wrapper or source checkout.

The source profile `prts` selects [LocoManiLab/PRTS](https://github.com/LocoManiLab/PRTS) at **`1100999f565e8af3ef597456e24b12e50b1dc25c`**. The main repository's gitlink is authoritative; a floating upstream installation is not the integration described here. Source access and the source/model licenses must be checked separately before use or redistribution.

## CPU data walkthrough

Run from the **LocoManiLab checkout root**, with the main package's `[data]` extra installed and that environment activated; see the [CPU quickstart](../../README.md#start-on-cpu). No model source, checkpoint, GPU, simulator, or hardware is needed. This block refuses an existing workspace or symlink.

<!-- cpu-smoke:start -->
```bash
(
set -eu
test ! -e workspaces/prts && test ! -L workspaces/prts || {
  echo "workspaces/prts already exists; choose a fresh workspace before running this example." >&2
  exit 1
}
mkdir -p workspaces/prts
cp policies/prts/workspace.example.yaml workspaces/prts/env.local.yaml
lab --config workspaces/prts/env.local.yaml doctor --stage data
lab --config workspaces/prts/env.local.yaml sample-data --out datasets/cpu-example
lab --config workspaces/prts/env.local.yaml validate workspaces/prts/datasets/cpu-example
lab --config workspaces/prts/env.local.yaml prepare workspaces/prts/datasets/cpu-example --merge-only --out datasets/cpu-prepared
lab --config workspaces/prts/env.local.yaml validate workspaces/prts/datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

The synthetic state[4]/action[2] sample checks the data pipeline. It is **not training data for this model**. The [workspace example](workspace.example.yaml) belongs at `workspaces/prts/env.local.yaml`: schema v2 resolves `../../external/PRTS` to this checkout's source and keeps configured data, outputs, and logs in the workspace. Override `repos.prts` for an existing compatible source elsewhere. CLI dataset/base/checkpoint inputs are relative to the invoking directory; output paths are relative to the workspace. All examples below also run from the project root.

## Source and separate model environment

The following is an **optional source download**, outside the CPU walkthrough. The first command only prints a plan; `--apply` explicitly initializes missing sources, without weights or a model environment. Existing checkouts are preserved.

```bash
(
set -eu
python3 scripts/setup/profiles.py prts
python3 scripts/setup/profiles.py prts --apply
source_revision=$(git -C external/PRTS rev-parse HEAD)
test "$source_revision" = 1100999f565e8af3ef597456e24b12e50b1dc25c || {
  echo "Source revision differs from this recipe; review it with Git before installing. Nothing was reset." >&2
  exit 1
}
)
```

If the guard fails, inspect `git -C external/PRTS status` and `git -C external/PRTS diff`, preserve your changes, and review the fixed revision before selecting it with standard Git. Do not reset or clean a user-owned checkout. With a `repos.*` override, use that checkout's path for the same revision and environment checks.

Use the pinned [native environment guide](https://github.com/LocoManiLab/PRTS/blob/1100999f565e8af3ef597456e24b12e50b1dc25c/README.md#1-environment) in a separate model terminal: create a **new Conda environment `prts`, Python 3.11**, install `lerobot==0.3.3` first, install `requirements.txt`, then `python -m pip install -e .` from `external/PRTS`. Follow its CUDA/PyTorch/transformers compatibility requirements. Preserve an existing environment rather than recreating it; `conda_envs.prts` may select another name. Add `pyzmq`, `msgpack`, and `msgpack-numpy` there for the compatible server. The native guide discusses FlashAttention; this adapter explicitly selects **SDPA**, so do not silently change the attention/position configuration while comparing runs. Fresh GPU dependency installation has not been verified here.

Provision a complete local **`TeleEmbodied/PRTS-4B`** checkpoint at `workspaces/prts/assets/PRTS-4B`. The fixed native [checkpoint guide](https://github.com/LocoManiLab/PRTS/blob/1100999f565e8af3ef597456e24b12e50b1dc25c/README.md#2-download-the-pre-trained-checkpoint) also describes the Qwen3-VL assets. Training/serving additionally load the **`physical-intelligence/fast` action tokenizer**: the fixed server expects its complete files directly under `<HF_HUB_CACHE>/models--physical-intelligence--fast`, not just a Hub snapshot reference. Provision that cache explicitly, and use the same `HF_HUB_CACHE` during training and serving; its default here is `~/.cache/huggingface/hub`. Check tokenizer/model terms separately. This is an asset requirement, not an automatic part of source initialization.

From the main CLI environment:

```bash
lab --config workspaces/prts/env.local.yaml doctor --model prts --stage train
```

## Verified contract and command plans

**This adapter currently permits only SONIC absolute state[43]/action[78].** HeFT and AMO conversion/training are refused before starting the backend: their manifest entries do not imply compatible serving. Provide prepared LeRobot **v2.1** data with RGB `observation.images.ego_view`, `observation.state[43]`, and absolute `action.motion_token[64]` + `teleop.left_hand_joints[7]` + `teleop.right_hand_joints[7]`. Preserve modality slices, feature widths, task/episode metadata, and the SONIC checkpoint variant. Tokens are not state-relative actions.

The adapter generates a workspace recipe bound to your requested dataset; it does not train on the native example's preselected dataset. An optional `--config-name` YAML template must satisfy the same checked column/group/width/absolute-action contract. Changing a column name or vector width does not establish a new controller contract.

With a real prepared training copy at `workspaces/prts/datasets/robot/prepared`, inspect:

```bash
lab --config workspaces/prts/env.local.yaml validate workspaces/prts/datasets/robot/prepared
lab --config workspaces/prts/env.local.yaml convert --model prts --dataset workspaces/prts/datasets/robot/prepared --dry-run
lab --config workspaces/prts/env.local.yaml train --model prts --dataset workspaces/prts/datasets/robot/prepared --gpus 1 --gpu-ids 0 --task first-run --dry-run
lab --config workspaces/prts/env.local.yaml deploy --model prts --model-path workspaces/prts/outputs/prts/first-run/checkpoint-10000 --prompt 'open the drawer' --dry-run
```

Real conversion writes the selected recipe under workspace `artifacts/prts/recipes/` and invokes native [compute_stats.py](https://github.com/LocoManiLab/PRTS/blob/1100999f565e8af3ef597456e24b12e50b1dc25c/scripts/compute_stats.py); the native script writes dataset normalization statistics when absent. Use a training copy if raw collections must remain immutable. Dry-run validates the recipe and prints the plan without writing it. Train uses native [scripts/train.py](https://github.com/LocoManiLab/PRTS/blob/1100999f565e8af3ef597456e24b12e50b1dc25c/scripts/train.py), the configured local base, action width 78, SDPA, and `qwen3_vl` positional encoding, writing `workspaces/prts/outputs/prts/first-run`. Removing `--dry-run` is an explicit model/GPU operation, not a CPU walkthrough step.

Select an actual saved checkpoint directory and its companion processor, action-tokenizer, and normalization assets before serving. The compatible [run_prts_server.py](https://github.com/LocoManiLab/PRTS/blob/1100999f565e8af3ef597456e24b12e50b1dc25c/deployment/run_prts_server.py) uses the SONIC motion-token/hand split and QUANTILE state normalization matching the training recipe. The default deployment starts a policy server on port 5550, not cameras or robot control; the native websocket mode is a separate protocol.

## Optional simulation plan

The CPU walkthrough does not provision simulation. Separately initialize the simulator/controller sources, their environment, rendering dependencies, and matching controller weights using [runtime setup](../../docs/data_collection.md) and [deployment guidance](../../docs/deployment.md). Model-server and simulator environments stay separate. For demonstrations recorded with SONIC `v1_1`, inspect this plan only after those paths/manifests exist:

```bash
lab --config workspaces/prts/env.local.yaml eval --model prts --sim --model-path workspaces/prts/outputs/prts/first-run/checkpoint-10000 --prompt 'open the drawer' --backend sonic --sonic-ckpt v1_1 --env EmptyEnv --episodes 1 --max-steps 60 --video-dir runs/eval/prts --dry-run
```

Select the actual SONIC checkpoint variant used for your demonstrations. `EmptyEnv` checks wiring, not task success. A dry-run is a static command plan; it does not check readiness, load weights, or execute a policy. Simulation and hardware collection use `g1_backend`; complete model deployment has separate [runtime requirements](../../docs/deployment.md#robot-runtimes).

## Validation and common issues

The CPU data/configuration walkthrough and command/data-contract checks are separate from model execution. The fixed source entry points were inspected, but fresh installation of the full model environment, GPU training, learned-policy inference, simulation task success, and real hardware have **not** been validated for this guide. A successful `doctor` or `--dry-run` does not establish those results.

- Rejected variant or template: retain the validated SONIC absolute 43/78 contract. Do not bypass the guard to train a checkpoint the selected server cannot interpret.
- Missing quantile statistics: execute conversion in the prepared PRTS environment, preserve the correct dataset statistics, and keep checkpoint normalization assets with the saved run.
- Missing FAST tokenizer or offline model: provision the complete local files at the fixed expected cache path; a partial Hub cache or different tokenizer layout is insufficient.
- Position/normalization mismatch: preserve `qwen3_vl`, QUANTILE, and absolute actions consistently across the run and its compatible server.
