# DM0.5 (DM05)

Use `--model dm05`. This directory is a guide and workspace example; execution stays in the shared `lab` CLI, [dm05 adapter](../../src/locomanilab/models/dm05.py), and [manifest](../../src/locomanilab/configs/dm05.yaml). It adds no training wrapper or source checkout.

The source profile `dm05` selects [Dexmal/OpenDM](https://github.com/Dexmal/OpenDM) (upstream, unmodified) at **`a6016e48a4ebcb21bb8496cd9ce830d28f6f51cf`**. The main repository's gitlink is authoritative; a floating upstream installation is not the integration described here. Source access and the source/model licenses must be checked separately before use or redistribution.

## CPU data walkthrough

Run from the **LocoManiLab checkout root**, with the main package's `[data]` extra installed and that environment activated; see the [CPU quickstart](../../README.md#start-on-cpu). No model source, checkpoint, GPU, simulator, or hardware is needed. This block refuses an existing workspace or symlink.

<!-- cpu-smoke:start -->
```bash
(
set -eu
test ! -e workspaces/dm05 && test ! -L workspaces/dm05 || {
  echo "workspaces/dm05 already exists; choose a fresh workspace before running this example." >&2
  exit 1
}
mkdir -p workspaces/dm05
cp policies/dm05/workspace.example.yaml workspaces/dm05/env.local.yaml
lab --config workspaces/dm05/env.local.yaml doctor --stage data
lab --config workspaces/dm05/env.local.yaml sample-data --out datasets/cpu-example
lab --config workspaces/dm05/env.local.yaml validate workspaces/dm05/datasets/cpu-example
lab --config workspaces/dm05/env.local.yaml prepare workspaces/dm05/datasets/cpu-example --merge-only --out datasets/cpu-prepared
lab --config workspaces/dm05/env.local.yaml validate workspaces/dm05/datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

The synthetic state[4]/action[2] sample checks the data pipeline. It is **not training data for this model**. The [workspace example](workspace.example.yaml) belongs at `workspaces/dm05/env.local.yaml`: schema v2 resolves `../../external/opendm` to this checkout's source and keeps configured data, outputs, and logs in the workspace. Override `repos.opendm` for an existing compatible source elsewhere. CLI dataset/base/checkpoint inputs are relative to the invoking directory; output paths are relative to the workspace. All examples below also run from the project root.

## Source and separate model environment

The following is an **optional source download**, outside the CPU walkthrough. The first command only prints a plan; `--apply` explicitly initializes missing sources, without weights or a model environment. Existing checkouts are preserved.

```bash
(
set -eu
python3 scripts/setup/profiles.py dm05
python3 scripts/setup/profiles.py dm05 --apply
source_revision=$(git -C external/opendm rev-parse HEAD)
test "$source_revision" = a6016e48a4ebcb21bb8496cd9ce830d28f6f51cf || {
  echo "Source revision differs from this recipe; review it with Git before installing. Nothing was reset." >&2
  exit 1
}
)
```

If the guard fails, inspect `git -C external/opendm status` and `git -C external/opendm diff`, preserve your changes, and review the fixed revision before selecting it with standard Git. Do not reset or clean a user-owned checkout. With a `repos.*` override, use that checkout's path for the same revision and environment checks.

Use the fixed OpenDM [local installation guide](https://github.com/Dexmal/OpenDM/blob/a6016e48a4ebcb21bb8496cd9ce830d28f6f51cf/README.md#local-installation) in a **separate model terminal**: create a new Conda environment named `opendm` with Python 3.10, install the CUDA/PyTorch and FlashAttention dependencies specified there, then run `python -m pip install -e .` from `external/opendm`. Preserve an existing environment and inspect it instead of recreating it; an alternate name belongs in `conda_envs.dm05`. The fresh GPU dependency resolution/build has not been verified here. Install `pyarrow`, `pyzmq`, `msgpack`, and `msgpack-numpy` in that same environment for the packaged conversion/server helpers. The converter also uses system `ffprobe` from FFmpeg. OpenDM's optional `fast-infer`/TensorRT backend is not required by this recipe.

Obtain a complete local DM05 base using the pinned [checkpoint instructions](https://github.com/Dexmal/OpenDM/blob/a6016e48a4ebcb21bb8496cd9ce830d28f6f51cf/docs/en/dm05_inference.md#dm05-base-pretrained-model). Use `workspaces/dm05/assets/DM05` for this guide. The released 32-D action interface must be widened before G1 SONIC fine-tuning. With the model environment already installed, run this from your active main CLI terminal. It resolves the packaged helper for the separate model interpreter and writes a **new** checkpoint directory:

```bash
(
set -eu
test ! -e workspaces/dm05/assets/DM05-sonic78 && test ! -L workspaces/dm05/assets/DM05-sonic78
widen_helper=$(python -c 'from locomanilab.runtime import script_path; print(script_path("dm05/widen.py"))')
conda run -n opendm python "$widen_helper" --src workspaces/dm05/assets/DM05 --dst workspaces/dm05/assets/DM05-sonic78 --action-dim 78
)
```

This tool expects `config.json` and a single `model.safetensors`, not an arbitrary sharded or adapter-only checkpoint. Its implementation and checks live in the [packaged widening helper](../../src/locomanilab/runtime/dm05/widen.py). Keep the main CLI environment activated in your workflow terminal:

```bash
lab --config workspaces/dm05/env.local.yaml doctor --model dm05 --stage train
```

## SONIC data and command plans

Provide prepared LeRobot **v2.1** data: RGB `observation.images.ego_view`, state[43] in joint/hand order, and absolute action[78] formed from `action.motion_token[64]`, `teleop.left_hand_joints[7]`, and `teleop.right_hand_joints[7]`. Preserve episode/task metadata, statistics, and the SONIC checkpoint provenance. Prompts must describe the demonstrated task. Other controller layouts are not this recipe.

**Current serving restriction:** the manifest selects `dataset_name=locomanilab_train50_merged`. To keep conversion, training normalization, and serving aligned, name your real prepared dataset **`train50_merged`**, as below. The generated registry record is `locomanilab_train50_merged`; existing conversion records retain their recorded registration name when reused for training or serving. This page does not claim arbitrary dataset names can be served through the current adapter. Keep the converted record and original source videos available while training/serving.

After provisioning the model environment and the real dataset, inspect these plans; they do not execute conversion, training, or serving:

```bash
lab --config workspaces/dm05/env.local.yaml validate workspaces/dm05/datasets/robot/train50_merged
lab --config workspaces/dm05/env.local.yaml convert --model dm05 --dataset workspaces/dm05/datasets/robot/train50_merged --dry-run
lab --config workspaces/dm05/env.local.yaml train --model dm05 --dataset workspaces/dm05/datasets/robot/train50_merged --gpus 1 --gpu-ids 0 --task first-run --dry-run
lab --config workspaces/dm05/env.local.yaml deploy --model dm05 --model-path workspaces/dm05/outputs/dm05/first-run/checkpoint-10000 --prompt 'open the drawer' --dry-run
```

Remove `--dry-run` from conversion only when you intend to write `datasets/opendm/train50_merged` in the workspace. It creates JSONL episode/registry records referencing the original videos. Training still takes the **same original source path**, resolves that conversion, and writes `outputs/dm05/first-run`; it checks the local base's action width is 78. Select an actual checkpoint containing configuration, weights, and its matching `norm_stats.json` or `norm_stats/` assets before serving. The deployment command starts a compatible policy server on port 5550; it does not start cameras or robot control.

The adapter calls OpenDM's native [DM05 launcher](https://github.com/Dexmal/OpenDM/blob/a6016e48a4ebcb21bb8496cd9ce830d28f6f51cf/script/dm05_launcher.sh) with LocoManiLab's absolute packaged [experiment](../../src/locomanilab/runtime/dm05/experiment.py), [converter](../../src/locomanilab/runtime/dm05/convert.py), [registry](../../src/locomanilab/runtime/dm05/registry/locomanilab_dm05_datasets.py), and [server](../../src/locomanilab/runtime/dm05/server.py). Those helpers run in `opendm`; that heavy environment does **not** need a separate installation of the main CLI package. Do not substitute OpenDM's released 32-D training experiment for the packaged 78-D experiment.

## Released checkpoint

A DM0.5 checkpoint fine-tuned on the benchmark's train50 split (step 50000, 23 GB) is released on the Hugging Face Hub with its `config.json`, `model.safetensors`, `norm_stats.json`, and tokenizer files; see [data, assets, and checkpoints](../../docs/hf_resources.md). It contains no optimizer state and cannot resume training. Serve it from the `opendm` environment like a local checkpoint:

```bash
python scripts/hf/download.py checkpoint dm05 --out checkpoints/hf
lab --config workspaces/dm05/env.local.yaml deploy --model dm05 --model-path checkpoints/hf/checkpoint/dm05 --prompt 'open the fridge door in front of you' --dry-run
```

Serving the released checkpoint has not been validated for this guide.

## Optional simulation plan

The CPU walkthrough does not provision simulation. Separately initialize the simulator/controller sources, their environment, rendering dependencies, and matching controller weights using [runtime setup](../../docs/data_collection.md) and [deployment guidance](../../docs/deployment.md). Model-server and simulator environments stay separate. For demonstrations recorded with SONIC `v1_1`, inspect this plan only after those paths/manifests exist:

```bash
lab --config workspaces/dm05/env.local.yaml eval --model dm05 --sim --model-path workspaces/dm05/outputs/dm05/first-run/checkpoint-10000 --prompt 'open the drawer' --backend sonic --sonic-ckpt v1_1 --env EmptyEnv --episodes 1 --max-steps 60 --video-dir runs/eval/dm05 --dry-run
```

Select the actual SONIC checkpoint variant used for your demonstrations. `EmptyEnv` checks wiring, not task success. A dry-run is a static command plan; it does not check readiness, load weights, or execute a policy. Simulation and hardware collection use `g1_backend`; complete model deployment has separate [runtime requirements](../../docs/deployment.md#robot-runtimes).

## Validation and common issues

The CPU data/configuration walkthrough and command/data-contract checks are separate from model execution. The fixed source entry points were inspected, but fresh installation of the full model environment, GPU training, learned-policy inference, simulation task success, and real hardware have **not** been validated for this guide. A successful `doctor` or `--dry-run` does not establish those results.

- Missing `opendm_dataset.json` or unknown dataset profile: run the real conversion first, keep `paths.dm05_data_root` unchanged, and use the documented `train50_merged` basename.
- Action width 32 or absent local base: widen a supported complete base into a new directory and check `paths.dm05_pretrained`; the CPU sample does not solve this mismatch.
- Missing videos, `ffprobe`, transport imports, or normalization: check these within the model environment and keep source/converted/statistics assets together.
- Offline asset failure: defaults use offline Hub access; provision assets explicitly instead of relying on an implicit download during training.
