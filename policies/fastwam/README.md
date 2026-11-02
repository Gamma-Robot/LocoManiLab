# FastWAM

Use `--model fastwam`. This directory is a guide and workspace example; execution stays in the shared `lab` CLI, [fastwam adapter](../../src/locomanilab/models/fastwam.py), and [manifest](../../src/locomanilab/configs/fastwam.yaml). It adds no training wrapper or source checkout.

The source profile `fastwam` selects [LocoManiLab/FastWAM](https://github.com/LocoManiLab/FastWAM) at **`fec4bda864b14f61d11b2579b6b698109b65d249`**. The main repository's gitlink is authoritative; a floating upstream installation is not the integration described here. Source access and the source/model licenses must be checked separately before use or redistribution.

## CPU data walkthrough

Run from the **LocoManiLab checkout root**, with the main package's `[data]` extra installed and that environment activated; see the [CPU quickstart](../../README.md#start-on-cpu). No model source, checkpoint, GPU, simulator, or hardware is needed. This block refuses an existing workspace or symlink.

<!-- cpu-smoke:start -->
```bash
(
set -eu
test ! -e workspaces/fastwam && test ! -L workspaces/fastwam || {
  echo "workspaces/fastwam already exists; choose a fresh workspace before running this example." >&2
  exit 1
}
mkdir -p workspaces/fastwam
cp policies/fastwam/workspace.example.yaml workspaces/fastwam/env.local.yaml
lab --config workspaces/fastwam/env.local.yaml doctor --stage data
lab --config workspaces/fastwam/env.local.yaml sample-data --out datasets/cpu-example
lab --config workspaces/fastwam/env.local.yaml validate workspaces/fastwam/datasets/cpu-example
lab --config workspaces/fastwam/env.local.yaml prepare workspaces/fastwam/datasets/cpu-example --merge-only --out datasets/cpu-prepared
lab --config workspaces/fastwam/env.local.yaml validate workspaces/fastwam/datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

The synthetic state[4]/action[2] sample checks the data pipeline. It is **not training data for this model**. The [workspace example](workspace.example.yaml) belongs at `workspaces/fastwam/env.local.yaml`: schema v2 resolves `../../external/FastWAM` to this checkout's source and keeps configured data, outputs, and logs in the workspace. Override `repos.fastwam` for an existing compatible source elsewhere. CLI dataset/base/checkpoint inputs are relative to the invoking directory; output paths are relative to the workspace. All examples below also run from the project root.

## Source and separate model environment

The following is an **optional source download**, outside the CPU walkthrough. The first command only prints a plan; `--apply` explicitly initializes missing sources, without weights or a model environment. Existing checkouts are preserved.

```bash
(
set -eu
python3 scripts/setup/profiles.py fastwam
python3 scripts/setup/profiles.py fastwam --apply
source_revision=$(git -C external/FastWAM rev-parse HEAD)
test "$source_revision" = fec4bda864b14f61d11b2579b6b698109b65d249 || {
  echo "Source revision differs from this recipe; review it with Git before installing. Nothing was reset." >&2
  exit 1
}
)
```

If the guard fails, inspect `git -C external/FastWAM status` and `git -C external/FastWAM diff`, preserve your changes, and review the fixed revision before selecting it with standard Git. Do not reset or clean a user-owned checkout. With a `repos.*` override, use that checkout's path for the same revision and environment checks.

Follow the fixed [environment setup](https://github.com/LocoManiLab/FastWAM/blob/fec4bda864b14f61d11b2579b6b698109b65d249/README.md#environment-setup) in a separate model terminal: a new Conda environment named `fastwam`, Python 3.10, PyTorch `2.7.1+cu128` / torchvision `0.22.1+cu128`, then `python -m pip install -e .` from `external/FastWAM`. Preserve an existing environment rather than recreating it; use `conda_envs.fastwam` for an alternate name. Install `pyzmq`, `msgpack`, and `msgpack-numpy` in that environment for the compatible policy server. Native package/GPU installation has not been freshly validated here.

Explicitly provision the Wan2.2 components and ActionDiT backbone using the pinned [model preparation guide](https://github.com/LocoManiLab/FastWAM/blob/fec4bda864b14f61d11b2579b6b698109b65d249/README.md#model-preparation). This adapter uses `external/FastWAM/checkpoints` as `DIFFSYNTH_MODEL_BASE_PATH`, including `checkpoints/Wan-AI/Wan2.2-TI2V-5B` (where DiffSynth places `Wan-AI/Wan2.2-TI2V-5B`) and the configured ActionDiT backbone. These large local assets are separate from source initialization. A real `convert` can build the backbone on CUDA and may download absent Wan components; do not run it as a CPU-only data check. Keep the main CLI environment activated for:

```bash
lab --config workspaces/fastwam/env.local.yaml doctor --model fastwam --stage train
```

## SONIC data and command plans

The current SONIC recipe consumes prepared LeRobot **v2.1** data with RGB `observation.images.ego_view`, state[43], task prompts, and absolute action[78] = motion tokens[64] + hand targets[7+7]. See the fixed [custom_1cam data recipe](https://github.com/LocoManiLab/FastWAM/blob/fec4bda864b14f61d11b2579b6b698109b65d249/configs/data/custom_1cam.yaml) for column names, image processing, temporal sampling, and normalization. Its recorded-rate/stride assumptions must match the demonstrations; the example is not a universal video/action recipe. Retain the recording controller and SONIC checkpoint variant. Additional manifest variants have not been established as end-to-end recipes here.

Use your own prepared training copy at `workspaces/fastwam/datasets/robot/prepared`. Real conversion writes **`cache_t5/` inside this input dataset**, as well as the backbone in the source checkout's `checkpoints/`; preserve an immutable raw collection separately if needed. Inspect first:

```bash
lab --config workspaces/fastwam/env.local.yaml validate workspaces/fastwam/datasets/robot/prepared
lab --config workspaces/fastwam/env.local.yaml convert --model fastwam --dataset workspaces/fastwam/datasets/robot/prepared --dry-run
lab --config workspaces/fastwam/env.local.yaml train --model fastwam --dataset workspaces/fastwam/datasets/robot/prepared --gpus 1 --gpu-ids 0 --output-dir outputs/fastwam/first-run --dry-run
```

Conversion runs native [backbone preprocessing](https://github.com/LocoManiLab/FastWAM/blob/fec4bda864b14f61d11b2579b6b698109b65d249/scripts/preprocess_action_dit_backbone.py) when needed, then [text-embedding preparation](https://github.com/LocoManiLab/FastWAM/blob/fec4bda864b14f61d11b2579b6b698109b65d249/scripts/precompute_text_embeds.py). Training delegates to [train_zero1.sh](https://github.com/LocoManiLab/FastWAM/blob/fec4bda864b14f61d11b2579b6b698109b65d249/scripts/train_zero1.sh) with the requested dataset/cache paths. For this adapter **`--task` selects a native Hydra recipe**, not a run label; leave its SONIC default `custom_1cam_1e-4` and use `--output-dir` to name the run. The relative output above means `workspaces/fastwam/outputs/fastwam/first-run`. Removing `--dry-run` is an explicit GPU/model operation.

## Deployment boundary

**`lab deploy --model fastwam` selects the native full-robot WBC launcher.** This native pipeline is this project's **recommended best practice for existing complete FastWAM robot deployment**. It additionally needs the separately installed `GR00T-WholeBodyControl` runtime, its cameras/SDK/control environment, and the Isaac-GR00T source required by that deployment profile. The optional `fastwam-real` profile pins three repositories: FastWAM `fec4bda864b14f61d11b2579b6b698109b65d249`, Isaac-GR00T `7e73a6fb8813de6f31850e94d4ae0f94a73e6e40`, and WholeBodyControl `e7b1a32e918882808f9392d7d4ee9b027985ff66`; see [deployment setup](../../docs/deployment.md). This guide does not initialize those extra sources or operate hardware.

For an existing local trained run, this is **only the robot-launch command plan**:

```bash
lab --config workspaces/fastwam/env.local.yaml deploy --model fastwam --run-dir workspaces/fastwam/outputs/fastwam/first-run --base-model external/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B --prompt 'open the drawer' --t5-mode encode --dry-run
```

An independent policy server is available through the adapter's evaluation/inference hook, using native [run_fastwam_server.py](https://github.com/LocoManiLab/FastWAM/blob/fec4bda864b14f61d11b2579b6b698109b65d249/deployment/run_fastwam_server.py); it does not use the WBC launcher. A run contains saved configuration and `checkpoints/weights/step_*.pt`; select an existing run/step. `encode` loads T5 for the prompt. `cache` instead requires the exact prompt embedding under **the run's** `cache_t5`, not merely the dataset cache created by conversion.

## Optional simulation plan

The CPU walkthrough does not provision simulation. Separately initialize the simulator/controller sources, their environment, rendering dependencies, and matching controller weights using [runtime setup](../../docs/data_collection.md) and [deployment guidance](../../docs/deployment.md). Model-server and simulator environments stay separate. For demonstrations recorded with SONIC `v1_1`, inspect this plan only after those paths/manifests exist:

```bash
lab --config workspaces/fastwam/env.local.yaml eval --model fastwam --sim --run-dir workspaces/fastwam/outputs/fastwam/first-run --base-model external/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B --prompt 'open the drawer' --backend sonic --sonic-ckpt v1_1 --env EmptyEnv --episodes 1 --max-steps 60 --video-dir runs/eval/fastwam --dry-run
```

Select the actual SONIC checkpoint variant used for your demonstrations. `EmptyEnv` checks wiring, not task success. A dry-run is a static command plan; it does not check readiness, load weights, or execute a policy. Simulation and hardware collection use `g1_backend`; complete model deployment has separate [runtime requirements](../../docs/deployment.md#robot-runtimes).

## Validation and common issues

The CPU data/configuration walkthrough and command/data-contract checks are separate from model execution. The fixed source entry points were inspected, but fresh installation of the full model environment, GPU training, learned-policy inference, simulation task success, and real hardware have **not** been validated for this guide. A successful `doctor` or `--dry-run` does not establish those results.

- Missing backbone/base assets: provision the fixed native model layout; source initialization alone does not provide weights, and real conversion may otherwise download them.
- Wrong training run name: use `--output-dir`, not `--task first-run`; `--task` names a native configuration.
- T5 cache/prompt mismatch: keep the dataset cache for training; serving cache mode requires its own matching run cache. Use explicit `encode` when no run cache exists.
- Unexpected cameras/tmux/WBC prerequisites on deploy: that command is the full native robot path. Use the evaluation server hook for simulation, and inspect the deployment boundary before real execution.
