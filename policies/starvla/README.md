# starVLA

Use `--model starvla`. This directory is a guide and workspace example; execution stays in the shared `lab` CLI, [starvla adapter](../../src/locomanilab/models/starvla.py), and [manifest](../../src/locomanilab/configs/starvla.yaml). It adds no training wrapper or source checkout.

The source profile `starvla` selects [LocoManiLab/starVLA](https://github.com/LocoManiLab/starVLA) at **`9740f0dc162669983829dbc02b6acce0f3a14444`**. The main repository's gitlink is authoritative; a floating upstream installation is not the integration described here. Source access and the source/model licenses must be checked separately before use or redistribution.

## CPU data walkthrough

Run from the **LocoManiLab checkout root**, with the main package's `[data]` extra installed and that environment activated; see the [CPU quickstart](../../README.md#start-on-cpu). No model source, checkpoint, GPU, simulator, or hardware is needed. This block refuses an existing workspace or symlink.

<!-- cpu-smoke:start -->
```bash
(
set -eu
test ! -e workspaces/starvla && test ! -L workspaces/starvla || {
  echo "workspaces/starvla already exists; choose a fresh workspace before running this example." >&2
  exit 1
}
mkdir -p workspaces/starvla
cp policies/starvla/workspace.example.yaml workspaces/starvla/env.local.yaml
lab --config workspaces/starvla/env.local.yaml doctor --stage data
lab --config workspaces/starvla/env.local.yaml sample-data --out datasets/cpu-example
lab --config workspaces/starvla/env.local.yaml validate workspaces/starvla/datasets/cpu-example
lab --config workspaces/starvla/env.local.yaml prepare workspaces/starvla/datasets/cpu-example --merge-only --out datasets/cpu-prepared
lab --config workspaces/starvla/env.local.yaml validate workspaces/starvla/datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

The synthetic state[4]/action[2] sample checks the data pipeline. It is **not training data for this model**. The [workspace example](workspace.example.yaml) belongs at `workspaces/starvla/env.local.yaml`: schema v2 resolves `../../external/starVLA` to this checkout's source and keeps configured data, outputs, and logs in the workspace. Override `repos.starvla` for an existing compatible source elsewhere. CLI dataset/base/checkpoint inputs are relative to the invoking directory; output paths are relative to the workspace. All examples below also run from the project root.

## Source and separate model environment

The following is an **optional source download**, outside the CPU walkthrough. The first command only prints a plan; `--apply` explicitly initializes missing sources, without weights or a model environment. Existing checkouts are preserved.

```bash
(
set -eu
python3 scripts/setup/profiles.py starvla
python3 scripts/setup/profiles.py starvla --apply
source_revision=$(git -C external/starVLA rev-parse HEAD)
test "$source_revision" = 9740f0dc162669983829dbc02b6acce0f3a14444 || {
  echo "Source revision differs from this recipe; review it with Git before installing. Nothing was reset." >&2
  exit 1
}
)
```

If the guard fails, inspect `git -C external/starVLA status` and `git -C external/starVLA diff`, preserve your changes, and review the fixed revision before selecting it with standard Git. Do not reset or clean a user-owned checkout. With a `repos.*` override, use that checkout's path for the same revision and environment checks.

Follow the pinned [environment setup](https://github.com/LocoManiLab/starVLA/blob/9740f0dc162669983829dbc02b6acce0f3a14444/README.md#-quick-start) in a separate model terminal. Its native steps use a **new Conda environment `starVLA` (case-sensitive), Python 3.10**, `python -m pip install -r requirements.txt`, a CUDA/PyTorch-compatible FlashAttention build, and `python -m pip install -e .` from `external/starVLA`. Preserve an existing environment instead of recreating it; override `conda_envs.starvla` for another name. Add `pyzmq`, `msgpack`, and `msgpack-numpy` there for the compatible server. This guide has not freshly resolved or built the full GPU environment.

Separately obtain a complete local **`StarVLA/Qwen3-VL-4B-Instruct-Action`** snapshot, with its tokenizer/processor assets, at `workspaces/starvla/assets/Qwen3-VL-4B-Instruct-Action`. The explicit `--base-model` below selects it. Preserve the local backbone path/cache needed by the saved run; offline defaults do not download a missing base. Run prerequisite checks from your main CLI environment:

```bash
lab --config workspaces/starvla/env.local.yaml doctor --model starvla --stage train
```

## SONIC data and command plans

This recipe uses the native **QwenGR00T** framework and `sonic_g1` mixture. Supply prepared LeRobot **v2.1** demonstrations with RGB ego video, state[43] joint/hand groups, task language, and absolute action[78] = motion tokens[64] + left/right hand targets[7+7]. The [fixed mixture registry](https://github.com/LocoManiLab/starVLA/blob/9740f0dc162669983829dbc02b6acce0f3a14444/starVLA/dataloader/gr00t_lerobot/mixtures.py#L45) uses `.` so the adapter's requested dataset root is used directly. Keep modality/episode/task/statistics metadata and SONIC checkpoint provenance. Other controller variants in the manifest are not end-to-end validation claims.

With real data at `workspaces/starvla/datasets/robot/prepared`, inspect:

```bash
lab --config workspaces/starvla/env.local.yaml validate workspaces/starvla/datasets/robot/prepared
lab --config workspaces/starvla/env.local.yaml convert --model starvla --dataset workspaces/starvla/datasets/robot/prepared --dry-run
lab --config workspaces/starvla/env.local.yaml train --model starvla --dataset workspaces/starvla/datasets/robot/prepared --base-model workspaces/starvla/assets/Qwen3-VL-4B-Instruct-Action --gpus 1 --gpu-ids 0 --task first-run --dry-run
lab --config workspaces/starvla/env.local.yaml deploy --model starvla --model-path workspaces/starvla/outputs/starvla/first-run/checkpoints/steps_1000_pytorch_model.pt --prompt 'open the drawer' --dry-run
```

`convert` checks modality/mixture requirements and does not rewrite this shared dataset. A successful check does not prove native loading or model training. Train delegates through Accelerate to native [train_starvla.py](https://github.com/LocoManiLab/starVLA/blob/9740f0dc162669983829dbc02b6acce0f3a14444/starVLA/training/train_starvla.py), passing the requested dataset, local base, state/action widths, and run ID. The example writes `workspaces/starvla/outputs/starvla/first-run`. Remove `--dry-run` only for an intended, separately provisioned model operation.

Replace the illustrative `steps_1000_pytorch_model.pt` with an **actual saved checkpoint file** and retain its companion run configuration/normalization assets. Deployment uses the compatible [run_starvla_server.py](https://github.com/LocoManiLab/starVLA/blob/9740f0dc162669983829dbc02b6acce0f3a14444/deployment/model_server/run_starvla_server.py) on port 5550; it starts a policy server, not cameras or robot control. The native websocket server is a separate protocol and is not the default shown here.

## Optional simulation plan

The CPU walkthrough does not provision simulation. Separately initialize the simulator/controller sources, their environment, rendering dependencies, and matching controller weights using [runtime setup](../../docs/data_collection.md) and [deployment guidance](../../docs/deployment.md). Model-server and simulator environments stay separate. For demonstrations recorded with SONIC `v1_1`, inspect this plan only after those paths/manifests exist:

```bash
lab --config workspaces/starvla/env.local.yaml eval --model starvla --sim --model-path workspaces/starvla/outputs/starvla/first-run/checkpoints/steps_1000_pytorch_model.pt --prompt 'open the drawer' --backend sonic --sonic-ckpt v1_1 --env EmptyEnv --episodes 1 --max-steps 60 --video-dir runs/eval/starvla --dry-run
```

Select the actual SONIC checkpoint variant used for your demonstrations. `EmptyEnv` checks wiring, not task success. A dry-run is a static command plan; it does not check readiness, load weights, or execute a policy. Simulation and hardware collection use `g1_backend`; complete model deployment has separate [runtime requirements](../../docs/deployment.md#robot-runtimes).

## Validation and common issues

The CPU data/configuration walkthrough and command/data-contract checks are separate from model execution. The fixed source entry points were inspected, but fresh installation of the full model environment, GPU training, learned-policy inference, simulation task success, and real hardware have **not** been validated for this guide. A successful `doctor` or `--dry-run` does not establish those results.

- Environment not found: the default name is `starVLA`, not lowercase `starvla`; model ID and environment name differ.
- Wrong dataset or missing modality: use the SONIC mixture and real prepared data with the correct state/action groups; the generic CPU sample is insufficient.
- Checkpoint-directory error: pass the saved `.pt` file, not just a run directory; keep its original configuration and statistics.
- FlashAttention/CUDA or offline backbone error: follow the fixed native platform requirements and provision the local assets. A printed command cannot verify those dependencies.
