# GR00T

Use `--model gr00t`. This directory is a guide and workspace example; execution stays in the shared `lab` CLI, [gr00t adapter](../../src/locomanilab/models/gr00t.py), and [manifest](../../src/locomanilab/configs/gr00t.yaml). It adds no training wrapper or source checkout.

The source profile `gr00t` selects [LocoManiLab/Isaac-GR00T](https://github.com/LocoManiLab/Isaac-GR00T) at **`7e73a6fb8813de6f31850e94d4ae0f94a73e6e40`**. The main repository's gitlink is authoritative; a floating upstream installation is not the integration described here. Source access and the source/model licenses must be checked separately before use or redistribution.

## CPU data walkthrough

Run from the **LocoManiLab checkout root**, with the main package's `[data]` extra installed and that environment activated; see the [CPU quickstart](../../README.md#start-on-cpu). No model source, checkpoint, GPU, simulator, or hardware is needed. This block refuses an existing workspace or symlink.

<!-- cpu-smoke:start -->
```bash
(
set -eu
test ! -e workspaces/gr00t && test ! -L workspaces/gr00t || {
  echo "workspaces/gr00t already exists; choose a fresh workspace before running this example." >&2
  exit 1
}
mkdir -p workspaces/gr00t
cp policies/gr00t/workspace.example.yaml workspaces/gr00t/env.local.yaml
lab --config workspaces/gr00t/env.local.yaml doctor --stage data
lab --config workspaces/gr00t/env.local.yaml sample-data --out datasets/cpu-example
lab --config workspaces/gr00t/env.local.yaml validate workspaces/gr00t/datasets/cpu-example
lab --config workspaces/gr00t/env.local.yaml prepare workspaces/gr00t/datasets/cpu-example --merge-only --out datasets/cpu-prepared
lab --config workspaces/gr00t/env.local.yaml validate workspaces/gr00t/datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

The synthetic state[4]/action[2] sample checks the data pipeline. It is **not training data for this model**. The [workspace example](workspace.example.yaml) belongs at `workspaces/gr00t/env.local.yaml`: schema v2 resolves `../../external/Isaac-GR00T` to this checkout's source and keeps configured data, outputs, and logs in the workspace. Override `repos.isaac_gr00t` for an existing compatible source elsewhere. CLI dataset/base/checkpoint inputs are relative to the invoking directory; output paths are relative to the workspace. All examples below also run from the project root.

## Source and separate model environment

The following is an **optional source download**, outside the CPU walkthrough. The first command only prints a plan; `--apply` explicitly initializes missing sources, without weights or a model environment. Existing checkouts are preserved.

```bash
(
set -eu
python3 scripts/setup/profiles.py gr00t
python3 scripts/setup/profiles.py gr00t --apply
source_revision=$(git -C external/Isaac-GR00T rev-parse HEAD)
test "$source_revision" = 7e73a6fb8813de6f31850e94d4ae0f94a73e6e40 || {
  echo "Source revision differs from this recipe; review it with Git before installing. Nothing was reset." >&2
  exit 1
}
)
```

If the guard fails, inspect `git -C external/Isaac-GR00T status` and `git -C external/Isaac-GR00T diff`, preserve your changes, and review the fixed revision before selecting it with standard Git. Do not reset or clean a user-owned checkout. With a `repos.*` override, use that checkout's path for the same revision and environment checks.

For x86_64 dGPU, follow the pinned [platform and installation guide](https://github.com/LocoManiLab/Isaac-GR00T/blob/7e73a6fb8813de6f31850e94d4ae0f94a73e6e40/README.md#installation): CUDA/PyTorch compatibility and system FFmpeg are prerequisites. Its default installation is `uv sync --python 3.10`, run from `external/Isaac-GR00T`; it downloads/builds GPU dependencies including FlashAttention and TensorRT. ARM/Jetson paths differ; use that guide rather than applying the dGPU recipe to them. This guide has not validated a fresh GPU installation.

For an explicitly requested **new** dGPU environment, after reviewing the native requirements and the revision guard above:

```bash
(
set -eu
test ! -e external/Isaac-GR00T/.venv && test ! -L external/Isaac-GR00T/.venv || {
  echo 'Existing GR00T environment preserved; inspect it separately.' >&2
  exit 1
}
cd external/Isaac-GR00T
uv sync --frozen --python 3.10
)
```

`flash-attn==2.7.4.post1` has no prebuilt wheel for the pinned `torch==2.7.1`, so `uv sync` compiles it: this needs a CUDA toolkit with `nvcc` (12.8 or newer for RTX 50-series GPUs) and `CUDA_HOME`, and the build takes a long time. Without a toolkit, `uv sync --frozen --python 3.10 --no-install-package flash-attn` installs everything else; GR00T then logs `flash_attn is not installed. Falling back to sdpa attention.` and serves with PyTorch SDPA attention, which is slower but loads the same weights. `--frozen` installs the locked versions without rewriting `uv.lock`; newer uv releases otherwise re-resolve and modify the checkout.

The native requirements include the ZMQ/msgpack transport. Model commands use `uv run --no-sync`, so they do not repair a missing environment. Obtain the base model **`nvidia/GR00T-N1.7-3B`** separately, respecting its model terms, and place its complete local snapshot at `workspaces/gr00t/assets/GR00T-N1.7-3B`, matching `paths.gr00t_base`. See the pinned [fine-tuning guide](https://github.com/LocoManiLab/Isaac-GR00T/blob/7e73a6fb8813de6f31850e94d4ae0f94a73e6e40/getting_started/finetune_new_embodiment.md). Check prerequisites from the main CLI environment:

```bash
lab --config workspaces/gr00t/env.local.yaml doctor --model gr00t --stage train
```

## SONIC data and command plans

This example is for prepared LeRobot **v2.1** G1 SONIC data with RGB `ego_view`, state[43] joint/hand groups, task-language metadata, and absolute motion tokens[64] plus left/right hand targets[7+7]. Its `meta/modality.json` must expose the native [UNITREE_G1_SONIC modality groups](https://github.com/LocoManiLab/Isaac-GR00T/blob/7e73a6fb8813de6f31850e94d4ae0f94a73e6e40/gr00t/configs/data/embodiment_configs.py#L67). Preserve the recording controller and SONIC checkpoint variant. The manifest lists additional embodiments; their presence does not establish a tested recipe for another controller.

With real demonstrations at `workspaces/gr00t/datasets/robot/prepared`, inspect:

```bash
lab --config workspaces/gr00t/env.local.yaml validate workspaces/gr00t/datasets/robot/prepared
lab --config workspaces/gr00t/env.local.yaml convert --model gr00t --dataset workspaces/gr00t/datasets/robot/prepared --dry-run
lab --config workspaces/gr00t/env.local.yaml train --model gr00t --dataset workspaces/gr00t/datasets/robot/prepared --gpus 1 --gpu-ids 0 --task first-run --per-gpu-batch 1 --dry-run
lab --config workspaces/gr00t/env.local.yaml deploy --model gr00t --model-path workspaces/gr00t/outputs/gr00t/first-run/checkpoint-20000 --server-only --dry-run
```

`convert` does not rewrite the shared dataset format; its success is not a full native dataloader check. The train plan invokes native [launch_finetune.py](https://github.com/LocoManiLab/Isaac-GR00T/blob/7e73a6fb8813de6f31850e94d4ae0f94a73e6e40/gr00t/experiment/launch_finetune.py), uses the configured local base, and writes `workspaces/gr00t/outputs/gr00t/first-run`. The explicit smaller batch is a command example, not a GPU memory guarantee. Remove `--dry-run` only when you deliberately intend that model operation; use a new task/output for a new run.

For serving, replace `checkpoint-20000` with an actual compatible saved directory. **Keep `--server-only`** for a standalone policy server: it uses the native [run_gr00t_server.py](https://github.com/LocoManiLab/Isaac-GR00T/blob/7e73a6fb8813de6f31850e94d4ae0f94a73e6e40/gr00t/eval/run_gr00t_server.py) in the Isaac-GR00T environment. Omitting it selects the native WholeBodyControl robot launcher, with camera/SDK/control prerequisites. That complete native WBC pipeline is this project's **recommended best practice for existing GR00T robot deployment**. See [deployment boundaries](../../docs/deployment.md) before considering a robot workflow.

## Released checkpoint

A GR00T checkpoint fine-tuned on the benchmark's train50 split (step 10000) is released on the Hugging Face Hub; see [data, assets, and checkpoints](../../docs/hf_resources.md). From the checkout root, with the main package installed with the `hf` extra:

```bash
python scripts/hf/download.py checkpoint gr00t --out checkpoints/hf
lab --config workspaces/gr00t/env.local.yaml deploy --model gr00t --model-path checkpoints/hf/checkpoint/gr00t --server-only
```

The server loads the gated backbone `nvidia/Cosmos-Reason2-2B` when it starts and contacts the Hub even when the backbone is already cached, so `HF_HUB_OFFLINE=1` does not work. Accept the backbone's terms on huggingface.co and log in (`hf auth login`) on the serving machine. When the server reports `Server is ready and listening`, check its responses with:

```bash
external/Isaac-GR00T/.venv/bin/python scripts/diagnostics/check_gr00t_server.py --port 5550
external/Isaac-GR00T/.venv/bin/python scripts/hf/gr00t_openloop.py --port 5550 \
  --dataset datasets/hf/dataset/main/atomic/seen/IThorKitchenOpenFridge_scene13
```

The second check needs that task's data (`python scripts/hf/download.py dataset --task IThorKitchenOpenFridge_scene13 --out datasets/hf`) and compares served actions with the recorded ones. The released checkpoint contains no optimizer state and cannot resume training.

## Optional simulation plan

The CPU walkthrough does not provision simulation. Separately initialize the simulator/controller sources, their environment, rendering dependencies, and matching controller weights using [runtime setup](../../docs/data_collection.md) and [deployment guidance](../../docs/deployment.md). Model-server and simulator environments stay separate. For demonstrations recorded with SONIC `v1_1`, inspect this plan only after those paths/manifests exist:

```bash
lab --config workspaces/gr00t/env.local.yaml eval --model gr00t --sim --model-path workspaces/gr00t/outputs/gr00t/first-run/checkpoint-20000 --prompt 'open the drawer' --backend sonic --sonic-ckpt v1_1 --env EmptyEnv --episodes 1 --max-steps 60 --video-dir runs/eval/gr00t --dry-run
```

To evaluate a benchmark task with its recorded settings (scene, prompt, episodes, step budget, SONIC `v1_1`, reset jitter), name the task instead; the task's scene assets must be provisioned:

```bash
lab --config workspaces/gr00t/env.local.yaml eval --model gr00t --sim --model-path workspaces/gr00t/outputs/gr00t/first-run/checkpoint-20000 --task IThorKitchenOpenFridge_scene13 --dry-run
```

`lab tasks` lists the tasks; `--task-set atomic/seen` (or another selection) runs several and writes a `summary.json`, see the [project README](../../README.md#evaluate-on-benchmark-tasks).

Select the actual SONIC checkpoint variant used for your demonstrations. `EmptyEnv` checks wiring, not task success. A dry-run is a static command plan; it does not check readiness, load weights, or execute a policy. Simulation and hardware collection use `g1_backend`; complete model deployment has separate [runtime requirements](../../docs/deployment.md#robot-runtimes).

## Validation and common issues

The CPU data/configuration walkthrough and command/data-contract checks are separate from model execution. The fixed source entry points were inspected. A fresh `uv sync --frozen --no-install-package flash-attn` environment served the released checkpoint on one x86_64 RTX 50-series GPU (SDPA attention), and its served actions were compared open-loop with recorded benchmark actions. With the simulator environment from [simulation setup](../../docs/simulation.md), `lab eval --task IThorKitchenOpenFridge_scene13 --episodes 2` succeeded in both closed-loop episodes. GPU training, a FlashAttention build, benchmark-wide success rates, and real hardware have **not** been validated for this guide. A successful `doctor` or `--dry-run` does not establish those results.

- Missing base or backend environment: set `paths.gr00t_base` to a complete local N1.7 snapshot and provision the selected Isaac-GR00T `.venv`; installing the CLI alone is insufficient.
- Unknown modality/tag or missing hands: compare the dataset metadata with the fixed native SONIC configuration. Another controller's equal-sized vector is not a replacement.
- `doctor --stage deploy` checks the full native WBC deployment boundary; the server-only command above instead uses Isaac-GR00T's `.venv`.
- Missing FFmpeg/torchcodec/CUDA dependency: follow the pinned platform guide in the backend environment; a dry-run cannot establish video decoding or CUDA readiness.
