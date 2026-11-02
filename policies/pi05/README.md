# pi0.5 / openpi

This entry connects LocoManiLab's shared humanoid dataset to the **pinned G1 integration fork** of openpi. It uses native JAX training and a GR00T-compatible policy server. Model execution has not been revalidated end to end for this release; a successful CPU walkthrough or dry run does not establish GPU training, inference, or robot readiness.

The original project is [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi); use the integration fork and revision below for this guide.

| Item | Source of truth |
|---|---|
| Model ID | `pi05` |
| Source repository | [LocoManiLab/openpi](https://github.com/LocoManiLab/openpi) |
| Recorded gitlink | `a8f841ef8aa1a7fc04156cf49f68977fd3750d51` |
| Main adapter / manifest | [`pi05.py`](../../src/locomanilab/models/pi05.py) / [`pi05.yaml`](../../src/locomanilab/configs/pi05.yaml) |
| Native recipe registry | [`src/openpi/training/config.py` at the pin](https://github.com/LocoManiLab/openpi/blob/a8f841ef8aa1a7fc04156cf49f68977fd3750d51/src/openpi/training/config.py) |
| Native serving contract | [`deployment/run_openpi_server.py` at the pin](https://github.com/LocoManiLab/openpi/blob/a8f841ef8aa1a7fc04156cf49f68977fd3750d51/deployment/run_openpi_server.py) |

Use the superproject's current gitlink if maintainers update it; do not replace this fork with a floating upstream branch. The fork contains G1 data factories, checkpoint-loading changes, and controller-specific serving contracts that a generic upstream checkout may lack.

## 1. Check the CPU data path first

Run from the **main checkout root**, with LocoManiLab `.[data]` already installed and its `lab` command on `PATH`. This block needs no openpi checkout, weights, GPU, simulator, or robot. It creates a new `workspaces/pi05` and refuses an existing directory or symlink so it cannot replace your local configuration.

<!-- cpu-smoke:start -->
```bash
(
  set -eu
  if [ -e workspaces/pi05 ] || [ -L workspaces/pi05 ]; then
    echo "workspaces/pi05 already exists; keep it and use a fresh workspace for this walkthrough." >&2
    exit 1
  fi
  mkdir -p workspaces/pi05
  cp policies/pi05/workspace.example.yaml workspaces/pi05/env.local.yaml
  cd workspaces/pi05
  lab doctor --stage data
  lab sample-data --out datasets/cpu-example
  lab validate datasets/cpu-example
  lab prepare datasets/cpu-example --merge-only --out datasets/cpu-prepared
  lab validate datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

The sample contains two small synthetic episodes. It exercises installation, parquet/video validation, and preparation; **it does not satisfy pi05's humanoid state/action contract**. Do not pass it to the real-data commands below. Preparation currently accepts positive integer FPS only; fractional-rate originals can be validated but are not resampled by `prepare`.

[`workspace.example.yaml`](workspace.example.yaml) is intended for `workspaces/pi05/env.local.yaml`. Its relative `repos.*` and `paths.*` values resolve from that workspace. CLI `--dataset` and `--model-path` arguments resolve from the shell's current directory. Preserve and merge an existing workspace configuration instead of copying over it.

## 2. Explicitly provision the model environment

These steps download source and Python/CUDA packages. They are separate from the CPU walkthrough and must be chosen explicitly. From the main checkout root:

```bash
python3 scripts/setup/profiles.py pi05
python3 scripts/setup/profiles.py pi05 --apply
git -C external/openpi rev-parse HEAD
```

Check that the printed commit matches the recorded gitlink. Existing checkouts are not reset by the profile helper. The [pinned native installation instructions](https://github.com/LocoManiLab/openpi/blob/a8f841ef8aa1a7fc04156cf49f68977fd3750d51/README.md#installation) and lockfile require a **separate Python 3.11+ environment**; this pin declares `jax[cuda12]==0.5.3`. Model training/inference needs a compatible NVIDIA driver/GPU and sufficient memory. The main CLI's Python 3.10+ environment does not replace it.

With uv already installed, provision the selected checkout:

```bash
(
  set -eu
  cd external/openpi
  GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
  GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python --no-deps -e .
  uv pip install --python .venv/bin/python --no-deps msgpack-numpy==0.4.8
  uv run --no-sync python -c 'import jax, msgpack_numpy, zmq; print(jax.devices())'
)
```

The compatibility server imports `msgpack_numpy`, which is absent from this pin's uv lockfile; the explicit [msgpack-numpy 0.4.8](https://pypi.org/project/msgpack-numpy/0.4.8/) addition supplies it without replacing the locked NumPy/msgpack dependencies. The lock already includes PyZMQ. Recheck this addition after resynchronizing the environment. No PyTorch conversion or `transformers` source patch is required by the JAX workflow described here; the alternate PyTorch training path is not established by this adapter.

From `workspaces/pi05`, inspect the selected prerequisites:

```bash
lab doctor --model pi05 --stage train
```

`doctor` checks configured checkout/environment paths and the uv executable. It does not load weights, verify GPU memory, prove recipe compatibility, or run inference.

## 3. Provide the required local assets

| Asset | Needed for | Location in the example workspace |
|---|---|---|
| PaliGemma tokenizer | Normalization, training, serving | `cache/openpi/big_vision/paligemma_tokenizer.model` |
| pi05 base Orbax parameters | Starting a training run | `cache/openpi/openpi-assets/checkpoints/pi05_base/params` |
| Valid humanoid demonstrations | Normalization/training | `datasets/<native-repo-id>/` |
| Dataset-bound normalization | Training | Under `artifacts/pi05/`, produced by `lab convert` |
| Trained checkpoint with `params/` and `assets/` | Local serving/evaluation | A numeric step directory under `checkpoints/pi05/<recipe>/<run>/` |

Weights and demonstrations are not bundled. Obtain them under their own terms; see [third-party boundaries](../../docs/THIRD_PARTY.md). If assets already exist elsewhere, set `paths.openpi_data_home` and `paths.pi05_base` to those local locations. A base `params` directory alone is not a trained humanoid policy checkpoint.

The following commands **explicitly download** the native assets into the workspace cache. Run them only after provisioning openpi, from `workspaces/pi05`. Native openpi uses gsutil when available for `openpi-assets`, with a gcsfs fallback; network/storage access must be available. For a fully local run, stage these assets before starting conversion or inference.

```bash
# Tokenizer: needed even when serving an existing trained checkpoint.
OPENPI_DATA_HOME="$PWD/cache/openpi" \
  uv run --project ../../external/openpi --no-sync python - <<'PY'
from openpi.shared import download
print(download.maybe_download(
    "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
))
PY
```

```bash
# Large base weights: skip when only serving an existing trained checkpoint.
OPENPI_DATA_HOME="$PWD/cache/openpi" \
  uv run --project ../../external/openpi --no-sync python - <<'PY'
from openpi.shared import download
print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params"))
PY
```

The workspace config makes the adapter pass `OPENPI_DATA_HOME` to native jobs. Missing cached tokenizer files can still trigger native downloads; `lab` does not make an uncached model environment offline.

## 4. Match real data to a registered recipe

This pin registers the following defaults. These are code/contract mappings, not new end-to-end validation claims:

| Controller | Default `--config-name` | Required local dataset suffix (`data.repo_id`) | Native state / action |
|---|---|---|---|
| SONIC | `pi05_bench_sonic_all31` | `all31_merged` | 43 / 78: motion token 64 + hands 7 + 7 |
| HeFT | `pi05_bench_heft_openfridge` | `bench_heft_KitchenOpenFridge` | 29 / 50: `action.ref_qpos` 36 + `action.hands` 14 |
| AMO — blocked at this pin | `pi05_bench_amo_closefridge` | `bench_amo_KitchenCloseFridge` | State 32; factory/transform disagree on the action streams |

The fork's model recipes pad actions to 80 internally. For SONIC, raw features include `observation.images.ego_view`, `observation.state`, `action.motion_token`, `teleop.left_hand_joints`, and `teleop.right_hand_joints`. The source must also carry the state/action/video groups required by the [manifest](../../src/locomanilab/configs/pi05.yaml), plus task text, episode metadata, aligned videos, and statistics. A matching vector size alone does not establish controller or action meaning.

The AMO entry is listed for visibility: its pinned factory passes a bare `action` array, but its updated transform expects named `action` (22) and `hands` (14) streams. A CPU transform probe reproduces an indexing failure before model execution. Do not use this pin's AMO normalization/training path until the fork and adapter are corrected together. The runnable model examples here use SONIC.

The adapter detects the controller from `meta/modality.json`, inspects the native recipe without importing JAX, checks its exact data-factory class, and derives `HF_LEROBOT_HOME` so the recipe reads the requested dataset. For another dataset ID, register a **literal native `TrainConfig`** with an explicit `data.repo_id` and the matching G1 factory in your openpi checkout, then use its name consistently. Dynamically constructed recipes cannot be established by this static check. Do not substitute a generic ALOHA/LIBERO/DROID recipe or bypass the data binding with `--set data...`.

### Convert and train

The following SONIC example assumes you have supplied and validated real compatible demonstrations at `workspaces/pi05/datasets/all31_merged`. That dataset name is a native registry binding; the repository does not supply the original dataset or benchmark results. Run from `workspaces/pi05`:

```bash
RECIPE=pi05_bench_sonic_all31
DATASET=datasets/all31_merged
lab validate "$DATASET"
lab convert --model pi05 --dataset "$DATASET" --config-name "$RECIPE" --dry-run
lab convert --model pi05 --dataset "$DATASET" --config-name "$RECIPE"
```

For pi05, `convert` computes native normalization rather than writing another copy of the parquet dataset. Its artifacts are bound to the dataset location and metadata and include `norm_stats.json` plus `locomanilab_dataset.json`. Recompute after changing that dataset or recipe; do not reuse another controller's statistics.

Inspect a short, genuine GPU training run before launching it:

```bash
lab train --model pi05 --dataset "$DATASET" --config-name "$RECIPE" \
  --task first-run --gpus 1 --set num-train-steps=10 --set save-interval=5 \
  --set batch-size=2 --set num-workers=0 --set no-wandb-enabled --dry-run
lab train --model pi05 --dataset "$DATASET" --config-name "$RECIPE" \
  --task first-run --gpus 1 --set num-train-steps=10 --set save-interval=5 \
  --set batch-size=2 --set num-workers=0 --set no-wandb-enabled
```

This still initializes and trains the full model; it is not a CPU smoke test or a memory guarantee. `paths.pi05_base` supplies the local base parameters, and `paths.pi05_outputs` sets the output root. Use a fresh `--task` for another run. For a full experiment, choose steps/batch size for your resources and the pinned recipe. `--gpus` controls native single-node FSDP; it is not a multi-node launcher. The example disables native W&B logging explicitly.

## 5. Serve a local trained checkpoint

After the ten-step example completes, this pin saves its final zero-based step as `9`. From `workspaces/pi05`:

```bash
RECIPE=pi05_bench_sonic_all31
CHECKPOINT="checkpoints/pi05/$RECIPE/first-run/9"
test -d "$CHECKPOINT/params"
test -f "$CHECKPOINT/assets/all31_merged/norm_stats.json"
# Set this to an instruction used in your actual demonstrations.
PROMPT='pick up the object'
lab deploy --model pi05 --model-path "$CHECKPOINT" --config-name "$RECIPE" \
  --prompt "$PROMPT" --port 5550 --dry-run
lab deploy --model pi05 --model-path "$CHECKPOINT" --config-name "$RECIPE" \
  --prompt "$PROMPT" --port 5550
```

For another training run, use its actual numeric step directory. Pass the **step directory**, containing both `params/` and `assets/<data.repo_id>/norm_stats.json`, rather than its `params/` child or an experiment parent. Retain the tokenizer cache and the same registered recipe on the serving machine. The recipe chooses the controller's deployment contract; the checkpoint path does not choose it automatically.

`deploy` starts the fork's compatible policy server in the foreground. It does not start a robot controller, cameras, or hardware. It needs neither a simulator nor the original training dataset when the required checkpoint assets are present. Stop this server before running `infer-check` or `eval` on the same port: those commands launch their own server.

## 6. Optional offline and simulation checks

`infer-check` needs the model environment, trained checkpoint, real dataset, and a separately provisioned client environment (selected with `--sim-env`, default `tele_benchmark`). It performs open-loop predicted-versus-recorded-action checks; it is not task-success evaluation:

```bash
lab infer-check --model pi05 --dataset "$DATASET" --model-path "$CHECKPOINT" \
  --config-name "$RECIPE" --prompt "$PROMPT" --episode 0 --frames 20 \
  --out runs/infer-check/pi05.png --dry-run
```

Remove `--dry-run` only after provisioning that client environment. Closed-loop simulation additionally requires the `sim` source profile, the humanoid benchmark/robot assets, a working simulator environment, and controller checkpoints matching the demonstrations. From the main checkout root, `python3 scripts/setup/profiles.py sim --apply` initializes source only. Provision the [simulator separately](../../docs/sim_validation.md).

With those resources ready, run from `workspaces/pi05`:

```bash
lab doctor --model pi05 --stage eval --sim
lab eval --model pi05 --sim --model-path "$CHECKPOINT" --config-name "$RECIPE" \
  --backend sonic --sonic-ckpt low_latency --env EmptyEnv --episodes 1 \
  --prompt "$PROMPT" --max-steps 20 --render video \
  --video-dir runs/eval/pi05 --results-json runs/eval/pi05.json --dry-run
```

This is an `EmptyEnv` wiring example, not a benchmark score. Choose the controller checkpoint variant from your collection manifest; `low_latency` is not interchangeable with SONIC `v1_1`. For scored runs, replace the environment/prompt and reset settings with the recorded task protocol. After inspecting the plan, remove `--dry-run` to execute. `doctor --sim` executes simulator import/readiness probes, unlike the standalone CPU block. See [controller contracts](../../docs/controller.md).

## Common problems

- **CPU sample rejected by pi05:** expected; it has a tiny generic action layout. Supply real controller-compatible data.
- **Dataset suffix or recipe mismatch:** `--dataset` must end in the selected literal `data.repo_id`. Register a matching native recipe and keep the controller's G1 data factory.
- **Missing dataset-bound normalization:** run `lab convert` for the same dataset/recipe. Changing paths or metadata can invalidate the recorded binding.
- **An older conversion is rejected after upgrading:** dataset-identity hashing now includes task text and source statistics. Run `lab convert` again for the same source and recipe to generate the current binding. Preserve the original data, existing checkpoints, and old statistics; old statistics alone do not satisfy the new binding.
- **Tokenizer download/network error:** stage the tokenizer into the configured `openpi_data_home`; a local checkpoint does not include that shared tokenizer automatically.
- **Missing `msgpack_numpy`:** install the explicit compatibility dependency in the openpi venv, not the main CLI venv; a later native `uv sync` can remove an unrecorded addition.
- **Missing checkpoint norm stats:** use a complete trained step directory with its `assets/<data.repo_id>` subtree. Base-model parameters or copied `params/` alone are insufficient.
- **No GPU or out of memory:** inspect `jax.devices()` in the model environment and use the pinned native resource guidance. CPU checks do not certify this prerequisite.
- **Unknown recipe / wrong action space:** retain the integration fork and use the same recipe for conversion, training, serving, and evaluation. Keep the dataset identity marker with its statistics; existing markers are read compatibly.
- **AMO normalization fails:** the pinned factory and two-stream transform are inconsistent; this is an integration limitation, not a dataset naming problem. Use the documented SONIC path or a separately reviewed fix.
