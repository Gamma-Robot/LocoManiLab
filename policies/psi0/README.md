# Psi0

Original project: [physical-superintelligence-lab/Psi0](https://github.com/physical-superintelligence-lab/Psi0). This guide uses the [LocoManiLab integration fork](https://github.com/LocoManiLab/Psi0) at **`1b757177a92ab4c540ee5e35cc0baae3a219739c`**. The fork supplies converters, controller trainers, and a compatible server; a generic upstream checkout is not a substitute. GPU training, serving, and hardware execution have not been revalidated end to end for this release.

Implementation: [adapter](../../src/locomanilab/models/psi0.py), [manifest](../../src/locomanilab/configs/psi0.yaml), pinned [SONIC converter](https://github.com/LocoManiLab/Psi0/blob/1b757177a92ab4c540ee5e35cc0baae3a219739c/scripts/data/locomanilab_sonic_to_psi_lerobot.py), [training entry](https://github.com/LocoManiLab/Psi0/blob/1b757177a92ab4c540ee5e35cc0baae3a219739c/scripts/train.py), and [policy server](https://github.com/LocoManiLab/Psi0/blob/1b757177a92ab4c540ee5e35cc0baae3a219739c/src/psi/deploy/run_psi0_server.py).

## CPU acceptance

From the main checkout root, with the main `.[data]` package installed and `lab` on `PATH`. This uses no model checkout, weights, GPU, or simulator and refuses an existing workspace.

<!-- cpu-smoke:start -->
```bash
(
  set -eu
  if [ -e workspaces/psi0 ] || [ -L workspaces/psi0 ]; then
    echo "workspaces/psi0 already exists; use a fresh workspace." >&2
    exit 1
  fi
  mkdir -p workspaces/psi0
  cp policies/psi0/workspace.example.yaml workspaces/psi0/env.local.yaml
  cd workspaces/psi0
  lab doctor --stage data
  lab sample-data --out datasets/cpu-example
  lab validate datasets/cpu-example
  lab prepare datasets/cpu-example --merge-only --out datasets/cpu-prepared
  lab validate datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

The synthetic sample checks the shared data tools, not Psi0 training compatibility. Do not use it in the model examples. The [example configuration](workspace.example.yaml) belongs in `workspaces/psi0`; schema-v2 configured paths are relative to that workspace, while CLI dataset/checkpoint arguments are relative to the calling directory. Merge settings into an existing workspace rather than replacing it.

## Explicit model setup and assets

From the main checkout root, inspect and initialize only this source profile:

```bash
python3 scripts/setup/profiles.py psi0
python3 scripts/setup/profiles.py psi0 --apply
```

Follow the [installation section at this pin](https://github.com/LocoManiLab/Psi0/blob/1b757177a92ab4c540ee5e35cc0baae3a219739c/README.md#installation). Its model environment requires **Python 3.10**, the `psi`, `serve`, and `viz` groups, and the native CUDA/FlashAttention stack. The main adapter invokes `uv run --no-sync` and `doctor` checks `external/Psi0/.venv`; use that location instead of blindly retaining the native README's `.venv-psi` name:

```bash
(
  set -eu
  cd external/Psi0
  uv venv .venv --python 3.10
  UV_PROJECT_ENVIRONMENT=.venv GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen \
    --group psi --group serve --group viz --index-strategy unsafe-best-match
  uv pip install --python .venv/bin/python flash_attn==2.7.4.post1 --no-build-isolation
)
```

This is an explicit package/native-extension installation, not part of the CPU block. The compatibility server also requires `pyzmq` and `msgpack-numpy` in that model environment. Provision the versions compatible with the pinned environment before serving; the main CLI does not supply them.

Acquire the two model directories listed in the pinned [checkpoint table](https://github.com/LocoManiLab/Psi0/blob/1b757177a92ab4c540ee5e35cc0baae3a219739c/README.md#checkpoints) from [USC-PSI-Lab/psi-model](https://huggingface.co/USC-PSI-Lab/psi-model): the VLM `psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k` and action expert `psi0/postpre.1by1.pad36.2601131206.ckpt.he30k`. Put their complete contents at the corresponding `paths.psi0_vlm_base` / `psi0_action_expert` locations or edit those paths. Preserve native checkpoint filenames/layouts. The native serving code additionally resolves the `Qwen/Qwen3-VL-2B-Instruct` config/processor from its HF cache; stage those files under `cache/psi/hf_hub` (`HF_HUB_CACHE`) for offline serving. No weights or robot datasets are downloaded by the profile helper.

From `workspaces/psi0`, `lab doctor --model psi0 --stage train` checks the selected source/uv environment; it does not validate weights or GPU memory. Read [third-party terms](../../docs/THIRD_PARTY.md) before acquiring assets.

## Real data, conversion, and training

The manifest declares SONIC state/action **43/78**, HeFT **29/50**, and AMO **32/36**, with separate converters/trainers. The server dispatches by trained dimensions. These are retained code contracts, not validation of every controller path. SONIC conversion currently requires metadata groups `video.ego_view`, `state.joint_pos`, and `action.motion_token/left_hand_joints/right_hand_joints`; retain the actual feature mappings rather than relabeling another layout. The converter reorders the 43-D state to the native body-plus-hands order and produces `stats_psi0.json`.

Supply real compatible demonstrations at `datasets/sonic_demo`. From `workspaces/psi0`:

```bash
DATASET=datasets/sonic_demo
lab validate "$DATASET"
lab convert --model psi0 --dataset "$DATASET" --dry-run
lab train --model psi0 --dataset "$DATASET" --task first-run --gpus 1 \
  --set train.max_training_steps=10 --set train.train_batch_size=1 --dry-run
```

These are plans. Execute conversion by removing its `--dry-run`, then inspect `datasets/psi0/sonic_demo/meta/stats_psi0.json` before executing training. Keep passing the **original dataset** to `train`; the adapter selects its converted copy and reads its dimensions. The short training override still loads/trains the real model. Default logging is offline; source/output/cache paths and compatible local assets remain prerequisites. Do not reuse a converted basename for different data.

## Local inference and optional evaluation

Choose an actual run directory under `checkpoints/psi0/sonic/` containing `run_config.json` and `checkpoints/ckpt_<N>`, or pass that checkpoint child directly. Run names include a native timestamp; the path below is a placeholder to replace, not an automatically selected checkpoint:

```bash
RUN=checkpoints/psi0/sonic/first-run.REPLACE_WITH_ACTUAL_TIMESTAMP
PROMPT='the instruction recorded in your demonstrations'
lab deploy --model psi0 --model-path "$RUN" --prompt "$PROMPT" --port 5550 --dry-run
lab eval --model psi0 --sim --model-path "$RUN" --backend sonic \
  --sonic-ckpt low_latency --env EmptyEnv --episodes 1 --max-steps 20 \
  --prompt "$PROMPT" --video-dir runs/eval/psi0 --dry-run
```

Remove `--dry-run` only after the corresponding prerequisites are ready. `deploy` runs the policy server, not a robot controller. Keep the run configuration, selected checkpoint, normalization metadata, and Qwen processor cache together. `eval` starts its own server: stop a separately launched server on that port first. Simulation also needs the `sim` profile, simulator environment, robot/controller assets, and a controller checkpoint matching the demonstrations. `EmptyEnv` is a wiring example; it is not task-success evaluation. See [simulation setup](../../docs/sim_validation.md).

Common failures: missing `stats_psi0.json` means real conversion/statistics did not finish; missing metadata groups means this source layout is not accepted by the selected converter; offline processor errors mean the Qwen cache is incomplete; a run with a different action dimension needs its matching controller. CPU success and command plans establish none of these model/runtime prerequisites.
