# Xiaomi-Robotics-1 / XR-1

Original project: [XiaomiRobotics/Xiaomi-Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1). This entry uses the [LocoManiLab integration fork](https://github.com/LocoManiLab/Xiaomi-Robotics-1) at **`8598deffcc560c4ca8aad616c82f925b248de8c1`**, including its SONIC data/layout, widening, and compatible serving changes. Model/GPU/hardware execution has not been revalidated end to end for this release.

Sources: [adapter](../../src/locomanilab/models/xr1.py), [manifest](../../src/locomanilab/configs/xr1.yaml), pinned [JSON converter](https://github.com/LocoManiLab/Xiaomi-Robotics-1/blob/8598deffcc560c4ca8aad616c82f925b248de8c1/xr1/scripts/data/locomanilab_sonic_to_xr1_json.py), [normalization/config generator](https://github.com/LocoManiLab/Xiaomi-Robotics-1/blob/8598deffcc560c4ca8aad616c82f925b248de8c1/xr1/scripts/data/compute_xr1_stats.py), and [policy server](https://github.com/LocoManiLab/Xiaomi-Robotics-1/blob/8598deffcc560c4ca8aad616c82f925b248de8c1/deploy/run_xr1_server.py).

## CPU acceptance

Run from the main checkout root, with main `.[data]` installed and `lab` on `PATH`. This uses no model checkout, weights, GPU, or simulator and refuses an existing workspace.

<!-- cpu-smoke:start -->
```bash
(
  set -eu
  if [ -e workspaces/xr1 ] || [ -L workspaces/xr1 ]; then
    echo "workspaces/xr1 already exists; use a fresh workspace." >&2
    exit 1
  fi
  mkdir -p workspaces/xr1
  cp policies/xr1/workspace.example.yaml workspaces/xr1/env.local.yaml
  cd workspaces/xr1
  lab doctor --stage data
  lab sample-data --out datasets/cpu-example
  lab validate datasets/cpu-example
  lab prepare datasets/cpu-example --merge-only --out datasets/cpu-prepared
  lab validate datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

The sample is not XR-1-compatible robot training data. [Configured paths](workspace.example.yaml) resolve from `workspaces/xr1`; CLI dataset/checkpoint inputs resolve from the caller's directory. Merge an existing workspace configuration rather than replacing it.

## Explicit environment and local assets

From the main checkout root:

```bash
python3 scripts/setup/profiles.py xr1
python3 scripts/setup/profiles.py xr1 --apply
```

Follow [the pinned XR-1 installation guide](https://github.com/LocoManiLab/Xiaomi-Robotics-1/blob/8598deffcc560c4ca8aad616c82f925b248de8c1/xr1/README.md#installation) in a separate conda environment named `xr1`. It requires Python 3.9+, CUDA/GPU support, and Transformers **4.57.1**. The editable installation is run from `external/Xiaomi-Robotics-1/xr1` (`pip install -e .`), followed by the native FlashAttention build. Compatibility serving also requires `pyzmq`/`msgpack-numpy` in that environment. The profile helper does not install packages or binary assets.

Acquire the released [Xiaomi-Robotics-1-5B](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B) `model_states.pt` and a complete `Qwen/Qwen3-VL-4B-Instruct` cache under their own terms. `paths.xr1_hf_home` sets `HF_HOME` for training and serving, whose defaults are offline. A model checkpoint alone does not supply the processor/tokenizer cache. See [third-party boundaries](../../docs/THIRD_PARTY.md).

The release has a 60-wide action head; SONIC needs the fork's **80-wide** head. The 80-wide head carries the 78-D SONIC action (`lab list-models` reports this data width as `action_dim=78`) plus padding. Prepare a separate checkpoint with the pinned [widening tool](https://github.com/LocoManiLab/Xiaomi-Robotics-1/blob/8598deffcc560c4ca8aad616c82f925b248de8c1/xr1/tools/widen_action_dim.py), explicitly in the model environment. From the main checkout root:

```bash
conda run -n xr1 python external/Xiaomi-Robotics-1/xr1/tools/widen_action_dim.py \
  --input workspaces/xr1/weights/model_states.pt \
  --output workspaces/xr1/weights/model_states_sonic80.pt --action-dim 80
```

The script pads the action-facing projectors; merely changing the model dimension flag does not reshape released weights. From the workspace, `lab doctor --model xr1 --stage train` checks checkout/conda prerequisites, not GPU capacity or checkpoint tensor contents.

## Real SONIC data, conversion, and training

This integration is **SONIC-focused**, not a general adapter for every upstream XR-1 embodiment. Raw action is 78 = motion token 64 + hands 7 + 7; the fork packs it to `(30, 80)` with two reserved channels. State 43 is packed into the existing 60-wide state tensor. Motion tokens remain absolute; do not reinterpret them as the upstream end-effector-delta action space.

The manifest currently requires `video.ego_view`, `state.joint_pos`, and `action.motion_token/left_hand_joints/right_hand_joints` metadata groups. Supply actual aligned, compatible data and preserve those feature mappings. From `workspaces/xr1`:

```bash
DATASET=datasets/sonic_demo
lab validate "$DATASET"
lab convert --model xr1 --dataset "$DATASET" --dry-run
lab train --model xr1 --dataset "$DATASET" --task first-run --gpus 1 \
  --set trainer.max_steps=10 --set trainer.save_interval=5 --dry-run
```

Remove conversion's `--dry-run` first. It writes JSON episodes under `datasets/xr1/sonic_demo` and generates `xr1/configs/data/locomanilab_sonic_demo.yaml` inside the selected model checkout, including normalization statistics. The native config name is derived from the dataset basename; keep generated data/configs local. Continue passing the **original dataset** to `train`, which selects that native config. Execute training only once these artifacts, widened weights, and the complete local VLM cache are ready. A short run still loads/trains the full GPU model.

## Local serving and optional evaluation

Use the entire native run directory, not a base `.pt` file. For the example's project/run names it is `checkpoints/xr1/project_locomanilab/first-run`, with `config.py` and saved Lightning checkpoints (`last.ckpt`, possibly a directory). Retain the complete checkpoint/config and referenced local model/cache assets.

```bash
RUN=checkpoints/xr1/project_locomanilab/first-run
PROMPT='the instruction recorded in your demonstrations'
lab deploy --model xr1 --model-path "$RUN" --prompt "$PROMPT" --port 5550 --dry-run
lab eval --model xr1 --sim --model-path "$RUN" --backend sonic \
  --sonic-ckpt low_latency --env EmptyEnv --episodes 1 --max-steps 20 \
  --prompt "$PROMPT" --video-dir runs/eval/xr1 --dry-run
```

These are previews. Remove `--dry-run` only after the corresponding prerequisites are ready. `deploy` starts the compatible policy server rather than the upstream tmux launcher or a robot controller. `eval` owns a separate server lifecycle; stop a prior server on its port first. The server resets its per-episode action/prefix state through the protocol's `reset` endpoint. Simulation additionally needs the separately provisioned `sim` profile, simulator environment, robot/controller assets, and the SONIC variant used for collection. `EmptyEnv` checks wiring, not task success. See [simulation setup](../../docs/sim_validation.md).

A run directory fine-tuned on the benchmark's train50 split (step 50000) is released on the Hugging Face Hub; see [data, assets, and checkpoints](../../docs/hf_resources.md). It holds `config.py`, `config.yaml`, and `epoch=0-step=50000.ckpt/` with the model states only (no optimizer shards, so it cannot resume training) and no `last.ckpt`, so select the step explicitly. It still needs the local Qwen3-VL cache described above. Serving it has not been validated for this guide.

```bash
python scripts/hf/download.py checkpoint xr1 --out checkpoints/hf
lab deploy --model xr1 --model-path checkpoints/hf/checkpoint/xr1 --ckpt-step 50000 --prompt "$PROMPT" --port 5550 --dry-run
```

Common failures: action-shape mismatches mean the released 60-D weights/config were used instead of the SONIC 80-D path; missing generated YAML means real conversion/statistics did not finish; missing `config.py` means a checkpoint file was passed instead of a run; offline Qwen errors mean its cache is incomplete. New generated data configs use the `locomanilab_` prefix; existing configs remain readable.
