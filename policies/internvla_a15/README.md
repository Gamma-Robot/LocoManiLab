# InternVLA-A1.5

Original project: [InternRobotics/InternVLA-A-series](https://github.com/InternRobotics/InternVLA-A-series). This entry requires the [LocoManiLab/InternVLA-A1.5 fork](https://github.com/LocoManiLab/InternVLA-A1.5) at **`f59f7036bb5e53d5f682c820b7729644d2fef078`**, including its G1 schema, conversion, widened-checkpoint, and serving changes. Model/GPU/hardware execution has not been revalidated end to end for this release. The upstream code/weight terms include noncommercial restrictions; see [third-party boundaries](../../docs/THIRD_PARTY.md).

Sources: [main adapter](../../src/locomanilab/models/internvla_a15.py), [manifest](../../src/locomanilab/configs/internvla_a15.yaml), pinned [v2.1-to-v3 converter](https://github.com/LocoManiLab/InternVLA-A1.5/blob/f59f7036bb5e53d5f682c820b7729644d2fef078/scripts/data/locomanilab_lerobot_v21_to_v30.py), [G1 schema](https://github.com/LocoManiLab/InternVLA-A1.5/blob/f59f7036bb5e53d5f682c820b7729644d2fef078/src/lerobot/dataset_schemas/configs/unitree_g1_sonic.yaml), and [compatible server](https://github.com/LocoManiLab/InternVLA-A1.5/blob/f59f7036bb5e53d5f682c820b7729644d2fef078/deployment/run_internvla_server.py).

## CPU acceptance

From the main checkout root, with main `.[data]` installed and `lab` on `PATH`. This creates a fresh workspace without model sources, weights, GPU, or simulator.

<!-- cpu-smoke:start -->
```bash
(
  set -eu
  if [ -e workspaces/internvla_a15 ] || [ -L workspaces/internvla_a15 ]; then
    echo "workspaces/internvla_a15 already exists; use a fresh workspace." >&2
    exit 1
  fi
  mkdir -p workspaces/internvla_a15
  cp policies/internvla_a15/workspace.example.yaml workspaces/internvla_a15/env.local.yaml
  cd workspaces/internvla_a15
  lab doctor --stage data
  lab sample-data --out datasets/cpu-example
  lab validate datasets/cpu-example
  lab prepare datasets/cpu-example --merge-only --out datasets/cpu-prepared
  lab validate datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

The synthetic sample is not valid G1 training data. [Configured paths](workspace.example.yaml) resolve from `workspaces/internvla_a15`; CLI dataset/checkpoint inputs resolve from the caller's directory. Merge an existing configuration rather than overwriting it.

## Explicit environment and checkpoint preparation

From the main checkout root:

```bash
python3 scripts/setup/profiles.py internvla_a15
python3 scripts/setup/profiles.py internvla_a15 --apply
```

Follow the **[installation tutorial at the fixed pin](https://github.com/LocoManiLab/InternVLA-A1.5/blob/f59f7036bb5e53d5f682c820b7729644d2fef078/tutorials/installation.md)** in a separate conda environment named `internvla_a1_5`. It specifies Python 3.11, CUDA 12.8, PyTorch 2.10.0, Transformers 5.2.0, FFmpeg/SVT-AV1, and the listed attention kernels. Its Transformers replacement-module step is required: apply it only to that isolated model environment. Installing the main CLI or merely creating the conda environment does not perform those steps. No model installation or download is part of the CPU block.

Acquire local [InternVLA-A1.5-base](https://huggingface.co/InternRobotics/InternVLA-A1.5-base) and `Qwen/Qwen3.5-2B` files under their own terms. The released action/state maxima are 32; changing training flags to 78/43 does not resize the tensors. In the provisioned model environment, explicitly run the pinned [widening tool](https://github.com/LocoManiLab/InternVLA-A1.5/blob/f59f7036bb5e53d5f682c820b7729644d2fef078/tools/widen_for_g1_sonic.py) on a separate output directory. From the main checkout root:

```bash
conda run -n internvla_a1_5 python external/InternVLA-A1.5/tools/widen_for_g1_sonic.py \
  --src workspaces/internvla_a15/weights/InternVLA-A1.5-base \
  --dst workspaces/internvla_a15/weights/InternVLA-A1.5-g1-sonic \
  --max-action-dim 78 --max-state-dim 43
```

The example config points at this widened directory and local Qwen directory. Default main-adapter training is **action-only** with video-foresight supervision disabled; it does not require downloading WAN2.2 video-generation weights for that path. Other native training modes have additional assets and are not established by this guide. From the workspace, `lab doctor --model internvla_a15 --stage train` checks the selected checkout/conda environment, not the checkpoint shapes, GPU, or patch correctness.

## Real SONIC data and training

This adapter's registered lane is **SONIC state 43 / action 78**, using `robot_type=unitree_g1_sonic` and absolute actions. The metadata must include `video.ego_view`, `action.motion_token`, and the manifest's left/right arm, hand, and waist state groups. HeFT/AMO are not covered. Preserve the pinned schema: an unknown robot type in the native registry can fall back to an unrelated identity mapping.

Supply real compatible v2.1 demonstrations at `datasets/sonic_demo`. From `workspaces/internvla_a15`:

```bash
DATASET=datasets/sonic_demo
lab validate "$DATASET"
lab convert --model internvla_a15 --dataset "$DATASET" --dry-run
lab train --model internvla_a15 --dataset "$DATASET" --task first-run --gpus 1 \
  --set steps=10 --set batch_size=1 --set num_workers=0 --set save_freq=5 --dry-run
```

Remove conversion's `--dry-run` first. It stages v2.1 links and materializes a v3.0 copy under `datasets/internvla_a15/v30/sonic_demo`. Continue passing the **original v2.1 dataset** to the main `train` command: it selects the v3.0 product and native repo ID itself. The adapter coordinates policy/dataset state tokenization and disables unrelated image-stat/VQA defaults. Inspect and execute training only after real conversion and widening succeed, using a fresh output run. A ten-step run still loads the complete GPU model.

## Local inference and optional simulation

Use a complete saved `pretrained_model` directory, including its configuration, weights, processor/normalization assets. From the workspace, after a real run has saved checkpoints:

```bash
CHECKPOINT=checkpoints/internvla_a15/first-run/checkpoints/last/pretrained_model
PROMPT='the instruction recorded in your demonstrations'
lab deploy --model internvla_a15 --model-path "$CHECKPOINT" \
  --prompt "$PROMPT" --port 5550 --dry-run
lab eval --model internvla_a15 --sim --model-path "$CHECKPOINT" --backend sonic \
  --sonic-ckpt low_latency --env EmptyEnv --episodes 1 --max-steps 20 \
  --prompt "$PROMPT" --video-dir runs/eval/internvla_a15 --dry-run
```

These are plans; remove `--dry-run` only after the corresponding environment/assets are ready. `deploy` starts a compatible policy server, not a hardware runtime. The local Qwen path is also used during serving. `eval` launches its own server, so do not leave a separate server on the same port. Simulation needs the separately provisioned `sim` profile, simulator environment, robot/controller assets, and the exact SONIC checkpoint variant used for collection. `EmptyEnv` tests wiring rather than task success. See [simulation setup](../../docs/sim_validation.md). Offline `infer-check` needs the **original v2.1 source**, not the v3.0 training copy.

Common failures: size-mismatch loading a 32-D base means widening was skipped; a missing G1 schema means the wrong fork/pin is checked out; missing v3 metadata means real conversion did not finish; missing Qwen/processor files cannot be fixed by a local action checkpoint alone. Keep the registered robot schema and the matching pinned converter together.
