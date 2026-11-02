# DreamZero

Original project: [NVIDIA GEAR / dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero). This guide uses [LocoManiLab/dreamzero](https://github.com/LocoManiLab/dreamzero) at **`7cd2c4c3c167d61e1364e13c0201312ab0b99ad7`**, with humanoid metadata, embodiment/data configs, and a stateful compatible server. Its model/GPU/hardware execution has not been revalidated end to end for this release.

Sources: [main adapter](../../src/locomanilab/models/dreamzero.py), [manifest](../../src/locomanilab/configs/dreamzero.yaml), pinned [GEAR converter](https://github.com/LocoManiLab/dreamzero/blob/7cd2c4c3c167d61e1364e13c0201312ab0b99ad7/scripts/data/convert_lerobot_to_gear.py), [SONIC Wan2.2 data config](https://github.com/LocoManiLab/dreamzero/blob/7cd2c4c3c167d61e1364e13c0201312ab0b99ad7/groot/vla/configs/data/dreamzero/unitree_g1_sonic_relative_wan22.yaml), and [policy server](https://github.com/LocoManiLab/dreamzero/blob/7cd2c4c3c167d61e1364e13c0201312ab0b99ad7/deployment/run_dreamzero_server.py).

## CPU acceptance

From the main checkout root, with main `.[data]` installed and `lab` on `PATH`. No model checkout, weights, GPU, or simulator is used; an existing workspace is preserved.

<!-- cpu-smoke:start -->
```bash
(
  set -eu
  if [ -e workspaces/dreamzero ] || [ -L workspaces/dreamzero ]; then
    echo "workspaces/dreamzero already exists; use a fresh workspace." >&2
    exit 1
  fi
  mkdir -p workspaces/dreamzero
  cp policies/dreamzero/workspace.example.yaml workspaces/dreamzero/env.local.yaml
  cd workspaces/dreamzero
  lab doctor --stage data
  lab sample-data --out datasets/cpu-example
  lab validate datasets/cpu-example
  lab prepare datasets/cpu-example --merge-only --out datasets/cpu-prepared
  lab validate datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

The synthetic sample is not a humanoid training demonstration. [Example configured paths](workspace.example.yaml) are relative to `workspaces/dreamzero`; CLI dataset/checkpoint inputs are relative to the calling directory. Merge settings for an existing workspace instead of replacing its configuration.

## Explicit environment and local weights

From the main checkout root:

```bash
python3 scripts/setup/profiles.py dreamzero
python3 scripts/setup/profiles.py dreamzero --apply
```

Follow the [fixed-pin installation instructions](https://github.com/LocoManiLab/dreamzero/blob/7cd2c4c3c167d61e1364e13c0201312ab0b99ad7/README.md#installation) in the separate conda environment `dreamzero` (Python 3.11, CUDA 12.9+ stack). Its basic editable install is `pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129`, followed by the native FlashAttention setup. Read the hardware-specific sections before installing Transformer Engine/TensorRT; GB200-specific steps are not universal requirements. The original distributed 14B inference hardware guidance is not a validation claim for this fork's G1 server. The compatible server also imports `pyzmq`/`msgpack-numpy`; ensure they exist in the model environment.

Select the backbone deliberately:

| Backbone | Locally required components | Meaning of this main-adapter path |
|---|---|---|
| `wan22` (default) | Wan2.2-TI2V-5B DiT, its T5 encoder and `Wan2.2_VAE.pth`; Wan2.1 CLIP encoder; UMT5 tokenizer | Starts from the Wan video prior with a newly initialized action head |
| `wan21` | Wan2.1 14B DiT, T5/CLIP, `Wan2.1_VAE.pth`, tokenizer; optionally a released DreamZero base | Can use `paths.dreamzero_pretrained` for the native released-checkpoint path |

Stage these large assets explicitly under their own terms; the profile helper downloads no weights. The [manifest](../../src/locomanilab/configs/dreamzero.yaml) gives exact component filenames, including `models_t5_umt5-xxl-enc-bf16.pth` and `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`. The workspace example supplies all mandatory path keys for the default backbone. Do not put a released Wan2.1 DreamZero checkpoint into the Wan2.2 5B path and assume the tensors match. The default training architecture is full training; a single LoRA flag is not a complete replacement for the native coupled architecture/DeepSpeed settings. See [third-party boundaries](../../docs/THIRD_PARTY.md).

From the workspace, `lab doctor --model dreamzero --stage train` checks the selected source/conda environment; it does not validate these components or GPU capacity.

## Real data and training

The complete main-CLI example below is for **SONIC state 43 / action 78**. The fork has SONIC data configs for Wan2.1/Wan2.2. It also has an AMO training lane, but only a Wan2.1 data config at this pin; the main adapter's deployment tag is fixed to SONIC. This guide does not claim a complete AMO serving/evaluation path or HeFT support. An unrecognized controller must not be treated as SONIC merely because a fallback produces a command.

Supply real compatible data, preserve its controller/feature mappings, then run these plans from `workspaces/dreamzero`:

```bash
SOURCE=datasets/sonic_demo
lab validate "$SOURCE"
lab convert --model dreamzero --dataset "$SOURCE" --dry-run
```

Inspect that plan, then remove conversion's `--dry-run` to create the prepared directory. Only after conversion succeeds, inspect training:

```bash
PREPARED=datasets/gear/sonic_demo
lab train --model dreamzero --dataset "$PREPARED" --config-name wan22 \
  --task first-run --gpus 1 --set max_steps=10 --set save_steps=5 --dry-run
```

The **prepared directory** links data/videos to the source, while metadata is copied and extended with `embodiment.json` and complete statistics. Keep the source available. Train on the prepared directory printed by conversion, not the raw source. Inspect its `original_key` mappings and statistics; matching widths alone do not establish action semantics. Real conversion and GPU assets are required before executing training. Choose a fresh run; a short run still loads the full model and does not guarantee it fits one GPU.

## Local serving and optional simulation

Use a complete saved checkpoint such as `checkpoints/dreamzero/first-run/checkpoint-10`, including its `config.json` and `experiment_cfg/`. Keep the video backbone assets and correct saved configuration accessible. The adapter overrides the tokenizer path, not every path embedded in a checkpoint from another machine.

```bash
CHECKPOINT=checkpoints/dreamzero/first-run/checkpoint-10
PROMPT='the instruction recorded in your demonstrations'
lab deploy --model dreamzero --model-path "$CHECKPOINT" --prompt "$PROMPT" --dry-run
lab eval --model dreamzero --sim --model-path "$CHECKPOINT" --backend sonic \
  --sonic-ckpt low_latency --env EmptyEnv --episodes 1 --max-steps 20 \
  --prompt "$PROMPT" --video-dir runs/eval/dreamzero --dry-run
```

These are plans; execute by removing `--dry-run` only after provisioning the relevant environment/assets. `deploy` starts the compatible server and its required local distributed process group, not a robot controller. `eval` launches its own server, so stop a separate server on that port first. The server retains per-episode frame history and a video KV cache: clients must call `reset` between episodes. A responding socket is insufficient to establish correct stateful inference.

Simulation additionally needs the `sim` source profile, simulator environment, robot/controller assets, and the SONIC checkpoint variant used for collection. `EmptyEnv` is a wiring example, not a score; use the recorded task/reset protocol for actual evaluation. See [simulation setup](../../docs/sim_validation.md).

Common failures: missing `embodiment.json`/original-key stats means conversion was skipped or the wrong dataset was passed to train; missing Wan/CLIP/tokenizer components cannot be fixed by an action checkpoint alone; AMO+Wan2.2 has no data config at this pin; cross-episode drift can indicate a missing reset. Preserve the fork's embodiment registration and checkpoint/normalization conventions.
