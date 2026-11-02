# Diffusion Policy (DP)

Use `--model dp`; the native LeRobot policy type is `diffusion`. DP shares the [LeRobot adapter](../../src/locomanilab/models/lerobot_baseline.py), conversion format, environment, and server with [ACT](../act/README.md). Model-specific defaults remain in the [DP manifest](../../src/locomanilab/configs/dp.yaml); no separate implementation or wrapper repository is introduced here.

The source is [Hugging Face LeRobot](https://github.com/huggingface/lerobot) (upstream, unmodified), pinned to **v0.3.3 / `b883328e6c95681ca90a18b102e4ae5e1f91e2bf`** by the main gitlink. See the pinned [source](https://github.com/huggingface/lerobot/tree/b883328e6c95681ca90a18b102e4ae5e1f91e2bf) and [license](https://github.com/huggingface/lerobot/blob/b883328e6c95681ca90a18b102e4ae5e1f91e2bf/LICENSE). A newer upstream installation is not a compatible drop-in replacement for this v2.1 integration.

## CPU data walkthrough

Run from the **LocoManiLab checkout root**, with the main package's `[data]` extra installed and its environment activated. This block needs no policy source, model weights, simulator, or GPU. It requires a new `workspaces/dp` directory and refuses an existing workspace or symlink.

<!-- cpu-smoke:start -->
```bash
(
set -eu
test ! -e workspaces/dp && test ! -L workspaces/dp || {
  echo "workspaces/dp already exists; choose a fresh workspace before running this example." >&2
  exit 1
}
mkdir -p workspaces/dp
cp policies/dp/workspace.example.yaml workspaces/dp/env.local.yaml
lab --config workspaces/dp/env.local.yaml doctor --stage data
lab --config workspaces/dp/env.local.yaml sample-data --out datasets/cpu-example
lab --config workspaces/dp/env.local.yaml validate workspaces/dp/datasets/cpu-example
lab --config workspaces/dp/env.local.yaml prepare workspaces/dp/datasets/cpu-example --merge-only --out datasets/cpu-prepared
lab --config workspaces/dp/env.local.yaml validate workspaces/dp/datasets/cpu-prepared
)
```
<!-- cpu-smoke:end -->

The sample's state[4]/action[2] validates the **data pipeline**, not DP training. The [workspace example](workspace.example.yaml) belongs at `workspaces/dp/env.local.yaml`: relative `repos.*` then finds `../../external/lerobot`, and data/output paths stay within this workspace. If you move the configuration or use an existing checkout, update `repos.*`. CLI dataset/checkpoint inputs remain relative to the invoking directory.

## Shared source and environment

`python3 scripts/setup/profiles.py dp` prints the dependency plan. Its explicit `--apply` form initializes the same `external/lerobot` gitlink as ACT; it does not install a model environment or fetch weights. Follow [ACT's shared installation section](../act/README.md#install-the-shared-lerobot-environment) once for both models, using Python 3.10 or 3.11 and the pinned native dependency constraints. Do not create another environment just for this README. Native v0.3.3 already includes the diffusion dependency; the additional server transport packages are shared too.

After provisioning, check the DP selection:

```bash
lab --config workspaces/dp/env.local.yaml doctor --model dp --stage train
```

This checks paths/prerequisites, not GPU operation or a saved policy. Source installation, Python/package downloads, and optional runtime/weight acquisition are explicit user actions, outside the CPU block. Model execution uses `uv run --no-sync` and will not repair a missing backend installation.

## Data and training

This example uses **G1 SONIC**: RGB ego video, state[43], and action[78] = motion_token[64] + left/right hand targets[7+7]. See [the complete shared contract](../act/README.md#robot-data-contract-and-training), including source-column names, episode/statistics requirements, and controller-checkpoint provenance. DP is not language-conditioned. Another action layout or SONIC decoder variant requires a separate compatibility check.

Place real prepared demonstrations at `workspaces/dp/datasets/robot/prepared`, or replace the source path consistently:

```bash
lab --config workspaces/dp/env.local.yaml validate workspaces/dp/datasets/robot/prepared
lab --config workspaces/dp/env.local.yaml convert --model dp --dataset workspaces/dp/datasets/robot/prepared --dry-run
lab --config workspaces/dp/env.local.yaml convert --model dp --dataset workspaces/dp/datasets/robot/prepared
lab --config workspaces/dp/env.local.yaml train --model dp --dataset workspaces/dp/datasets/robot/prepared --gpus 1 --gpu-ids 0 --task first-run --set batch_size=4 --set steps=100 --set save_freq=100 --dry-run
# Explicit short GPU training run after checking the data and command:
lab --config workspaces/dp/env.local.yaml train --model dp --dataset workspaces/dp/datasets/robot/prepared --gpus 1 --gpu-ids 0 --task first-run --set batch_size=4 --set steps=100 --set save_freq=100
```

`train` still takes the original source path; it selects `workspaces/dp/datasets/lerobot/prepared`. The run goes to `workspaces/dp/outputs/lerobot/dp/first-run`. The adapter supports one GPU, and the smaller example batch is an explicit override of the manifest default. A command plan and a short run do not establish policy quality.

ACT/DP conversion can be reused **for the same source path, metadata, and contract**. To reuse ACT's converted copy, set `paths.lerobot_data_root: ../act/datasets/lerobot` in the DP workspace and pass the same source path used for ACT. A source mismatch, changed metadata, or an old conversion without a source record is refused. Choose a fresh converted-data root and reconvert; do not edit the original demonstrations to bypass this check. The independent workspace examples use separate converted roots by default. Converted videos link to their source, which must remain available.

## Local server and optional simulation

Once a checkpoint exists:

```bash
lab --config workspaces/dp/env.local.yaml deploy --model dp --model-path workspaces/dp/outputs/lerobot/dp/first-run --dry-run
lab --config workspaces/dp/env.local.yaml deploy --model dp --model-path workspaces/dp/outputs/lerobot/dp/first-run
```

This starts the shared policy server on port 5550; it does not actuate hardware. A run directory selects its newest `pretrained_model`; use an explicit saved checkpoint for a fixed selection. Stop that server before `eval`, which launches its own.

With the simulator/controller sources, environment, matching controller weights, and rendering already provisioned, inspect a plan for data recorded with SONIC `v1_1`:

```bash
lab --config workspaces/dp/env.local.yaml eval --model dp --sim --model-path workspaces/dp/outputs/lerobot/dp/first-run --backend sonic --sonic-ckpt v1_1 --env EmptyEnv --episodes 1 --max-steps 60 --video-dir runs/eval/dp --dry-run
```

Select your data's actual SONIC variant. Read [ACT's simulation prerequisites](../act/README.md#local-server-and-optional-simulation) before removing `--dry-run`; `EmptyEnv` is a wiring check rather than a task-success benchmark. Hardware runtime readiness is separate from the policy server.

## Validation and troubleshooting

CPU data/configuration/converter checks and static inspection of the pinned native API are distinct from fresh GPU training, policy inference, simulation task success, or real-robot validation; those model/hardware checks have **not** been established for this pilot. A successful `doctor` or `--dry-run` does not change that status.

See [shared troubleshooting](../act/README.md#validation-and-common-issues) for missing environments, incompatible data, offline caches, protected outputs, and native resume configuration. DP additionally needs a checkpoint whose diffusion observation history and action horizon agree with training; the shared server resets its history between episodes. Defaults remain in the [manifest](../../src/locomanilab/configs/dp.yaml) and implementation remains in the shared adapter/runtime.

- Image/crop mismatch: the DP manifest defaults to `policy.crop_shape=[216,288]`; real RGB frames must be at least that large in height/width. The 64×64 CPU sample cannot train this recipe even after changing its action width. For a deliberately reviewed smaller real-image recipe, an override such as `--set 'policy.crop_shape=[128,128]'` is possible only when the actual data and preprocessing support it; do not change the crop blindly.
