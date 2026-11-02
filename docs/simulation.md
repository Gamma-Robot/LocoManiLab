# Minimal simulation setup

Start with **EmptyEnv and one controller**, then add assets for the task you
actually need. `humanoid_benchmark` supplies the MuJoCo environment; `g1_backend`
supplies the shared controllers and recording stack. Learned policies run in a
separate model environment.

The maintained simulator defaults to `G1FloatingBody` and also keeps `G1` and
`G1FixedLowerBody`. Other robot models, their dedicated demos, configurations,
and assets are retired. Existing G1 task definitions, scenes, controller
configurations, and shared simulation framework remain. See the simulator's
[G1 scope guide](https://github.com/LocoManiLab/humanoid_benchmark/blob/main/docs/g1_scope.md)
for interface changes and focused CPU checks. Listing tasks does not require
optional scene packs or controller weights; creating a selected task still
requires its resources.

## Source and workspace

From the LocoManiLab checkout root, with the main CLI installed and activated:

```bash
python3 scripts/setup/profiles.py sim
# Explicit source initialization; no controller weights or scene packs.
python3 scripts/setup/profiles.py sim --apply
lab init workspaces/sim
```

The gitlinks select the integration revisions; do not replace them with a
floating upstream checkout. The standalone
[humanoid_benchmark installation guide](https://github.com/LocoManiLab/humanoid_benchmark#installation)
uses that repository's maintained `main` branch instead. Existing `repos.*`
overrides remain supported.

Edit the newly created `workspaces/sim/env.local.yaml` for this layout:

```yaml
schema_version: 2
role: [data, collect, eval]
repos:
  sim_repo: ../../external/humanoid_benchmark
  g1_backend: ../../external/g1_backend
paths:
  logs: runs
deploy: {}
conda_envs: {}
```

These relative paths assume the configuration stays in `workspaces/sim`.
`lab` finds conda through `$CONDA_EXE` or a per-user `anaconda3`, `miniconda3`,
`miniforge3`, or `mambaforge` installation; for another location add
`conda_root: /path/to/conda` to this file.
Use your actual source paths if you keep provisioned checkouts elsewhere.

## Separate simulator environment

The following explicit installation can download Python/packages, including
model-runtime dependencies. It does not request scene packs or controller
weights. Use a fresh Conda environment; preserve an existing installation and
inspect it separately. Run from the main checkout root:

```bash
(
set -eu
conda create -c conda-forge -n tele_benchmark python=3.11
conda run -n tele_benchmark python -m pip install -e external/humanoid_benchmark/robosuite_src
conda run -n tele_benchmark python -m pip install -e external/humanoid_benchmark
conda run -n tele_benchmark python -m pip install -e external/g1_backend --no-deps --no-build-isolation --ignore-requires-python
conda run -n tele_benchmark python -m pip install "pyzmq==27.1.0" "msgpack==1.2.1" \
  "msgpack-numpy==0.4.8" "onnx==1.17.0" "tyro==1.0.15" "sshkeyboard==2.3.1" "tensordict==0.9.1"
conda run -n tele_benchmark python -m pip install --no-deps \
  "any4hdmi @ git+https://github.com/EGalahad/any4hdmi.git@5d0ae1915c904c64f2480a055aa1693523b9e4b4" \
  "mjhub @ git+https://github.com/EGalahad/mjhub.git@f118fae0e54aeb3484a9f576658217528427042b"
conda run -n tele_benchmark python -m robocasa.scripts.setup_macros
)
```

The compatibility flags preserve the simulator's dependency pins while installing
`g1_backend` in its Python 3.11 environment; the runtime metadata specifies Python
3.10. Because `g1_backend` is installed without dependencies, its runtime
requirements that the simulator does not already provide (the ZMQ/msgpack policy
transport, ONNX, `tyro`, `sshkeyboard`) are installed next at the versions locked in
`g1_backend/uv.lock`. Its motion library `any4hdmi` (imported whenever `robocasa`
registers its tasks) and that library's `mjhub` are installed at their locked commits
without dependencies: their unpinned requirements would otherwise replace the
simulator's `mujoco==3.3.1` and its PyTorch build. `tensordict==0.9.1` is the one
additional runtime dependency they need; pip then warns that `any4hdmi`'s optional
viewer and analysis packages (`viser`, `mjviser`, `matplotlib`, `scikit-learn`) are
absent; the collection and evaluation commands on this page do not import them. The final command writes local simulator macros. Follow the selected native revision's platform/rendering instructions for
system dependencies. Keep the main CLI environment active in your workflow
terminal; `lab` launches the simulator environment independently.

To use another environment name (for example, when `tele_benchmark` already exists
for other work), replace it in the commands above and pass the name to `lab`:
`collect --sim-conda-env <name>`, `eval --sim-env <name>`, and
`doctor --sim --sim-env <name>`.

## Select controller weights

Controller weights are published under `checkpoint/controller/` in the Hugging Face
repository [`GammaRobotics/LocoManiLab`](https://huggingface.co/datasets/GammaRobotics/LocoManiLab);
the simulator helper below fetches one selected family into `external/g1_backend/checkpoints`.

For SONIC `v1_1`, first inspect an **offline plan**:

```bash
conda run -n tele_benchmark python external/humanoid_benchmark/robocasa/scripts/download_g1_checkpoints.py --family sonic_v1_1 --dest external/g1_backend/checkpoints --plan
```

`--plan` makes no Hub requests or downloads. If you want a current remote file
list and exact sizes, replace it with `--dry-run` (`--dryrun` is an alias): that
step queries Hub metadata but transfers no weights. To explicitly fetch the
selection, omit the planning flag; the helper asks before transfer.

`--family` or `--all` is mandatory. The `sonic_v1_1` selection automatically adds
its shared planner, without fetching the low-latency encoder/decoder or unrelated
AMO/HeFT/ScaleBFM/ALMI weights. Choose another family only for the controller you
intend to run, and preserve the Git-tracked policy manifests. SONIC variants
must match the decoder used for recorded motion tokens.

## Assets by task

| First goal | What it needs |
|---|---|
| `EmptyEnv` | Native floor plus the bundled G1 model/meshes (about 105 MiB in source), selected controller weights, and a working renderer; no external room or object pack |
| `TabletopPickPlace` | Shared textures and Objaverse mug/plate models; preview categories with `download_kitchen_assets.py --type tex objs_objaverse --dry-run` in the simulator checkout |
| Other kitchen/living-room task | Its layout/style, furniture, texture, and object dependencies; select only relevant archive categories |
| iTHOR FloorPlan | Scene and THOR dependencies; the current installer still fetches the shared object library, about 13 GB on first use, even when selecting one scene |

The kitchen downloader requires `--type`; its `--dry-run` is an offline category
plan. `--type all` explicitly selects the full collection. Category archives can
still be large and are not individual-object downloads. EmptyEnv needs none of
them. Use the simulator's [asset instructions](https://github.com/LocoManiLab/humanoid_benchmark#assets-by-environment)
for the selected task and retain third-party asset terms.

For the benchmark tasks, `scripts/hf/download.py assets` fetches exactly the files the
selected tasks load, each from its original host and sha256-verified, instead of whole
libraries: one iTHOR task needs about 32 MB, all 234 tasks plus the randomization texture
pool about 3.4 GB. See [data, assets, and checkpoints](hf_resources.md):

```bash
python scripts/hf/download.py assets --task IThorKitchenOpenFridge_scene13 --assets-dir external/humanoid_benchmark/robocasa/models/assets
python scripts/hf/download.py assets --preset all --assets-dir external/humanoid_benchmark/robocasa/models/assets
```

## Check the selected lane

With the simulator environment and selected local weights available:

```bash
lab --config workspaces/sim/env.local.yaml doctor --stage collect --sim --backend sonic --sonic-ckpt v1_1
lab --config workspaces/sim/env.local.yaml collect --sim --env EmptyEnv --backend sonic --sonic-ckpt v1_1 --pico-source mock --task-prompt 'free teleoperation' --dry-run
```

The collection command prints a plan. Remove `--dry-run` only to intentionally
start simulation. EmptyEnv has no task-success score; it checks the environment
and controller wiring. PICO services are needed only for real headset input,
not the mock-input plan.

For learned-policy evaluation, also provision the chosen model/checkpoint and
add its `repos.*` path to this workspace, following its [policy guide](../policies/README.md).
A first evaluation plan should explicitly use EmptyEnv, for example:

```bash
lab --config workspaces/sim/env.local.yaml eval --model gr00t --sim --model-path /path/to/local/gr00t-checkpoint --backend sonic --sonic-ckpt v1_1 --env EmptyEnv --episodes 1 --prompt 'the training instruction' --dry-run
```

A simulator `eval --dry-run` resolves the selected controller first, so it reports
missing SONIC weights before printing the plan; download them as shown above. It
does not load the model or start the simulator.

This page describes source, asset selection, and command prerequisites; a plan
or `doctor` result does not establish model inference, rendering, controller
stability, or hardware readiness. See [simulation evaluation](sim_validation.md)
for measurement and validation boundaries.
