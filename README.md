# LocoManiLab

A local workflow tool for humanoid policy research: **collect → prepare → convert → train → evaluate → deploy**.

LocoManiLab keeps dataset contracts, model adapters, and process orchestration in one small Python package. Model training, simulators, and robot controllers run in their own environments. You can use the CPU data tools without any model repositories, GPU, robot, or pretrained weights.

[中文快速开始](docs/quickstart.zh-CN.md) · [Policy guides](policies/README.md) · [Data, assets & checkpoints](docs/hf_resources.md) · [Documentation](docs/README.md) · [Model status](docs/models.md) · [Development](CONTRIBUTING.md)

> **Status:** the CPU sample-data workflow is the entry point. Model integrations are retained for research and have not all been revalidated end to end; see [model status](docs/models.md). LocoManiLab's own code is released under the [MIT License](LICENSE); dependencies, model weights, datasets, and simulator assets keep their own terms (see [third-party boundaries](docs/THIRD_PARTY.md)).

## Start on CPU

Use Python 3.10 or newer in a dedicated Conda (or Miniforge) or uv environment, not the system Python. Do not initialize submodules for this walkthrough.

```bash
git clone https://github.com/Gamma-Robot/LocoManiLab.git
cd LocoManiLab
conda create -y -n locomanilab python=3.10
conda activate locomanilab
python -m pip install -e '.[data]'

lab init demo-workspace
cd demo-workspace
lab sample-data --out datasets/example
lab validate datasets/example
lab prepare datasets/example --merge-only --out datasets/prepared
lab validate datasets/prepared
```

The example contains two short synthetic episodes with small videos. It checks installation, dataset validation, and preparation; it is not a robot demonstration or a training benchmark. Generated data stays in your workspace.

With uv instead of Conda, replace the first three commands with `uv venv --python 3.10 .venv`, `source .venv/bin/activate`, and `uv pip install -e '.[data]'`.

## Use a model when you need one

Start with the [ACT](policies/act/README.md), [DP](policies/dp/README.md), or
[pi05](policies/pi05/README.md) guide for a copyable workspace configuration and
policy-specific data, environment, and checkpoint requirements.

```bash
# From the source checkout: show a plan, without downloading anything.
python3 scripts/setup/profiles.py act

# Explicitly initialize only the ACT/DP source repository, at its recorded gitlink.
python3 scripts/setup/profiles.py act --apply
```

Provision that model's environment separately, set its checkout and data/output paths in your workspace's `env.local.yaml`, and inspect the selected prerequisites:

```bash
lab list-models
lab doctor --model act --stage train
```

The CPU sample does not satisfy a model's robot-data contract. For ACT/DP, configure `paths.lerobot_data_root` and `paths.lerobot_outputs`, provision the LeRobot uv environment, and convert representative robot demonstrations before training; see [local training](docs/training.md). See the [dependency profiles](dependencies/README.md) and [model matrix](docs/models.md) for other adapters. Source initialization does not install CUDA packages, download weights, build SDKs, or fetch scene assets. A policy server can use a local checkpoint; LocoManiLab does not manage a training cluster or upload datasets.

## Released data, assets, and checkpoints

The benchmark data, the simulator asset manifest, the released GR00T, DM0.5, and XR-1 checkpoints, and the G1 whole-body controller weights are on the Hugging Face Hub at [`GammaRobotics/LocoManiLab`](https://huggingface.co/datasets/GammaRobotics/LocoManiLab). Download only what you need:

```bash
python -m pip install -e '.[data,hf]'
python scripts/hf/download.py list
python scripts/hf/download.py dataset --task IThorKitchenOpenFridge_scene13 --out datasets/hf
python scripts/hf/download.py checkpoint gr00t --out checkpoints/hf
python scripts/hf/minimal_test.py --work hf-minimal-test     # minimal download-and-test unit
```

Simulator assets are verified and extracted into an initialized simulator checkout with `scripts/hf/download.py assets`. See [data, assets, and checkpoints](docs/hf_resources.md) for the repository layout, asset presets, checkpoint requirements, and all test stages.

## Evaluate on benchmark tasks

The benchmark's 234 tasks and their evaluation settings (environment, scene, prompt, episodes, step budget, genuine-success threshold, SONIC variant, fixed-scene and reset-jitter seeds) are recorded in [`benchmark/tasks.yaml`](src/locomanilab/benchmark/tasks.yaml). Task names match the dataset directories (`<type>/<split>/<task>`). With the simulator environment, the task's scene assets, and a trained checkpoint in place, `--task` fills these settings for one task and `--task-set` evaluates a selection one task after another:

```bash
lab tasks atomic/seen                    # list tasks: all | atomic | composite | seen | unseen_task | unseen_scene | <type>/<split>
lab eval --model gr00t --sim --model-path /path/to/checkpoint --task IThorKitchenOpenFridge_scene13
lab eval --model gr00t --sim --model-path /path/to/checkpoint --task-set atomic/seen
```

Options given on the command line override the table, e.g. `--episodes 2` for a quick check. Each task writes its videos and `results.json` to `runs/eval/<model>/<task>/` (`--video-dir` sets the root); a rerun skips tasks whose results are complete, and `summary.json` in the root reports per-task, per-group and overall success rates. See [simulation validation](docs/sim_validation.md) for the pinned simulator revision and per-task settings.

## Collection and robot deployment

Simulation and real-robot collection use the shared `g1_backend` controller and recording stack; `humanoid_benchmark` supplies the simulation host. `--backend heft` remains available within that runtime and does not need a separate HeFT source checkout. See [collection](docs/data_collection.md) and [controller contracts](docs/controller.md).

For complete GR00T/FastWAM deployment on a physical robot, this project's **recommended best practice** is the native GR00T WholeBodyControl runtime and its model/camera/control environment. See [deployment guidance](docs/deployment.md).

## Commands

| Command | Purpose |
|---|---|
| `init` | Create a workspace configuration |
| `sample-data` | Generate the small CPU example |
| `validate`, `prepare` | Check and prepare the shared dataset |
| `list-models`, `doctor` | Inspect adapters and local configuration |
| `convert`, `train` | Call a selected model's conversion/training entry point |
| `tasks` | List the benchmark tasks and their evaluation settings |
| `eval`, `infer-check` | Run closed-loop simulation (one benchmark task with `--task`, a selection with `--task-set`) or offline policy checks |
| `collect`, `replay` | Work with an explicitly provisioned simulator or robot runtime |
| `deploy` | Start a model from a local checkpoint |

Use `lab <command> --help` for current options. Model execution commands support `--dry-run`; it shows the planned command and does not certify that a model environment, checkpoint, or robot is ready. The retired online RL launcher has been removed; model training uses the supervised `train` workflow.

## Repository layout

```text
src/locomanilab/           Python package: CLI, data tools, adapters, runtime helpers
external/                 Optional model/controller/simulator submodules
policies/                 Per-policy guides and portable workspace examples
scripts/setup/            Source dependency plans
scripts/diagnostics/      Optional runtime diagnostics
scripts/migrations/       Explicit dataset maintenance tools
dependencies/             Dependency profiles; revisions come from gitlinks
docs/                     Public guides and support boundaries
tests/                    CPU tests and optional environment-specific checks
```

The source checkout is not the workspace. Keep local datasets, checkpoints, and machine settings outside the package. [Architecture](DESIGN.md) explains the boundaries; [Contributing](CONTRIBUTING.md) covers CPU validation and standard Git submodule maintenance.

## Acknowledgements

LocoManiLab integrates its whole-body controllers (SONIC, HEFT, AMO, ScaleBFM, ALMI) through [g1_backend](https://github.com/LocoManiLab/g1_backend), which builds directly on the open-source [sim2real](https://github.com/EGalahad/sim2real) deployment stack. We thank its authors and the authors of the controllers it runs.

## Earlier project name

Use `lab` as the primary command or `python -m locomanilab` to run the package directly. Configuration uses `LOCOMANILAB_*` variables. See [migration compatibility](docs/migration.md) for existing installations and generated data.
