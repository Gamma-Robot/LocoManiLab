# Released data, assets, and checkpoints

The benchmark data, simulator assets, and released checkpoints are hosted in one
Hugging Face dataset repository,
[`GammaRobotics/LocoManiLab`](https://huggingface.co/datasets/GammaRobotics/LocoManiLab).
The source checkout contains none of them. `scripts/hf/download.py` fetches selected
parts; `scripts/hf/minimal_test.py` checks a fresh installation against them.

| Path in the repository | Contents | Size |
|---|---|---|
| `dataset/` | LeRobot v2.1 SONIC data: `main/` (234 benchmark tasks), `data_generation/`, `multi_controller/`, `real/`, `real2sim/` | 33 GB |
| `asset/` | Asset manifest and task map; third-party files are fetched from their original hosts, project-made files are in small shards | 18 MB |
| `checkpoint/gr00t` | GR00T N1.7 fine-tuned on the train50 split, step 10000 | 6.9 GB |
| `checkpoint/dm05` | DM0.5 fine-tuned on the train50 split, step 50000 | 23 GB |
| `checkpoint/xr1` | XR-1 fine-tuned on the train50 split, step 50000 (Lightning run directory) | 11 GB |
| `checkpoint/controller` | G1 whole-body controller weights (SONIC, HEFT, AMO, ScaleBFM, ALMI), mirroring `external/g1_backend/checkpoints/`; fetched per family with the simulator helper in [simulation setup](simulation.md) | 1.2 GB |

Checkpoints contain inference weights only; optimizer and RNG states were not released,
so they cannot resume training. The project's recorded data, the fine-tuned checkpoints, and
Gamma Robotics' own assets are released under the MIT License; third-party sources keep their own
terms: see [third-party boundaries](THIRD_PARTY.md) and the dataset card on the Hub.

## Install the download helper

From the checkout root, in the main CLI environment:

```bash
python -m pip install -e '.[data,hf]'
python scripts/hf/download.py list
```

`list` prints the asset files, checkpoint and controller-weight sizes, and dataset groups. Pass `--repo` (or set
`LOCOMANILAB_HF_REPO`) to use a mirror of the repository, and `--revision` to pin a Hub
revision. If the repository or a gated base model requires it, log in with `hf auth login`.

## Benchmark data

Download one task, a group, or a split. The local directory mirrors the repository layout:

```bash
python scripts/hf/download.py dataset --task IThorKitchenOpenFridge_scene13 --out datasets/hf
python scripts/hf/download.py dataset --group main/atomic/seen --out datasets/hf
python scripts/hf/download.py dataset --group data_generation/kimodo --task IThorKitchenOpenFridge_scene13 --out datasets/hf
lab validate datasets/hf/dataset/main/atomic/seen/IThorKitchenOpenFridge_scene13
```

`--group` defaults to `main`; `--task` is repeatable. Task directory names match
`lab tasks` and the `benchmark/tasks.yaml` table. Each task directory is a complete LeRobot
v2.1 dataset (`meta/`, `data/`, `videos/`) and works with `lab validate` and
`lab prepare --merge-only` unchanged. Twenty-two directories (`data_generation/kimodo/*`
and two under `real/`) have no `meta/stats.json`; validation and preparation do not need it.

## Simulator assets

Initialize the simulator source first (`python3 scripts/setup/profiles.py sim --apply`, see
[simulation setup](simulation.md)), then fetch the assets the selected tasks need:

```bash
python scripts/hf/download.py assets --task IThorKitchenOpenFridge_scene13 \
  --assets-dir external/humanoid_benchmark/robocasa/models/assets
python scripts/hf/download.py assets --task-set atomic/seen \
  --assets-dir external/humanoid_benchmark/robocasa/models/assets
python scripts/hf/download.py assets --preset all \
  --assets-dir external/humanoid_benchmark/robocasa/models/assets
```

The Hub repository does not mirror third-party asset libraries. `asset/manifest.tsv.gz` lists
every file the benchmark uses (path, sha256, size, source) and `asset/tasks.json.gz` maps each
task to its files. The helper downloads only those files, each from its original host, and
checks every file's sha256 before moving it into place:

| Source | What is fetched | Terms |
|---|---|---|
| [MolmoSpaces](https://huggingface.co/datasets/allenai/molmospaces) (pinned revision) | the iTHOR, ProcTHOR and Holodeck scenes, THOR objects, and Objaverse objects the tasks load, one package at a time | CC BY 4.0; Objaverse subset ODC-BY 1.0 with per-object licenses |
| [RoboCasa](https://github.com/robocasa/robocasa) asset archives | individual files (HTTP range requests, not whole archives): textures, the generative-texture randomization pool, and the object categories the tasks sample from | CC BY 4.0 |
| [NVIDIA PhysicalAI Kitchen MJCF](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF) (pinned revision) | Lightwheel fixtures and objects, as individual zip members | CC BY 4.0 |
| simulator checkout | files already tracked in `humanoid_benchmark` (TeleLab scenes and objects, layouts, task specs) | the simulator's terms |
| this repository, `asset/local.tar.gz` | Gamma Robotics' converted box, a folder converted from [ArtVIP](https://huggingface.co/datasets/X-Humanoid/ArtVIP), and one RoboCasa file with a corrected texture path | per file, see `asset/NOTICE.md` (box: MIT; ArtVIP: Apache-2.0; RoboCasa: CC BY 4.0) |

Two of the 47 Objaverse objects the tasks load are CC BY-NC 4.0: framed wall art
`596fd518055a49a28d56c193cd087470` and `f33af1b399a24448880930fef312ef3f` in Holodeck scene 117.
Their non-commercial terms apply wherever they are used or rendered.

Object categories are kept whole because the simulator samples instances from directory
listings; the generative-texture pool is kept whole for `--randomize` and replay augmentation.
Scene `*.robocasa.xml` files are generated by the simulator on first use and are not
downloaded. Files already present with the right checksum are skipped, so reruns resume;
`--force` refetches them. `--preset ithor|holodeck|telelab|all` selects task families,
`--task` and `--task-set` (the `lab tasks` selections) select individual tasks.

| Selection | Files | Size |
|---|---|---|
| `--task IThorKitchenOpenFridge_scene13` | 1,526 | 32 MB |
| `--preset all` (all 234 benchmark tasks, the dataset-only TeleLab tasks, randomization pool) | 22,745 | 3.4 GB |

## Checkpoints

```bash
python scripts/hf/download.py checkpoint gr00t --out checkpoints/hf
python scripts/hf/download.py checkpoint xr1 --no-weights --out checkpoints/hf   # configs only
```

Serve or evaluate a downloaded checkpoint with the commands in its policy guide, using
`checkpoints/hf/checkpoint/<name>` as `--model-path`:

| Checkpoint | Model environment and extra requirements |
|---|---|
| `gr00t` | [GR00T guide](../policies/gr00t/README.md). Loads the gated backbone `nvidia/Cosmos-Reason2-2B` at startup: accept its terms on the Hub and log in on the serving machine; the server contacts the Hub even when the backbone is cached. |
| `dm05` | [DM05 guide](../policies/dm05/README.md) (`opendm` environment). |
| `xr1` | [XR-1 guide](../policies/xr1/README.md) (`xr1` environment and its Qwen3-VL cache). The run directory has no `last.ckpt`; pass `--ckpt-step 50000`. |

For example, a benchmark-task evaluation with the released GR00T checkpoint:

```bash
lab --config workspaces/gr00t/env.local.yaml eval --model gr00t --sim \
  --model-path checkpoints/hf/checkpoint/gr00t --task IThorKitchenOpenFridge_scene13
```

The task needs its scene assets (the `ithor` preset covers iTHOR tasks) and the SONIC
`v1_1` controller weights from `checkpoint/controller`, fetched as described in [simulation setup](simulation.md).

## Minimal download-and-test unit

`scripts/hf/minimal_test.py` runs the smallest end-to-end checks of a new installation.
Stages are selected with `--stages`; each prints PASS or FAIL.

| Stage | Downloads | Checks | Needs |
|---|---|---|---|
| `cpu` (default) | 1 task, 31 MB | `lab validate`, `lab prepare --merge-only`, `lab validate` | main CLI with `data,hf` extras |
| `checkpoint` (default) | configs of all three checkpoints, < 50 MB | checkpoints are complete and portable (no machine-specific backbone path) | same |
| `assets` | the files of one iTHOR task, 32 MB | sha256 of every file, and with `--sim-python` a MuJoCo compile and 200 physics steps of the task scene | initialized simulator source; simulator environment for `--sim-python` |
| `gpu` | none (uses `cpu` stage data) | open-loop actions of a running GR00T server on recorded frames versus the recorded actions | GR00T environment and a server serving the released checkpoint |

```bash
python scripts/hf/minimal_test.py --work hf-minimal-test
python scripts/hf/minimal_test.py --work hf-minimal-test --stages assets \
  --assets-dir external/humanoid_benchmark/robocasa/models/assets \
  --sim-python "$(conda run -n tele_benchmark which python)"
python scripts/hf/minimal_test.py --work hf-minimal-test --stages gpu \
  --gr00t-python external/Isaac-GR00T/.venv/bin/python --server-port 5550
```

The default stages take a few minutes on a 7 MB/s connection; the asset stage takes about
six. Run the `gpu` stage after starting the server as shown in the GR00T guide. These stages
check downloads, data contracts, asset loading, and served actions; they are not training,
closed-loop task success, or hardware validation.
