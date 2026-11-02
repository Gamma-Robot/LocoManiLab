# Simulation validation

Closed-loop simulation uses two external codebases: `humanoid_benchmark` for the environment/tasks and `g1_backend` for controller/runtime behavior. A policy model runs in its own environment and serves actions through the compatible protocol.

```bash
# Source checkout: inspect first; --apply only initializes selected code.
python3 scripts/setup/profiles.py sim
```

Provision the simulator environment and the controller checkpoint separately. EmptyEnv needs no room asset pack, but is not free of robot/controller resources. Room tasks may require additional independently licensed assets. No setup helper downloads those assets automatically.

After these prerequisites and a local model checkpoint are ready:

```bash
lab eval --model gr00t --sim --model-path /path/to/checkpoint   --env EmptyEnv --episodes 1 --prompt 'the training instruction' --dry-run
```

All benchmark tasks run on the single `humanoid_benchmark` revision recorded by the gitlink. A few tasks were evaluated before a later simulator change to their reset; [`sim_task_overrides.yaml`](../src/locomanilab/runtime/sim_task_overrides.yaml) restores their evaluated configuration for the pinned scene, and the eval client prints each override it applies (`--no-task-overrides` disables them). With the overrides, the seeded initial states (`initial_state` in `--results-json`) match the evaluation records.

Check `lab eval --help` and the task's actual success definition before a scored run. Record task/scene, controller and model revisions, prompt, checkpoint, reset distribution, seed, action horizon, and inference rate. A video, a socket round-trip, and task success each provide different evidence.

`scripts/diagnostics/preflight_sim.py` is an optional visual sanity check. Its darkness heuristic is not a physical stability measurement and is not a task-success metric. The simulator/controller integration has not been revalidated end to end for this release.

The obsolete online RL launcher has been removed. Use the model-specific supervised training workflow and this separate evaluation process.

## Benchmark tasks

[`benchmark/tasks.yaml`](../src/locomanilab/benchmark/tasks.yaml) records the evaluation settings of every benchmark task: environment, scene, prompt, number of episodes, step budget, genuine-success threshold, SONIC variant, and the fixed-scene/reset-jitter seeds. Task names match the dataset directories. `lab tasks [selection]` lists them; `lab eval --task <name>` fills these options (options given on the command line win), and `--task-set` runs a selection one task after another:

```bash
lab eval --model gr00t --sim --model-path /path/to/checkpoint --task IThorKitchenOpenFridge_scene13
lab eval --model gr00t --sim --model-path /path/to/checkpoint --task-set atomic/seen
```

Selections combine a type (`atomic`, `composite`) and a split (`seen`, `unseen_task`, `unseen_scene`), e.g. `atomic/seen`; `all` selects every task. Each task writes its videos and `results.json` to `runs/eval/<model>/<task>/` (`--video-dir` sets the root); a rerun skips tasks whose results are complete, and `summary.json` in the root reports per-task, per-group and overall success rates. Each task needs its scene assets.
