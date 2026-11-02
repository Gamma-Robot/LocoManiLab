# Data collection and preparation

## CPU first

Run the README's `sample-data → validate → prepare → validate` walkthrough before installing model runtimes. The sample is synthetic, deliberately small, and unsuitable as a robot benchmark.

The shared format is LeRobot-style v2.1 data: per-episode parquet under `data/chunk-*`, videos under `videos/`, and metadata under `meta/`. The metadata records tasks, feature layout, modalities, episode lengths, and controller/schema identity. Preserve the source dataset and prepare into a new output directory.

`validate` reads every episode and fully decodes declared videos to check frame counts. Large real datasets can take substantial time; this is a complete scan, not a fast metadata-only check. Validation is read-only. The small CPU sample keeps the first walkthrough quick.

```bash
lab validate datasets/example
lab prepare datasets/example --merge-only --out datasets/prepared
lab validate datasets/prepared
```

`--merge-only` is useful for the short sample because ordinary preparation applies trajectory cleaning. For real data, inspect `lab prepare --help` before choosing cleaning, episode filtering, or terminal-hold behavior. Video/parquet frame alignment and task text are part of the contract. Do not merge incompatible action signatures or controller variants merely because vector sizes match.

Preparation currently requires a positive integer FPS shared by the inputs. A valid fractional-rate dataset, such as 29.97 FPS, can be validated and retained in its original form, but `prepare` refuses it before writing output. It does not truncate FPS or resample videos. Use an explicitly reviewed resampling workflow before preparing such data.

## Collection

Collection requires independently provisioned runtimes. Both simulation and real-robot collection use `g1_backend` for controller selection, retargeting, and recording. `humanoid_benchmark` supplies the simulation host and tasks. Select the controller with `--backend sonic|heft|scalebfm|amo|almi`; HeFT remains a built-in backend using its exported policy and does not require a standalone HeFT training/research checkout.

Use `--sim` for simulation; omitting it selects real-robot collection. A model ID does not select the collection runtime. Full GR00T/FastWAM policy deployment has its own [recommended native WBC runtime](deployment.md#robot-runtimes), separate from this shared collection entry point.

Follow [minimal simulation setup](simulation.md) to install EmptyEnv and the selected controller without unrelated room packs. With those prerequisites already installed, inspect a collection plan:

```bash
lab collect --sim --env EmptyEnv --backend sonic --sonic-ckpt v1_1 --pico-source mock --task-prompt 'example task' --dry-run
# The same simulation host can select the HeFT controller:
lab collect --sim --env EmptyEnv --backend heft --pico-source mock --task-prompt 'example task' --dry-run
```

`EmptyEnv` avoids additional room assets, but still requires the G1 model and controller runtime/checkpoint. Real-headset and real-robot paths additionally require the appropriate services, SDK, camera, and controller configuration. Follow those repositories' instructions; core setup does not install or download them.

For automatic camera startup on a robot over SSH, configure
`deploy.orin_ssh_host` in your workspace's `env.local.yaml` and, for headset
listen mode, `deploy.orin_listen`. Use the robot's address on the headset network
for the latter. Without an SSH host, collection leaves camera startup to you and
prints a configuration hint. Set `deploy.camera_runtime: none` when another
process already manages the camera.

Historical HeFT datasets remain valid inputs to `validate` and compatible `prepare` workflows; they do not need the removed research source checkout. Preserve their existing modality/action layout rather than relabeling them as SONIC.

Each dataset should retain its collection manifest and controller checkpoint identity. Changing state/action semantics requires an explicit migration or recollection; old laboratory repair scripts are not a supported public migration API.
