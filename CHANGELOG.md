# Changelog

## 1.0.0 — first public release

- `lab` CLI for the humanoid policy workflow: collect, prepare, convert, train, evaluate,
  and deploy, with a CPU sample-data walkthrough that needs no GPU, robot, or model repository.
- Benchmark task table of 234 tasks (`lab tasks`, `lab eval --task` / `--task-set`) with
  per-task evaluation settings and summaries.
- Policy guides and workspace examples for the integrated models; 14 pinned dependency
  repositories as optional git submodules, initialized per profile with `scripts/setup/profiles.py`.
- G1 simulation through `humanoid_benchmark` and shared whole-body controllers through
  `g1_backend` (SONIC, HEFT, AMO, ScaleBFM, ALMI).
- Released data, simulator asset manifest, GR00T/DM0.5/XR-1 checkpoints, and G1 controller
  weights on the Hugging Face Hub (`GammaRobotics/LocoManiLab`), with `scripts/hf/download.py`
  and the minimal download-and-test unit `scripts/hf/minimal_test.py`.
- MIT License (Gamma Robotics) for LocoManiLab's own code; dependencies, weights, data, and
  assets keep their own terms.
