#!/usr/bin/env python3
"""Minimal download-and-test unit for the Hugging Face release.

Stages (run in order; select with --stages, default ``cpu,checkpoint``):

  cpu         download one benchmark task (~31 MB) and run `lab validate`,
              `lab prepare --merge-only`, `lab validate` on it.
  checkpoint  download the released checkpoints' configs (no weights) and check that they
              are portable (no machine-specific backbone path) and complete.
  assets      fetch the asset files of one iTHOR task (~32 MB, sha256-verified) into
              --assets-dir; with --sim-python, also compile and step the task scene in MuJoCo.
  gpu         run scripts/hf/gr00t_openloop.py against a GR00T server already serving the
              released checkpoint (see policies/gr00t/README.md), using --gr00t-python.

Run from the LocoManiLab checkout root with the main package installed with the `data` and
`hf` extras:

    python scripts/hf/minimal_test.py --work hf-minimal-test
    python scripts/hf/minimal_test.py --work hf-minimal-test --stages assets \\
        --assets-dir external/humanoid_benchmark/robocasa/models/assets \\
        --sim-python "$(conda run -n tele_benchmark which python)"

Every stage prints PASS or FAIL; the exit code is non-zero when any selected stage fails.
These checks cover downloads, data contracts, and asset loading; they are not training,
closed-loop evaluation, or hardware validation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download  # noqa: E402

TASK = "IThorKitchenOpenFridge_scene13"
TASK_GROUP = "main/atomic/seen"
SCENE_XML = "scenes/ithor/FloorPlan13_physics.xml"
SCENE_CHECK = """
import sys, mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(sys.argv[1])
d = mujoco.MjData(m)
for _ in range(200):
    mujoco.mj_step(m, d)
assert not np.isnan(d.qpos).any(), "NaN after 200 steps"
print(f"bodies={m.nbody} geoms={m.ngeom} meshes={m.nmesh} textures={m.ntex}")
"""


def run(cmd, **kw) -> bool:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw).returncode == 0


def lab(*args) -> list:
    return [sys.executable, "-m", "locomanilab", *args]


def stage_cpu(a, work: Path) -> bool:
    data = work / "data"
    if download.main([*hub_args(a), "dataset", "--group", TASK_GROUP, "--task", TASK, "--out", str(data)]) != 0:
        return False
    task = data / "dataset" / TASK_GROUP / TASK
    prepared = work / "prepared"
    return (run(lab("validate", task))
            and run(lab("prepare", task, "--merge-only", "--out", prepared), cwd=work)
            and run(lab("validate", prepared)))


def stage_checkpoint(a, work: Path) -> bool:
    out = work / "checkpoints"
    ok = True
    for name in download.CHECKPOINTS:
        if download.main([*hub_args(a), "checkpoint", name, "--no-weights", "--out", str(out)]) != 0:
            return False
    root = out / "checkpoint"
    gr00t = json.loads((root / "gr00t/config.json").read_text())
    backbone = gr00t.get("model_name", "")
    if os.path.isabs(backbone):
        print(f"FAIL gr00t/config.json model_name is a machine path: {backbone!r} "
              "(expected the Hub id nvidia/Cosmos-Reason2-2B)")
        ok = False
    processor = json.loads((root / "gr00t/processor/processor_config.json").read_text())
    processor_backbone = processor.get("processor_kwargs", {}).get("model_name", "")
    if os.path.isabs(processor_backbone):
        print(f"FAIL gr00t/processor/processor_config.json model_name is a machine path: {processor_backbone!r}")
        ok = False
    required = {"gr00t": ["model.safetensors.index.json", "processor/statistics.json", "experiment_cfg/config.yaml"],
                "dm05": ["config.json", "norm_stats.json", "tokenizer.json"],
                "xr1": ["config.py", "config.yaml"]}
    for name, files in required.items():
        missing = [f for f in files if not (root / name / f).is_file()]
        if missing:
            print(f"FAIL {name}: missing {missing}")
            ok = False
    if not list((root / "xr1").glob("epoch=*-step=*.ckpt")):
        print("FAIL xr1: no epoch=*-step=*.ckpt run directory")
        ok = False
    return ok


def hub_args(a) -> list[str]:
    return ["--repo", a.repo] + (["--revision", a.revision] if a.revision else [])


def stage_assets(a, work: Path) -> bool:
    if not a.assets_dir:
        print("FAIL --assets-dir is required for the assets stage")
        return False
    if download.main([*hub_args(a), "assets", "--task", TASK, "--assets-dir", a.assets_dir]) != 0:
        return False
    scene = Path(a.assets_dir) / SCENE_XML
    if not scene.is_file():
        print(f"FAIL {scene} missing after the download")
        return False
    if not a.sim_python:
        print("(no --sim-python: skipped compiling the scene in MuJoCo)")
        return True
    return run([a.sim_python, "-c", SCENE_CHECK, scene])


def stage_gpu(a, work: Path) -> bool:
    if not a.gr00t_python:
        print("FAIL --gr00t-python is required for the gpu stage")
        return False
    task = work / "data" / "dataset" / TASK_GROUP / TASK
    if not task.is_dir():
        print("FAIL run the cpu stage first (it downloads the task data)")
        return False
    return run([a.gr00t_python, HERE / "gr00t_openloop.py", "--dataset", task, "--port", a.server_port])


STAGES = {"cpu": stage_cpu, "checkpoint": stage_checkpoint, "assets": stage_assets, "gpu": stage_gpu}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="hf-minimal-test", help="working directory for downloads and outputs")
    ap.add_argument("--stages", default="cpu,checkpoint", help=f"comma-separated: {','.join(STAGES)}")
    ap.add_argument("--repo", default=os.environ.get("LOCOMANILAB_HF_REPO") or download.DEFAULT_REPO)
    ap.add_argument("--revision", default=None, help="branch, tag, or commit of the Hub repo")
    ap.add_argument("--assets-dir", help="assets stage: <humanoid_benchmark>/robocasa/models/assets")
    ap.add_argument("--sim-python", help="assets stage: python of the simulator environment (has mujoco)")
    ap.add_argument("--gr00t-python", help="gpu stage: python of the Isaac-GR00T environment")
    ap.add_argument("--server-port", type=int, default=5550, help="gpu stage: port of the running GR00T server")
    a = ap.parse_args()

    selected = [s.strip() for s in a.stages.split(",") if s.strip()]
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        ap.error(f"unknown stages {unknown}")
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in selected:
        print(f"\n=== stage {name} ===", flush=True)
        results[name] = STAGES[name](a, work)
        print(f"=== {name}: {'PASS' if results[name] else 'FAIL'} ===", flush=True)
    print("\nsummary: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
