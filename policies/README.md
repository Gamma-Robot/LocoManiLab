# Policy guides

Choose one policy below. Each guide provides a local workspace example, a small
CPU walkthrough, native environment prerequisites, and commands for your own
robot data and checkpoints. Run source-checkout commands from the repository
root. Install the main CLI as described in the [CPU quickstart](../README.md#start-on-cpu)
before following a guide.

| Policy | Source profile | Guide |
|---|---|---|
| ACT | `act` | [act](act/README.md) |
| Diffusion Policy | `dp` | [dp](dp/README.md); shares the ACT/LeRobot environment and converter |
| pi0.5 | `pi05` | [pi05](pi05/README.md) |
| OpenDM / DM0.5 | `dm05` | [dm05](dm05/README.md) |
| GR00T | `gr00t` | [gr00t](gr00t/README.md) |
| FastWAM | `fastwam` | [fastwam](fastwam/README.md) |
| starVLA | `starvla` | [starvla](starvla/README.md) |
| PRTS | `prts` | [prts](prts/README.md) |
| Psi0 | `psi0` | [psi0](psi0/README.md) |
| InternVLA-A1.5 | `internvla_a15` | [internvla_a15](internvla_a15/README.md) |
| DreamZero | `dreamzero` | [dreamzero](dreamzero/README.md) |
| Xiaomi Robotics 1 | `xr1` | [xr1](xr1/README.md) |

## What the examples validate

The marked CPU examples create new, ignored `workspaces/<policy>/` directories,
load the provided YAML, generate tiny synthetic videos/parquet files, validate
them, and prepare a second dataset. They need no policy source checkout,
checkpoint, simulator, robot, or GPU. Existing workspaces are refused; keep your
local configuration and data rather than rerunning setup over them.

The synthetic dataset is a data-pipeline example. Its state/action dimensions,
camera size, and controller identity are not a training recipe. Model command
previews and schema checks are reported separately from model execution; no
training quality, learned-policy inference, simulation success, or hardware
validation is implied. See the [model matrix](../docs/models.md) for integration
boundaries and [controller contracts](../docs/controller.md) for action meaning.

To replay the exact marked README examples in a temporary copy of the tracked source:

```bash
python scripts/diagnostics/check_policy_guides.py
# Or select the original pilots:
python scripts/diagnostics/check_policy_guides.py act dp pi05
```

This requires the installed main package with its `data` extra and Git, but does
not initialize optional sources. Omitting policy IDs checks every available
guide. The temporary workspaces are removed after the check.

## Where configuration and code live

- `workspace.example.yaml` is a machine-configuration example, copied into a new
  workspace as `env.local.yaml`. Its paths assume the documented workspace
  location; edit `repos.*` if you keep an existing checkout elsewhere.
- With schema v2, `repos.*` and `paths.*` are relative to the workspace directory.
  Model `--dataset`, `--model-path`, and `--base-model` arguments are relative to
  the invoking shell; each guide states its working directory explicitly.
- The executable adapters, model manifests, and runtime helpers stay under
  `src/locomanilab/`. The guides link to them rather than maintaining another
  implementation or copying default training parameters.
- `external/` holds optional upstream/fork sources. The committed gitlink is the
  revision authority. Source profiles do not install model environments or
  obtain weights; review the explicit installation and asset steps first.

When adding a guide, keep source provenance, controller/data requirements,
installation, local weights, real-data commands, validation scope, and common
errors explicit. Add one marked CPU block following the existing guides, update
this index and the model matrix, and run the guide checker. Never put local
checkpoints, datasets, tokens, or infrastructure settings in a guide directory.
