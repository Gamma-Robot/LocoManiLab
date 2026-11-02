# Environment setup

## Core

Python 3.10+ and a normal virtual environment are sufficient for the core. Install `.[data]` for parquet/video data tools and `.[dev,data]` for development. See the README for the complete CPU walkthrough. Conda is not required for the core.

`bash scripts/setup.sh` only prints help. `bash scripts/setup.sh lab --plan` shows a core install plan. `bash scripts/setup.sh lab` explicitly creates the checkout's `.venv` and installs the data-enabled CLI, then prints its activation command. It cannot activate the calling shell.

## Workspace

Create a workspace with `lab init <directory>`. Keep `env.local.yaml`, datasets, outputs, and local checkpoint paths there. New schema-v2 configurations anchor relative `repos` and `paths` entries at the configuration directory unless `workspace_root` is specified. Existing configurations without `schema_version` keep their legacy path behavior; migrate deliberately rather than relocating active model environments.

An existing checkout can be selected with `repos.<key>`. Do not duplicate a provisioned model environment merely to fit an `external/` convention. Run `lab doctor` after configuring the resources you intend to use. Missing unrelated repositories do not mean the CPU data tools require them.

The default simulator source path is now `external/humanoid_benchmark`. An explicit `repos.sim_repo` path may still point to an existing checkout under its old name; it is not moved or deleted by setup. Use the new `LOCOMANILAB_CONFIG` and `LOCOMANILAB_WORKSPACE` variables for configuration selection. Their legacy `SOMA_CONFIG` and `SOMA_WORKSPACE` equivalents remain fallbacks when the new variables are unset.

## Optional repositories

From the source checkout:

```bash
python3 scripts/setup/profiles.py --list
python3 scripts/setup/profiles.py sim
python3 scripts/setup/profiles.py sim --apply
```

Only `--apply` initializes missing selected source checkouts. Existing checkouts are not changed. Git LFS smudging is disabled during this operation; no weights, SDKs, kitchen assets, or other binary resources are downloaded. The selected fork and commit must be accessible to you. An authentication failure does not justify silently substituting an incompatible upstream revision.

Provision model and runtime environments according to their pinned repositories. Check Python, CUDA, PyTorch/JAX, and native library compatibility per integration. The source profile is not an environment lock or an end-to-end install guarantee. Keep simulator, robot, and model environments separate unless a tested workflow explicitly requires otherwise.
