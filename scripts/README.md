# Maintainer tools

Use `lab` for collection, data preparation, conversion, training, serving, and
evaluation. Model-specific implementations live in the installable
`src/locomanilab/runtime/` package and run in the selected model environment.

| Tool | Purpose |
|---|---|
| `setup/profiles.py` | Preview or explicitly initialize selected source dependencies |
| `setup.sh` | Environment setup launcher; preview with `--plan` |
| [hf](hf/README.md) | Download released data, simulator assets, and checkpoints from the Hugging Face Hub; minimal test unit |
| [diagnostics](diagnostics/README.md) | Optional CPU, protocol, headset, and visual checks |
| [migrations](migrations/README.md) | Explicit historical dataset migrations |
| `release/check_public.py` | Check the current source snapshot for private material |

The old top-level model forwarding scripts have been removed. Adapters locate
their packaged helpers through `locomanilab.runtime.script_path()`; they do not
depend on a source checkout's `scripts/` directory. The
[DM05 guide](../policies/dm05/README.md) shows how to select its checkpoint-widening
helper for a separate interpreter.
