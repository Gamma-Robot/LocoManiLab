# Development guide

The architecture and ownership boundaries are in [DESIGN.md](../DESIGN.md). The canonical contribution commands are in [CONTRIBUTING.md](../CONTRIBUTING.md).

Develop the core without submodules. Use temporary workspaces and synthetic datasets for CPU checks; keep model integration checks opt-in. Test built wheels outside the repository so imports and resources cannot fall back to editable source files.

Model-specific runtime helpers may import that model inside its own process. The parent CLI must not import it. Resolve helpers as package resources and pass explicit checkpoint/dataset paths when execution changes the working directory.

When adding an adapter, pair its manifest with an adapter module, state the required fork/SHA and environment, document supported controller variants, and provide command-construction and protocol checks. Label actual model/GPU/hardware validation separately. Prefer a small adapter over copying an upstream trainer.

Move standalone converters, protocol wrappers, and portable recipes selectively from forks. Keep internal registry, data-loader, model-dimension, or training-correctness modifications in the fork until a verified extension hook or upstream patch replaces them. Preserve third-party notices. No automatic source mutation or unpinned upstream fallback belongs in an installation path.

The Python namespace is `locomanilab`; keep a single package, registry, and CLI implementation. Use `lab` in current commands and `locomanilab_` / `LOCOMANILAB_` for new artifacts and integration variables. Legacy reads and external SDK exceptions are documented in [migration compatibility](migration.md).

## Current entry points

Use the CLI for user workflows and the [maintainer tools](../scripts/README.md)
for setup, diagnostics, migration, and public-source checks. The old model forwarding
scripts and unused `common.prepare_dataset`, `common.infer_client`, and
`sim_eval.client` Python aliases have been removed. Data preparation lives in
`locomanilab.data.prepare`; child-process clients live in `locomanilab.runtime`.

The unavailable online RL launcher and its manifest configuration are retired.
The former `eval --pose/--latent` options had no effect and are removed; use the
model's `--action-space` and the selected controller's `--backend` and policy
configuration. Existing command aliases, workspace configuration, and historical
dataset compatibility remain supported.
