# Architecture

LocoManiLab is a local orchestration and dataset layer for humanoid policy research. Its stable boundary is the shared dataset and policy-server contract. It does not combine model environments or implement their neural networks.

## Ownership

| Component | Responsibility |
|---|---|
| `src/locomanilab` | CLI, workspace configuration, dataset preparation/validation, command construction, process lifecycle, evaluation orchestration |
| Model adapters | Translate public workflow options into the selected repository's native commands |
| Model runtime helpers | Convert a dataset or expose a compatible server inside that model's environment |
| `g1_backend` | Independent controller/retargeting/recording runtime and hardware integration |
| `humanoid_benchmark` | Independent simulation, task definitions, and simulator execution |
| GR00T WholeBodyControl | Project-recommended native runtime for complete GR00T/FastWAM robot deployment |

All code remains in one installable `src/locomanilab` package. Adapter modules stay flat under `models`; model-specific runtime helpers can be grouped when needed. Avoid framework hierarchies that merely rename individual scripts.

`policies/<model-id>/` provides user guides and workspace examples. It does not
introduce another adapter registry, executable wrapper layer, or source checkout.
ACT and DP share the LeRobot adapter/runtime and a compatible, fixed 0.3.3 source
revision; their v2.1 data and normalization APIs must be reviewed before upgrading
to a newer LeRobot major data format.

## Execution and data

The main CLI imports only its own lightweight dependencies. It resolves a workspace and a model manifest, validates inputs, constructs a command, and delegates execution to the target environment. A model server and simulator may use different environments and communicate through the policy protocol. Runtime helpers are package resources so wheel installs work outside the source tree.

Simulation and real-robot collection share `g1_backend` for controllers, retargeting, and recording; `humanoid_benchmark` provides the simulation host. HeFT remains a controller option in that runtime without a standalone HeFT source dependency. Replay likewise depends on the controller/robot/simulator runtime, not the selected learned policy. Dataset preparation belongs to LocoManiLab and runs independently of model repositories. Controller variants determine state/action meaning; equal vector dimensions are insufficient evidence of compatibility. Keep source provenance, action ordering, frame counts, and episode boundaries explicit.

## Configuration and installation

New workspaces use configuration schema v2. Relative repository and data paths are based on the workspace configuration directory unless `workspace_root` explicitly overrides it. Legacy configurations retain their documented source-root behavior. Explicit repository paths take precedence over optional checkout conventions.

The current source layout has 14 optional submodules. The standalone HeFT research checkout was removed; its controller option and historical dataset compatibility remain in `g1_backend` and the data tools. Users initialize selected profiles; gitlink commits are the revision source of truth. A later optional-checkout manager must distinguish project-managed dependencies from user-owned paths before replacing this mechanism. Existing checkouts must not be treated as disposable caches.

## Support boundaries

The native GR00T WholeBodyControl pipeline is this project's recommended best practice for existing complete GR00T/FastWAM robot deployment. It owns that deployment's camera/control integration; it is not a collection entry point.

A retained adapter is not a fresh end-to-end validation claim. Track code presence, command/data contract checks, and actual GPU/hardware validation separately. The obsolete online RL launcher and manifest entries have been removed; new training integrations require an implemented, validated backend.

Fork-local registry, dataloader, or model modifications stay in the fork until a tested external extension point or upstream change replaces them. Standalone converters, compatibility servers, recipes, and checkpoint preparation tools can move into LocoManiLab selectively. See [model notes](docs/models.md).
