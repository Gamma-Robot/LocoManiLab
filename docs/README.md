# Documentation

Start with the [CPU walkthrough](../README.md#start-on-cpu) or [中文快速开始](quickstart.zh-CN.md). The synthetic example needs no optional repositories.

| Guide | Scope |
|---|---|
| [Environment setup](environment_setup.md) | Python core, workspaces, selective source profiles |
| [Data collection and preparation](data_collection.md) | Dataset layout, portable preparation, collection prerequisites |
| [Models](models.md) | All adapters, fork boundaries, validation status |
| [Policy guides](../policies/README.md) | Per-policy workspace examples, CPU walkthroughs, installation and model commands |
| [Data, assets, and checkpoints](hf_resources.md) | Hugging Face downloads of benchmark data, simulator assets, released checkpoints; minimal test unit |
| [Training](training.md) | Local native launchers and model-specific prerequisites |
| [Simulation setup](simulation.md) | EmptyEnv first, selected controller weights, task-specific assets |
| [Simulation validation](sim_validation.md) | Separate simulator/controller runtime and closed-loop checks |
| [Deployment](deployment.md) | Local checkpoints, policy servers, complete robot dependencies |
| [Controller contracts](controller.md) | Variant identity and state/action compatibility |
| [Headset video](headset_video.md) | Optional PICO video prerequisites |
| [Domain randomization](domain_rand.md) | Reproducible evaluation settings |
| [Development](development.md) | Architecture and CPU/package validation |
| [Third parties](THIRD_PARTY.md) | Code, weights, datasets, and asset license boundaries |
| [Releasing](RELEASING.md) | Versioning, release gates, and publishing |

The model matrix distinguishes retained integrations from validation completed for this release. Historical laboratory records and private infrastructure instructions are not part of the public documentation.
