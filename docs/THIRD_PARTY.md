# Third-party code and assets

This repository combines a small orchestration layer with optional external research implementations. A main-project license does not grant rights to every dependency, checkpoint, dataset, simulator asset, SDK, or generated scene. The main project's own code is licensed under the [MIT License](../LICENSE); that license covers only this repository's code and makes no blanket commercial-use claim for the optional integrations, weights, data, or assets described below.

## Code provenance

The 14 current submodule URLs live in `.gitmodules`; the superproject gitlinks identify the selected commits. `external/lerobot` (Hugging Face LeRobot) and `external/opendm` (Dexmal OpenDM) are upstream, unmodified. The other repositories are in the LocoManiLab organization: each keeps its upstream history plus one LocoManiLab commit, keeps its upstream LICENSE and starts its README with a "LocoManiLab modifications" section naming the upstream base commit and the changed files. `g1_backend` is LocoManiLab's own controller backend under the MIT License; its `THIRD_PARTY_NOTICES.md` lists the components it vendors and their terms. A public upstream URL does not establish that the fork's locked commit is available or equivalent.

| Group | Repositories |
|---|---|
| Model implementations | FastWAM, Isaac-GR00T, starVLA, PRTS, openpi, Psi0, lerobot, InternVLA-A1.5, dreamzero, Xiaomi-Robotics-1, opendm |
| Robot/controller/simulation | GR00T-WholeBodyControl, g1_backend, humanoid_benchmark |

The standalone HeFT checkout is no longer a source dependency, but HeFT-derived controller code and exported-policy support remain in `g1_backend`. Its HeFT provenance, and the original authorship and LICENSE/NOTICE obligations of vendored WBT and other imported components, must be retained. Removing a checkout does not remove attribution or change the terms of code, weights, or historical datasets.

Record each locked revision's upstream provenance, LICENSE/NOTICE files, fork modifications, and distribution method when releasing. If integration code is copied into the main package, preserve its attribution and applicable license instead of treating it as newly original code.

## Restricted integrations

Current upstream [PRTS](https://github.com/TeleHuman/PRTS#-license) states CC BY-NC 4.0 for its released code and weights. [InternVLA-A-series](https://github.com/InternRobotics/InternVLA-A-series#license-and-citation) states CC BY-NC-SA 4.0 for code. Verify the actual locked fork revision and any separately licensed components before distribution or use. Their inclusion as optional integrations does not make them unrestricted commercial dependencies.

## Weights, data, and scenes

Acquire base model weights from their original providers and follow their model cards and acceptance requirements. Do not include weights in a source archive. The fine-tuned checkpoints and benchmark data distributed through the project's Hugging Face repository ([data, assets, and checkpoints](hf_resources.md)) remain subject to the terms of the base models, datasets, and asset libraries they derive from; the dataset card lists those sources. Third-party simulator assets are not redistributed: `scripts/hf/download.py assets` fetches each file the benchmark uses from its original host (MolmoSpaces, RoboCasa, NVIDIA PhysicalAI Kitchen MJCF) under that host's terms, and the Hugging Face repository holds only an asset manifest and a few files listed with their origin and terms in `asset/NOTICE.md` (Gamma Robotics' own assets, a modified ArtVIP folder under Apache-2.0, and a corrected RoboCasa file under CC BY 4.0). Datasets require their own provenance and terms. Simulator assets may have separate download, attribution, or redistribution rules; a simulator source-code license is insufficient.

Generated or manually assembled scene assets must have their own provenance and redistribution review. Tasks requiring unavailable private assets are not a reproducible public quickstart. The synthetic CPU sample is generated locally and contains no released robot demonstrations.
