# Model integrations

Twelve model IDs use eleven adapter modules; ACT and DP share one implementation. The table describes retained code and expected dependency boundaries, **not fresh end-to-end validation**. Core CPU data tests and command-construction tests do not establish training quality, deployment readiness, or model task success. Optional repository access, compatible environments, local base weights/checkpoints, and any required fork changes remain prerequisites.

| Model ID | Adapter in `src/locomanilab/models/` | External repository | Data / native training entry | Release status |
|---|---|---|---|---|
| [`act`](../policies/act/README.md) | `lerobot_baseline.py` | `lerobot` | Main-package flat-action converter; `src/lerobot/scripts/train.py` | Pinned LeRobot v0.3.3 integration; E2E not revalidated |
| [`dp`](../policies/dp/README.md) | `lerobot_baseline.py` | `lerobot` | Shares ACT converter; native diffusion policy | Pinned LeRobot v0.3.3 integration; E2E not revalidated |
| [`dm05`](../policies/dm05/README.md) | `dm05.py` | `opendm` | Main-package JSONL converter/experiment/registry; `script/dm05_launcher.sh` | Unmodified external implementation with local integration; E2E not revalidated |
| [`gr00t`](../policies/gr00t/README.md) | `gr00t.py` | `Isaac-GR00T` | Shared dataset; `gr00t/experiment/launch_finetune.py` | Pinned integration fork required; E2E not revalidated |
| [`fastwam`](../policies/fastwam/README.md) | `fastwam.py` | `FastWAM` | Native ActionDiT/text preprocessing; `scripts/train_zero1.sh` | Pinned recipes/server fork required; E2E not revalidated |
| [`starvla`](../policies/starvla/README.md) | `starvla.py` | `starVLA` | Shared dataset via registered loader; `starVLA/training/train_starvla.py` | Pinned registry/server/trainer integration; E2E not revalidated |
| [`prts`](../policies/prts/README.md) | `prts.py` | `PRTS` | SONIC dataset + statistics/config; `scripts/train.py` | Pinned SONIC server contract only; restricted upstream terms; E2E not revalidated |
| [`pi05`](../policies/pi05/README.md) | `pi05.py` | `openpi` | Registered data/config + norm stats; `scripts/train.py` (JAX) | Pinned G1 integration; alternate PyTorch base path not established; E2E not revalidated |
| [`psi0`](../policies/psi0/README.md) | `psi0.py` | `Psi0` | Psi0 layout converter + statistics; `scripts/train.py` | Pinned converters/server and variant trainers; E2E not revalidated |
| [`internvla_a15`](../policies/internvla_a15/README.md) | `internvla_a15.py` | `InternVLA-A1.5` | v2.1 → v3.0 conversion; `src/lerobot/scripts/lerobot_train.py` | Pinned schema/model preprocessing integration; restricted upstream terms; E2E not revalidated |
| [`dreamzero`](../policies/dreamzero/README.md) | `dreamzero.py` | [`dreamzero`](../policies/dreamzero/README.md) | Prepared GEAR metadata; `groot/vla/experiment/experiment.py` | Pinned embodiment/transform/server integration; E2E not revalidated |
| [`xr1`](../policies/xr1/README.md) | `xr1.py` | `Xiaomi-Robotics-1` | JSON episodes + statistics; `xr1/scripts/train.sh` | Pinned dataset/model-dimension/server integration; E2E not revalidated |

Initialize one policy source profile with `python3 scripts/setup/profiles.py <model-id> --apply` from the source checkout. Add the `sim` profile for simulation evaluation. For complete GR00T/FastWAM robot deployment, the native WholeBodyControl runtime is this project's recommended best practice; use their `*-real` source profiles. Simulation and real-robot collection share `g1_backend`, which includes the HeFT controller without a standalone HeFT source dependency. These are source selections, not environment installers.

## Package runtime helpers

ACT/DP conversion and serving live under `src/locomanilab/runtime/lerobot/`. DM05 conversion, experiment, dataset registry, checkpoint widening, and server live under `src/locomanilab/runtime/dm05/`. They execute in their corresponding model environment. Their model implementations remain external.

Other adapters currently call integration files in their pinned forks. A compatible server wrapper is a reasonable candidate to move into the main package, but internal model/registry changes cannot disappear at the same time:

| Integration | Candidate for a separate main-package helper | Retain in the fork until a verified extension/upstream change exists |
|---|---|---|
| FastWAM | `deployment/run_fastwam_server.py`; portable G1 data/task recipes | Native training/backbone code; recipe registration/search behavior |
| GR00T | G1 modality definitions through `--modality-config-path` | Internal embodiment/tag references and native model/server implementation |
| starVLA | `deployment/model_server/run_starvla_server.py` | G1 data config/embodiment/mixture registration and trainer-correctness fixes |
| PRTS | `deployment/run_prts_server.py`; post-train YAML inputs | `prts/data/embodiment_tag.py` registration and native model/statistics code |
| openpi | `deployment/run_openpi_server.py`; G1 input/output transforms and recipes | Internal TrainConfig/DataConfig registration until externally loadable |
| Psi0 | `scripts/data/locomanilab_*_to_psi_lerobot.py`; `src/psi/deploy/run_psi0_server.py` | Variant trainer registration and native normalization/model behavior |
| InternVLA | Converter, ingestion check, widening tool, compatibility server | Package-scanned embodiment schema, processor/tokenizer and configuration fixes |
| DreamZero | GEAR converter wrapper; stateful compatibility server | Embodiment IDs, collate branches, data registries, action/state-dimension propagation |
| XR-1 | Converter, statistics/widening tools, compatibility server | Sonic dataset/datamodule and model shape parameterization |

Check the actual fork file contents and licenses before moving any of these. Preserve normalization, prompt construction, episode reset, camera handling, and action ordering. For example, DreamZero has per-episode frame/KV state; XR-1 and InternVLA need compatible widened checkpoints. Moving a file is not proof of behavioral equivalence.

## Controller variants and limits

The current manifests describe GR00T sonic/heft/amo/almi lanes; FastWAM/starVLA/pi05/Psi0 sonic/heft/amo lanes; DreamZero sonic/amo lanes; and sonic-focused XR-1/InternVLA/DM05 integrations. ACT/DP defaults use the sonic flat-action contract. PRTS currently accepts the verified SONIC absolute state[43]/action[78] contract; its pinned server cannot decode HeFT/AMO actions, so those datasets are refused before conversion/training. pi05's pinned AMO recipe also has an incompatible action-array/two-stream transform: its normalization and training are refused before launching the backend. The AMO registration alone is not a usable training path. A declared variant still requires the corresponding fork implementation and correctly recorded dataset schema. Read the selected manifest and inspect `--dry-run` rather than assuming every model supports every controller.

DreamZero serving through this adapter is fixed to the SONIC embodiment; its AMO training-side declaration is not a supported AMO deployment path. DM05 serving uses a fixed native dataset registry name, described in its policy guide.

No accuracy numbers or private evaluation checkpoints are included. The obsolete online RL launcher and its configuration have been removed. See [third-party terms](THIRD_PARTY.md) before using or redistributing an integration.
