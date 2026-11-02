# Controller and policy contracts

A learned policy, a whole-body controller, and the robot host have different responsibilities. The policy predicts an action chunk; the controller interprets that command; the host advances simulation or hardware. A model name does not select an interchangeable action space.

| Controller lane | Action meaning |
|---|---|
| `sonic` | Latent motion tokens plus hand commands; tied to the matching SONIC decoder |
| `heft`, `scalebfm` | Retargeted reference motion and hand targets |
| `amo` | Explicit arm/torso and locomotion command components |
| `almi` | Locomotion twist plus upper-body reference components |

All five lanes run through `g1_backend` during simulation and real-robot collection. The independent HeFT research checkout is no longer required: `--backend heft` selects the runtime's HeFT controller and exported policy. This does not remove HeFT model variants or change existing HeFT datasets.

A policy manifest's `heft` variant trains or serves that learned policy on
HeFT-recorded data. It does not install or train the retired standalone HeFT
controller repository. Kimodo's optional research rollout has its own actor,
tracker, and resource prerequisites; it is outside the normal collection setup.

The recorded modality/schema is authoritative for component names, ordering, widths, coordinate frames, and hand layout. Some historical schemas differ even for the same lane. Do not infer compatibility from equal dimensions or a shared `robot_type` string. Merge validation should reject incompatible action signatures.

For a compatible policy server, validate observation/action keys, per-chunk dimensions, finite numeric values, prompt and normalization parity, and `reset` semantics. Stateful policies such as DreamZero must clear per-episode frame/KV state. Checkpoint-specific controller variants must match the one used during collection.

Collection controllers, low-level tracking, retargeting, and hardware transport remain in `g1_backend`. For complete GR00T/FastWAM robot deployment, the [native WBC runtime](deployment.md#robot-runtimes) is the project's recommended best practice; this is a separate execution scope from collection. The model adapter should not copy those implementations.
