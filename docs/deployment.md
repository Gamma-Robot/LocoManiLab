# Deployment from local checkpoints

Deployment starts from a checkpoint already present on the machine. LocoManiLab does not fetch from a training cluster, upload data, configure tunnels, or distribute credentials. Model-specific base weights and any TensorRT engines must also be available locally.

A policy server and a complete robot deployment are different execution scopes. The server loads the model and implements the compatible policy protocol; the robot runtime owns cameras, controller execution, retargeting, and hardware connections.

After provisioning the selected environment and setting local paths, inspect a server plan:

```bash
lab deploy --model gr00t --model-path /path/to/checkpoint   --server-only --prompt 'the training instruction' --dry-run
```

Other adapters expose model-specific checkpoint options through `lab deploy --help`. A prompt, embodiment tag, state/action layout, and normalization configuration must agree with training. A responding socket alone is insufficient: check action keys, dimensions, finite values, and episode-reset behavior.

For pi0.5, pass `--config-name <matching-native-recipe>` to `deploy`, `eval`, and `infer-check`, using the recipe selected for conversion/training. The recipe determines the controller's input/output contract; selecting a checkpoint directory does not select its recipe automatically. Keep its dataset-bound normalization assets with the checkpoint.

## Robot runtimes

For existing complete GR00T/FastWAM deployment on a physical robot, this project's **recommended best practice** is the native **GR00T WholeBodyControl** runtime. Keep its full launcher, camera/control environment, local controller weights, and hardware connections together. A standalone policy compatibility server provides one part of that deployment and does not replace the native runtime.

The `gr00t-real` and `fastwam-real` [source profiles](../dependencies/README.md) select the relevant repositories. GR00T's normal `deploy` uses its native WBC launcher; `--server-only` selects just the Isaac-GR00T policy server. FastWAM's `deploy` uses the full WBC launcher, while simulation evaluation uses its separate policy-server hook. Follow each [policy guide](../policies/README.md) for the exact fixed source revision and model environment.

Simulation and real-robot **collection** both use `g1_backend`, independently of this complete-deployment recommendation. It retains the HeFT controller alongside SONIC, AMO, ScaleBFM, and ALMI; a standalone HeFT training repository is not required. See [collection](data_collection.md) for that entry point.

Configure only the runtime you actually provisioned. Verify controller checkpoint variant and recording/training action semantics before enabling a physical robot. Model integrations have not received fresh hardware validation during this source cleanup; follow the chosen robot runtime's documented operating procedure.

## Optional diagnostics

`scripts/diagnostics/check_gr00t_server.py` checks a compatible server's response structure; `xr1_roundtrip.py` checks the XR-1 contract. Run these only in their required environment, against a server and checkpoint you selected. They do not establish task success or robot readiness.
