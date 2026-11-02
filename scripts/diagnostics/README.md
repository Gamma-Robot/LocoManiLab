# Optional diagnostics

The policy guide checker runs in the main data-enabled environment. Other tools run in an already provisioned target environment. They do not install SDKs, acquire checkpoints, or certify task success.

| Script | Purpose |
|---|---|
| `check_gr00t_server.py` | Check compatible policy response structure and optional dataset contract |
| `xr1_roundtrip.py` | XR-1 policy round-trip with synthetic observations |
| `preflight_sim.py` | Visual simulation sanity check; darkness is a heuristic, not a stability metric |
| `pico_sdk_probe.py` | Inspect a connected PICO SDK service with competing collectors stopped |
| `check_policy_guides.py` | Replay marked README CPU examples in temporary workspaces; needs only the main data-enabled environment |

Use `--help` for the selected script. Keep addresses, ports, paths, and checkpoints explicit and local. Physical-robot operation remains subject to the selected runtime's operating procedure.
