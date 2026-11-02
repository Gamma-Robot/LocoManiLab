# Optional headset video

PICO/headset video is an optional runtime feature. It requires a compatible headset application, reachable networking, the selected camera/video sender, and any runtime-specific SDK/service. Core installation does not provide these components.

Inspect `lab collect --help` for the current headset-video and source options. A desktop capture source requires a display and the intended window; a simulator stream requires a running compatible publisher. Keep addresses and service settings in local configuration, not committed recipes.

The generic `scripts/diagnostics/pico_sdk_probe.py` can inspect SDK input in an already provisioned environment. Stop competing collector clients first because the service may support only one consumer. This probe does not install an SDK, pair a headset, or validate the full robot pipeline.
