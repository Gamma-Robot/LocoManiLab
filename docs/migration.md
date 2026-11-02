# Naming and existing installations

Use `lab`, the `locomanilab` package, `LOCOMANILAB_*` integration variables,
and the pinned LocoManiLab forks for current workflows. New generated dataset
IDs and configuration names use `locomanilab/` or `locomanilab_`.

Existing user files are read in place; setup does not rename datasets,
checkpoints, workspaces, or repositories.

| Existing item | Compatibility |
|---|---|
| `soma` console command and setup selector | Alias for the same CLI; current instructions use `lab` |
| `SOMA_CONFIG`, `SOMA_WORKSPACE`, and runtime settings | Legacy fallback after the corresponding `LOCOMANILAB_*` setting |
| `SOMA_EGO_STREAM*` | Simulator fallback only when the new setting is absent; an explicitly empty new endpoint disables publishing. Collection exports both endpoint names for older user-owned simulator checkouts |
| pi05 `soma_dataset.json` | Read when the new `locomanilab_dataset.json` is absent; the new marker takes precedence |
| DM05 registration records | Training uses the name stored in `opendm_dataset.json`; the default `train50_merged` serving profile also uses that record |
| XR1 `soma_<dataset>.yaml` | Training reads it when the canonical generated config is absent |
| Old model converter paths | Deprecated forwarders in updated forks; adapters also recognize older user-owned checkouts |

Keep a converted dataset's registration record, statistics, and source videos
together. Existing checkpoints remain bound to the dataset/controller recipe
used during training. Explicit custom manifest values remain supported.

The robot camera launcher exposes `--zmq`. Its independently installed
OrinVideoSender SDK still receives its native `--soma-zmq` argument; changing
that external interface requires an SDK update. Upstream vocabulary entries,
copyright notices, and historical Git records retain their original text.
