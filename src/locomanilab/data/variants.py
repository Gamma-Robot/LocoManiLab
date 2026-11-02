"""Which whole-body controller produced a dataset — detected from the data itself.

locomanilab collects with three controllers and each records a different action/state space:

    sonic  state 43  action 78 = motion_token(64) + hands(14)          latent tokens
    heft   state 29  action 50 = ref_qpos(36) + hands(14)              tracking reference
    amo    state 32  action 36 = hands(14)+arms(14)+rpy(3)+height(1)   command stream
                                 + torso_vx/vy/vyaw + target_yaw

A model plugin therefore needs different settings per controller (embodiment tag, data
config, converter script, action/state dims...). Rather than making the user remember a
flag, each manifest declares a `variants:` table and the plugin asks this module which
one a dataset is:

    variants:
      amo:
        detect: { action: [torso_vx, target_yaw] }   # cheap discriminator
        require_modality:                            # full check once detected
          video:  [ego_view]
          action: [left_arm, right_arm, torso_vx, ...]
        <model-specific keys: embodiment_tag / script / overrides / dims ...>

`detect` matches on the SOURCE dataset's meta/modality.json group keys, which is the one
description of the action space every dataset carries regardless of model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from . import metadata as lerobot_meta
except ImportError:  # Direct file execution in a data environment, without locomanilab installed.
    import metadata as lerobot_meta


def action_signature(dataset: Path | str) -> frozenset[str] | None:
    """The dataset's action-space fingerprint: its meta/modality.json action keys.

    Model-independent, so it works for questions like "were these recorded through the
    same controller?" without knowing which model is going to consume them.
    """
    try:
        mod = lerobot_meta.load_modality(dataset)
    except Exception:  # noqa: BLE001  -- unreadable counts as "unknown" here
        return None
    if mod is None:
        return None
    return frozenset(lerobot_meta.modality_keys(mod, "action"))


def missing_modality(modality: dict | None, require: dict[str, list[str]]) -> list[str]:
    """`group.key` entries of `require` that `modality` does not declare."""
    return [f"{group}.{k}"
            for group, keys in (require or {}).items()
            for k in keys
            if k not in lerobot_meta.modality_keys(modality, group)]


def check_modality(dataset: Path | str, require: dict[str, list[str]], *,
                   what: str) -> str | None:
    """None if the dataset's modality.json declares every key in `require`; else why not.

    The same check every "no rewrite needed" convert() runs before telling the
    user to go train: a missing key here is a failure that would otherwise
    surface minutes into the model's own dataloader. `what` names the consumer
    in the message ("starVLA 'unitree_g1_sonic' config").
    """
    mj = lerobot_meta.meta_path(dataset, "modality.json")
    if not mj.exists():
        return (f"{mj} not found — not a GR00T/LeRobot dataset? "
                f"(run `lab prepare` first, or point --dataset at outputs_pre/<ds>)")
    try:
        mod = lerobot_meta.read_json(mj)
    except Exception as e:  # noqa: BLE001
        return f"cannot parse {mj}: {e}"
    missing = missing_modality(mod, require)
    if not missing:
        return None
    return ("dataset modality.json is missing keys required by "
            f"{what}: {', '.join(missing)}\n"
            f"present: action={sorted(lerobot_meta.modality_keys(mod, 'action'))}, "
            f"state={sorted(lerobot_meta.modality_keys(mod, 'state'))}, "
            f"video={sorted(lerobot_meta.modality_keys(mod, 'video'))}")


def same_controller(datasets: list[Path]) -> str | None:
    """None if every dataset shares an action space; else a message naming the groups.

    Merging across controllers is not a thing you can do: the merged meta describes one
    action space while the rows come from several. `lab prepare` refuses it, and so does
    the underlying prepare_dataset.py.
    """
    groups: dict[frozenset[str], list[Path]] = {}
    for ds in datasets:
        sig = action_signature(ds)
        if sig is not None:
            groups.setdefault(sig, []).append(ds)
    if len(groups) <= 1:
        return None
    lines = []
    for sig, paths in groups.items():
        lines.append(f"  action keys [{', '.join(sorted(sig)) or '(none)'}]:")
        lines.extend(f"      {p}" for p in paths)
    return ("these datasets come from different controllers and cannot be merged — the "
            "merged meta would describe one action space while the rows come from "
            "several:\n" + "\n".join(lines) + "\n  Merge each controller's datasets separately.")


def detect(dataset: Path | str, variants: dict[str, Any]) -> tuple[str, dict[str, Any]] | str:
    """Return (variant_name, spec), or an error string explaining why not.

    Returning the message instead of raising keeps the caller free to fall back
    (e.g. a model that only needs the variant for a nicer default).
    """
    if not variants:
        return "no `variants:` table in this model's manifest"
    mj = lerobot_meta.meta_path(dataset, "modality.json")
    if not mj.exists():
        return f"{mj} not found — not a LeRobot dataset locomanilab can read"
    try:
        mod = lerobot_meta.read_json(mj)
    except Exception as e:  # noqa: BLE001
        return f"cannot parse {mj}: {e}"

    for name, spec in variants.items():
        spec = spec or {}
        if missing_modality(mod, spec.get("detect") or {}):
            continue
        missing = missing_modality(mod, spec.get("require_modality") or {})
        if missing:
            return (f"dataset looks like the '{name}' schema but its modality.json is "
                    f"missing keys this model needs: {', '.join(missing)}")
        return name, spec
    return (f"cannot tell which controller this dataset came from — its modality.json "
            f"action keys {sorted(lerobot_meta.modality_keys(mod, 'action'))} match none of "
            f"{list(variants)}")


# --------------------------------------------------------------------------- #
# Collection lanes: controller + which of its checkpoints
# --------------------------------------------------------------------------- #
#
# `detect()` above answers "which controller", read off the action space. That
# is the right granularity for picking a model plugin's settings, and the wrong
# one for deciding what may be merged or share a directory, because one
# controller can ship several checkpoints whose recorded action is the same
# SHAPE and a different QUANTITY.
#
# sonic is the case. Its action is `motion_token`, a 64-D latent that only means
# anything to the decoder that produced it: low_latency and v1_1 both emit 64
# numbers into an identically-named column, and mixing them yields a dataset
# whose tokens address two different decoders. Nothing downstream can see it --
# not the modality.json, not the dtypes, not `action_signature` above. The
# symptom is a policy that trains cleanly and scores zero, which is exactly the
# 2026-08-06 incident locomanilab/common/sim.py's byte-size guard came out of.
#
# So datasets are filed and gated by *lane*: the backend name, suffixed with the
# checkpoint variant when the backend has more than one and this is not the
# default. Lanes are directory names (`outputs_sim/<lane>/...`) and the hard
# line in prepare's merge gate, so they must be stable and machine-independent
# -- hence derived from the checkpoint's position in the tree, never from the
# absolute path, which differs per machine for the same checkpoint.

#: The sonic checkpoint every dataset predating the v1_1 integration used.
#: Its lane stays plain "sonic" permanently so those datasets keep their
#: existing directory and never mix with v1_1 tokens. This is the lane baseline,
#: not the user-facing default for new collections.
SONIC_DEFAULT_VARIANT = "low_latency"

#: The checkpoint selected when a new ``lab collect --backend sonic`` run does
#: not spell either ``--sonic-ckpt`` or ``--policy-config``. The CLI resolves it
#: to an explicit manifest before choosing the output directory, so the run lands
#: in ``sonic_v1_1`` while the legacy ``sonic`` lane remains low_latency.
SONIC_COLLECT_DEFAULT_VARIANT = "v1_1"


def sonic_ckpt_variant(policy_config: str | None) -> str | None:
    """`low_latency` / `v1_1` / ... from a path under `checkpoints/sonic/`.

    None when the path is absent or does not sit in that tree -- an explicitly
    unknown answer, not a guess, because guessing here silently merges token
    spaces. Callers treat None as "the default variant" only for the lane name;
    the merge gate additionally reports what it saw.
    """
    if not policy_config:
        return None
    parts = str(policy_config).replace("\\", "/").split("/")
    for i in range(len(parts) - 2):
        if parts[i] == "checkpoints" and parts[i + 1] == "sonic":
            return parts[i + 2] or None
    return None


def collection_lane(backend: str | None, policy_config: str | None = None) -> str:
    """The directory name a dataset belongs under, and the unit prepare merges.

    Equals `backend` for every controller that ships one checkpoint, and for
    sonic's historical lane baseline -- so nothing that exists today moves.
    The collect CLI resolves its v1_1 default to an explicit policy path before
    calling this function.
    """
    name = str(backend or "unknown")
    if name != "sonic":
        return name
    variant = sonic_ckpt_variant(policy_config)
    if variant in (None, SONIC_DEFAULT_VARIANT):
        return name
    return f"{name}_{variant}"


# --------------------------------------------------------------------------- #
# `--sonic-ckpt`: a short name for a manifest that exists in the tree
# --------------------------------------------------------------------------- #
#
# `--policy-config` is the ground truth (it names the manifest), and the lane
# above is read off that path. Typing the absolute path of the same four
# manifests on every machine is the friction this shortcut removes: the name
# is resolved against g1_backend's checkpoint tree *and verified to exist
# there* before it is forwarded as an ordinary `--policy-config`, so nothing
# downstream can tell the two spellings apart and the alias can never point
# at a checkpoint the tree does not hold (the failure `lab eval`'s old
# `--sonic-ckpt` had -- see locomanilab/common/sim.py's client_extra_args).
#
# Aliases mirror g1_backend/g1_bridge/sonic_utils/ckpt.py so the same word
# means the same export on both sides of the shell-out.

#: user spelling -> directory under checkpoints/sonic/
SONIC_CKPT_ALIASES: dict[str, str] = {
    "low_latency": "low_latency",
    "low-latency": "low_latency",
    "new": "low_latency",
    "v1_1": "v1_1",
    "v1.1": "v1_1",
    "v11": "v1_1",
    "release": "release",
    "old": "release",
}

#: The two ONNX graphs every g1 sonic export is made of; the planner lives only
#: under low_latency and is variant-independent, so it is not checked per variant.
SONIC_EXPORT_FILES = ("model_encoder.onnx", "model_decoder.onnx")


def canonical_sonic_ckpt(name: str) -> str:
    """`low_latency` / `v1_1` / `release` from any accepted spelling.

    Raises ValueError naming the accepted spellings on anything else.
    """
    key = SONIC_CKPT_ALIASES.get(str(name).strip().lower())
    if key is None:
        raise ValueError(
            f"unknown sonic checkpoint {name!r}; one of: "
            + ", ".join(sorted(SONIC_CKPT_ALIASES)))
    return key


def sonic_policy_config(g1_backend_root: Path | str, name: str) -> Path:
    """Absolute path of `checkpoints/sonic/<variant>/g1/policy.yaml` for `name`.

    Resolves the alias and checks the manifest is really there; raises
    FileNotFoundError otherwise. Whether the ONNX weights next to it were
    downloaded is a separate question -- see `missing_sonic_weights`.
    """
    variant = canonical_sonic_ckpt(name)
    path = (Path(g1_backend_root).expanduser().resolve()
            / "checkpoints" / "sonic" / variant / "g1" / "policy.yaml")
    if not path.is_file():
        raise FileNotFoundError(
            f"sonic checkpoint {variant!r} has no manifest at {path} -- the "
            "g1_backend submodule tracks every shipped manifest in git, so either "
            "the submodule is stale (`git submodule update`) or this variant does "
            "not exist in this g1_backend")
    return path


def missing_sonic_weights(policy_config: Path | str) -> list[str]:
    """ONNX files absent next to a sonic manifest (empty when all present).

    The manifests travel with git, the weights do not; a manifest alone starts
    and then dies at load. Callers decide whether that is a warning or an error.
    """
    d = Path(policy_config).parent
    return [f for f in SONIC_EXPORT_FILES if not (d / f).is_file()]

# --------------------------------------------------------------------------- #
# Matching SONIC variants for GR00T-WBC real-robot deployment
# --------------------------------------------------------------------------- #
#
# GR00T-WholeBodyControl's deployment launchers select SONIC with the
# `--low-latency` preset; the other exports go through their
# generic `--deploy-checkpoint <prefix> --deploy-obs-config <yaml>` (the C++
# binary appends `_encoder.onnx` / `_decoder.onnx` to the prefix). The table
# pins where each export lives under <gr00t_wbc>/gear_sonic_deploy -- the
# layout `download_from_hf.py [--low-latency|--sonic-v1-1]` writes.

#: variant -> (deploy checkpoint prefix, observation config), relative to gear_sonic_deploy/
GR00T_WBC_DEPLOY_ASSETS: dict[str, tuple[str, str]] = {
    "release": ("policy/release/model", "policy/release/observation_config.yaml"),
    "low_latency": ("policy/low_latency/model", "policy/low_latency/observation_config.yaml"),
    "v1_1": ("policy/sonic_v1_1/model", "policy/sonic_v1_1/observation_config.yaml"),
}


def gr00t_wbc_deploy_files(gr00t_wbc_root: Path | str, variant: str) -> list[Path]:
    """The three files the deploy binary opens for `variant` (encoder, decoder, obs config)."""
    prefix, obs = GR00T_WBC_DEPLOY_ASSETS[variant]
    deploy = Path(gr00t_wbc_root) / "gear_sonic_deploy"
    return [deploy / f"{prefix}_encoder.onnx", deploy / f"{prefix}_decoder.onnx", deploy / obs]
