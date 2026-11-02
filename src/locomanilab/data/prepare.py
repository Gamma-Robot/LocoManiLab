"""Preprocess / merge collected LeRobot datasets — locomanilab's own, controller-agnostic.

This is the implementation behind `lab prepare`. It used to live in the sonic repo
(`GR00T-WholeBodyControl/gear_sonic/scripts/prepare_dataset.py`), which meant every
controller's data was cleaned by sonic's script: the hand-labelling stage sliced
`observation.state[22:29]` / `[36:43]` — sonic's 43-dim legacy layout — out of whatever
it was given, and an almi dataset (29-dim state, blob schema) died with
`IndexError: index 0 is out of bounds for axis 1 with size 0`.

What this one does differently:

* **No controller's layout is hardcoded.** Every optional stage asks the *dataset's
  own* meta/modality.json whether it applies (the one constant below, `JOINT_MAX_ABS`,
  is the Dex3 hand's range -- the same hand on every backend, not a controller). The hand stages read the `left_hand` /
  `right_hand` STATE groups (sonic declares them over observation.state; amo/almi/heft
  carry hands as an ACTION blob and no state group, so the stage skips itself). SMPL
  cleaning looks for whichever SMPL column is present — `teleop.smpl_joints` for
  bridge-recorded sonic, `teleop.smpl_pose` for the legacy gr00t_wbc exporter.
* **Merging is gated on `collection_info.backend`** (written by g1_bridge's recorder and
  by the sim collector). Datasets from different backends are refused; other
  collection_info differences (pico_source, policy_config, amo_arm_source, ...) are
  reported and carried into the output as `mixed_across_sources`, never silently
  inherited from the first input. Datasets that predate collection_info skip the check.
* **No foreign fields.** The merged output carries the INTERSECTION of the inputs'
  columns (what one source has and another doesn't is dropped and named on stdout), and
  a derived column is produced only if *every* input can produce it. The meta declares
  exactly the columns that were written — no more, no less.

Runs under the configured data interpreter with pandas + pyarrow + numpy. PyAV
is needed when a video must be re-encoded. Only stdlib helper siblings are imported;
the model environments need not install the orchestration CLI.

Usage:
    python locomanilab/data/prepare.py --dataset-paths outputs/a outputs/b \\
        --output-path outputs_pre/merged
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# This script is launched by absolute path in the configured data environment.
# The helpers below are stdlib-only siblings; do not expose the CLI environment's
# site-packages on PYTHONPATH just to locate locomanilab.
if __package__:
    from .variants import collection_lane
    from .paths import dataset_file, output_directory, staged_output
else:
    from variants import collection_lane
    from paths import dataset_file, output_directory, staged_output

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: SMPL reference columns, best first. `teleop.smpl_joints` (72) is what g1_bridge's
#: sonic backend records today; `teleop.smpl_pose` (63) is the legacy gr00t_wbc
#: exporter's name. A dataset with neither simply doesn't get the stale-frame stage.
SMPL_COLUMNS = ("teleop.smpl_joints", "teleop.smpl_pose")

#: Hand-joint STATE groups, as named in meta/modality.json. Their slice bounds come from
#: the dataset, not from here — that is the whole point.
HAND_STATE_GROUPS = {"left": "left_hand", "right": "right_hand"}

#: Per-side teleop joint columns the smoothing stage reads, and the labels written.
HAND_JOINT_COLUMNS = {"left": "teleop.left_hand_joints", "right": "teleop.right_hand_joints"}
OPEN_COLUMNS = {"left": "teleop.left_hand_open", "right": "teleop.right_hand_open"}
SMOOTH_SUFFIX = "_smooth"

#: Finger flexion joints within a hand group, and their physical travel. Dex3 hardware:
#: index 0-3 are the flexion DoFs of the first two fingers, the rest are abduction.
FLEXION_JOINTS = [0, 1, 2, 3]
JOINT_MAX_ABS = np.array([1.6, 1.74, 1.6, 1.74], dtype=np.float32)
DEFAULT_CLOSURE_THRESHOLD = 0.2
DEFAULT_WINDOW = 31

#: Column names that changed in the collector, old -> new. Consumers that classify
#: columns by the LeRobot constants (PRTS matches OBS_STATE = "observation.state") see NO
#: state at all under a non-standard name, and report it as an empty-tensor error rather
#: than an unknown column. amo wrote "states" until 2026-08-07; renaming here means
#: re-running prepare migrates a legacy dataset instead of it having to be re-collected.
#: Delete this table once no pre-2026-08-07 amo data is still in use.
LEGACY_COLUMN_RENAMES = {"states": "observation.state"}

#: GR00T's modality.json convention: a group without `original_key` belongs to the
#: section's default feature.
MODALITY_DEFAULT_FEATURE = {"state": "observation.state", "action": "action"}

DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/episode_{episode_index:06d}.mp4"


class PrepareError(SystemExit):
    """Refusal with an explanation — always a message a human can act on."""

    def __init__(self, message: str):
        super().__init__("\nERROR: " + message)


# --------------------------------------------------------------------------- #
# On-disk helpers
# --------------------------------------------------------------------------- #

def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parquet_path(root: Path, info: dict, episode_index: int) -> Path:
    pattern = info.get("data_path", DEFAULT_DATA_PATH)
    chunk = episode_index // info.get("chunks_size", 1000)
    return dataset_file(root, pattern, episode_chunk=chunk, episode_index=episode_index)


def video_keys_of(info: dict) -> list[str]:
    keys = info.get("video_keys") or []
    if not keys:
        keys = [k for k, v in (info.get("features") or {}).items()
                if (v or {}).get("dtype") in ("video", "image")]
    return keys


def video_paths(root: Path, info: dict, episode_index: int) -> dict[str, Path]:
    pattern = info.get("video_path", DEFAULT_VIDEO_PATH)
    chunk = episode_index // info.get("chunks_size", 1000)
    return {
        key: dataset_file(root, pattern, video_key=key, episode_index=episode_index,
                          episode_chunk=chunk)
        for key in video_keys_of(info)
    }


def parquet_columns(path: Path) -> set[str]:
    """Column names of a parquet file, without reading any row group."""
    import pyarrow.parquet as pq

    return {LEGACY_COLUMN_RENAMES.get(c, c) for c in pq.ParquetFile(path).schema_arrow.names}


def group_feature(section: str, spec: dict) -> str:
    """Which LeRobot column a modality group reads from."""
    return (spec or {}).get("original_key") or MODALITY_DEFAULT_FEATURE.get(section, section)


# --------------------------------------------------------------------------- #
# Source datasets
# --------------------------------------------------------------------------- #

@dataclass
class Source:
    """One input dataset: its meta, its columns, and what stages it supports."""

    path: Path
    info: dict
    modality: dict | None
    episodes_meta: list[dict]
    tasks_meta: list[dict]
    columns: set[str]
    task_map: dict[int, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Source":
        info = load_json(path / "meta" / "info.json")
        if info is None:
            raise PrepareError(f"not a LeRobot dataset (no meta/info.json): {path}")
        modality = load_json(path / "meta" / "modality.json")
        episodes_meta = load_jsonl(path / "meta" / "episodes.jsonl")
        migrate_legacy_meta(info, modality)

        columns: set[str] = set()
        seen_parquets: set[Path] = set()
        # Validate all resource paths before any destination is created. Missing
        # videos must not silently turn a declared multimodal dataset into data-only.
        dataset_file(path, info.get("data_path", DEFAULT_DATA_PATH), episode_index=0)
        if video_keys_of(info):
            dataset_file(path, info.get("video_path", DEFAULT_VIDEO_PATH), episode_index=0,
                         video_key=video_keys_of(info)[0])
        for ep in episodes_meta:
            pq_path = parquet_path(path, info, ep["episode_index"])
            if pq_path in seen_parquets:
                raise PrepareError(f"multiple source episodes resolve to the same parquet: {pq_path}")
            seen_parquets.add(pq_path)
            if not pq_path.is_file():
                raise PrepareError(f"missing episode parquet: {pq_path}")
            if not columns:
                columns = parquet_columns(pq_path)
            for key, video in video_paths(path, info, ep["episode_index"]).items():
                if not video.is_file():
                    raise PrepareError(f"episode {ep['episode_index']} missing declared {key} video: {video}")
                import av
                with av.open(str(video)) as container:
                    if len(container.streams.video) != 1:
                        raise PrepareError(f"source must contain one video stream: {video}")
                    stream = container.streams.video[0]
                    if stream.frames and stream.frames != ep.get("length"):
                        raise PrepareError(f"source video frames={stream.frames} disagree with "
                                           f"episode length={ep.get('length')}: {video}")
        return cls(path=path, info=info, modality=modality, episodes_meta=episodes_meta,
                   tasks_meta=load_jsonl(path / "meta" / "tasks.jsonl"), columns=columns)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def collection_info(self) -> dict | None:
        """What produced this dataset. `collection_info` is g1_bridge's recorder and the
        sim collector; `data_collection_info` is the legacy gr00t_wbc exporter's own,
        differently-keyed record — recognised so its datasets report as "unlabelled"
        rather than being compared against keys that never meant the same thing."""
        ci = self.info.get("collection_info")
        return ci if isinstance(ci, dict) else None

    @property
    def backend(self) -> str | None:
        return (self.collection_info or {}).get("backend")

    def hand_state_slices(self) -> dict[str, tuple[str, int, int]] | None:
        """{side: (column, start, end)} for the hand joints, per this dataset's modality.

        None when the dataset has no hand STATE groups. Which datasets those are moved
        under this function: before g1_bridge schema_version 5 only sonic and amo
        declared measured fingers in observation.state, and heft/scalebfm/almi carried
        the hands as an ACTION blob alone (what the operator commanded, not what the
        fingers did). v5 gives all five the STATE groups, so this stage — and the
        open/close labels it feeds — now applies to a v5 tracking dataset and still
        self-disables on the v4 ones already collected.

        Nothing here needed changing for that: the group names are read from the
        dataset's own modality.json rather than assumed, which is the whole point of
        driving these stages off the metadata.
        """
        state = (self.modality or {}).get("state") or {}
        out: dict[str, tuple[str, int, int]] = {}
        for side, group in HAND_STATE_GROUPS.items():
            spec = state.get(group)
            if not spec:
                return None
            column = group_feature("state", spec)
            start, end = int(spec.get("start", 0)), int(spec.get("end", 0))
            if column not in self.columns or end - start <= max(FLEXION_JOINTS):
                return None
            out[side] = (column, start, end)
        return out

    def smpl_column(self) -> str | None:
        for column in SMPL_COLUMNS:
            if column in self.columns:
                return column
        return None

    def smooth_columns(self) -> list[str]:
        """Teleop hand-joint columns present here (the raw signals worth smoothing)."""
        return [c for c in HAND_JOINT_COLUMNS.values() if c in self.columns]


def migrate_legacy_meta(info: dict, modality: dict | None) -> None:
    """Apply LEGACY_COLUMN_RENAMES to a dataset's meta, in place.

    Both halves matter: the feature table names the column, and each modality group's
    `original_key` points at it. Renaming only the dataframe would leave the meta
    pointing at a column that no longer exists.
    """
    features = info.get("features") or {}
    for old, new in LEGACY_COLUMN_RENAMES.items():
        if old in features:
            features[new] = features.pop(old)
    for section, groups in (modality or {}).items():
        default = MODALITY_DEFAULT_FEATURE.get(section)
        for spec in (groups or {}).values():
            origin = (spec or {}).get("original_key")
            if origin in LEGACY_COLUMN_RENAMES:
                new = LEGACY_COLUMN_RENAMES[origin]
                # the collector omits original_key when it equals the section default
                if new == default:
                    spec.pop("original_key", None)
                else:
                    spec["original_key"] = new


# --------------------------------------------------------------------------- #
# Merge gates
# --------------------------------------------------------------------------- #

def gate_backend(sources: list[Source]) -> tuple[dict, dict[str, list]]:
    """Refuse to merge across controllers; report every other collection_info difference.

    Returns (shared collection_info, {differing key: [values]}).

    The hard line is the *lane* — the backend plus, where the backend ships more than
    one incompatible checkpoint, which one (`locomanilab.data.variants.collection_lane`).
    Each whole-body controller records a different action (sonic: motion_token+hands,
    heft: ref_qpos+hands, amo/almi: the teleop command stream), so a merged dataset
    would contain all of them while its meta describes one.

    Backend alone is not enough: sonic's two checkpoints write the same-shaped
    `action.motion_token` with different meanings, and gating on `backend` would merge
    them into a dataset that trains fine and scores zero (the 2026-08-06 incident --
    the full story is at `locomanilab.data.variants.collection_lane`). Lanes refuse that.

    The remaining keys were chosen because they change what the recorded *values* mean
    without changing the schema (amo_arm_source="gmr" vs "ik" is the motivating one:
    same columns, systematically different arm behaviour) — those are reported, and the
    mixture is written into the output's meta so nothing downstream has to guess.
    """
    labelled = [s for s in sources if s.collection_info is not None]
    unlabelled = [s for s in sources if s.collection_info is None]
    for s in unlabelled:
        legacy = "yes" if isinstance(s.info.get("data_collection_info"), dict) else "no"
        print(f"  {s.name}: no collection_info (legacy gr00t_wbc record: {legacy}) "
              "— skipping the controller check for it")
    if not labelled:
        print("  no dataset records collection_info — controller check skipped entirely")
        return {}, {}

    by_lane: dict[str, list[str]] = {}
    for s in labelled:
        lane = collection_lane(s.backend, (s.collection_info or {}).get("policy_config"))
        by_lane.setdefault(lane, []).append(s.name)
    if len(by_lane) > 1:
        groups = "\n".join(f"    {lane}: {', '.join(names)}"
                           for lane, names in by_lane.items())
        raise PrepareError(
            "these datasets were collected through different controllers and cannot be "
            "merged — their action spaces are not the same quantity, so a merged dataset "
            "would describe one and contain all:\n" + groups +
            "\n  Merge each lane's datasets separately.")

    shared: dict = {}
    mixed: dict[str, list] = {}
    for key in sorted({k for s in labelled for k in s.collection_info}):
        values = [s.collection_info.get(key) for s in labelled]
        distinct = []
        for v in values:
            if v not in distinct:
                distinct.append(v)
        if len(distinct) == 1:
            shared[key] = distinct[0]
        else:
            mixed[key] = distinct
    if mixed:
        print("  WARNING: the sources differ in how they were collected — merging anyway, "
              "and recording the mixture in the output's collection_info:")
        for key, values in mixed.items():
            print(f"    {key}: {' | '.join(json.dumps(v) for v in values)}")
    print(f"  lane={next(iter(by_lane))} across "
          f"{len(labelled)}/{len(sources)} labelled dataset(s)")
    return shared, mixed


def gate_schema_version(sources: list[Source]) -> None:
    """Refuse to merge recording schemas with different physical meanings.

    Matching column names/shapes do not imply matching coordinate frames, hand
    targets or controller checkpoint conventions. Historical repair scripts are
    not part of the supported pipeline: use the original collection version to
    inspect legacy data, keep schemas separate, or collect fresh demonstrations.
    """
    by_version: dict[object, list[str]] = {}
    for s in sources:
        version = (s.collection_info or {}).get("schema_version")
        by_version.setdefault(version, []).append(s.name)
    if len(by_version) > 1:
        groups = "\n".join(
            f"    schema_version={'(absent — pre-2026-08-11)' if v is None else v}: "
            f"{', '.join(names)}" for v, names in sorted(by_version.items(), key=lambda kv: str(kv[0])))
        raise PrepareError(
            "these datasets were collected under different recording schemas. Their "
            "columns have the same names and shapes but do not mean the same thing, so "
            "merging them would silently train one model on two conventions:\n" + groups +
            "\n  Keep each schema separate. Inspect legacy data with its original collection "
            "version or collect new demonstrations; unknown missing signals cannot be "
            "reconstructed reliably.")
    version = next(iter(by_version))
    if version is None:
        print("  schema_version: absent in every source (collected before it was "
              "tracked); matching column layouts cannot verify their physical conventions")
    else:
        print(f"  schema_version={version} across all {len(sources)} dataset(s)")


def gate_action_space(sources: list[Source]) -> None:
    """Structural backstop for datasets with no collection_info: same action keys.

    Works off meta/modality.json, so it still catches a cross-controller merge of
    datasets collected before collection_info existed.
    """
    seen: dict[frozenset, list[str]] = {}
    for s in sources:
        if s.modality:
            seen.setdefault(frozenset((s.modality.get("action") or {}).keys()),
                            []).append(s.name)
    if len(seen) <= 1:
        return
    groups = "\n".join(f"    [{', '.join(sorted(sig)) or 'no action keys'}]: {', '.join(names)}"
                       for sig, names in seen.items())
    raise PrepareError(
        "these datasets declare different action spaces in meta/modality.json and cannot "
        "be merged:\n" + groups + "\n  Merge each action space separately.")


def gate_fps(sources: list[Source]) -> int:
    rates = set()
    for source in sources:
        value = source.info.get("fps", 50)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0 or not float(value).is_integer()):
            raise PrepareError(f"{source.name}: prepare requires a positive integer fps, got {value!r}; "
                               "resample fractional-rate video explicitly before preparing it")
        rates.add(int(value))
    if len(rates) > 1:
        detail = ", ".join(f"{s.name}={s.info.get('fps')}" for s in sources)
        raise PrepareError(
            f"the sources were recorded at different control rates ({detail}). A merged "
            "dataset has one fps and its timestamps would be wrong for half the episodes.")
    return rates.pop() if rates else 50


def resolve_columns(sources: list[Source]) -> set[str]:
    """The columns the output carries: what EVERY source has.

    A column present in one input and not another cannot be merged honestly — filling it
    for the rest is inventing data, and keeping it ragged makes the meta a lie. The sim
    collector's `observation.eef_state` / `next.reward` / `next.done` are the normal case
    (hardware cannot observe them), so this drops them and says so rather than refusing.
    """
    kept = set.intersection(*[s.columns for s in sources]) if sources else set()
    for s in sources:
        dropped = sorted(s.columns - kept)
        if dropped:
            print(f"  {s.name}: dropping {len(dropped)} column(s) absent from the other "
                  f"source(s): {', '.join(dropped)}")
    if not kept:
        raise PrepareError("the sources share no columns at all — nothing to merge.")
    return kept


def prune_modality(modality: dict | None, kept_columns: set[str], *,
                   video_keys: set[str] | None = None) -> dict:
    """Reference modality minus every group whose backing feature is not in the output.

    A dropped ACTION group is fatal: the action space is what the model learns, and the
    reference meta would otherwise advertise a target that half the rows do not carry.
    Dropped STATE groups are reported and removed — that is how a sim dataset's
    eef_state groups disappear when it is merged with hardware data, and it is also what
    repairs a dataset whose meta already over-declared. VIDEO groups are checked against
    the output's video keys and ANNOTATION groups against its columns, for the same
    reason: a group pointing at a feature the output does not carry is a dangling
    reference the model's loader trips over at its first sample. (These two sections
    used to be skipped -- only state/action were pruned.)
    """
    modality = copy.deepcopy(modality) if modality else {}
    for section in list(modality):
        groups = modality.get(section) or {}
        if not isinstance(groups, dict):
            continue
        present = video_keys if section == "video" else kept_columns
        if present is None:
            continue
        phantom = [name for name, spec in groups.items()
                   if group_feature(section, spec) not in present]
        if phantom and section == "action":
            raise PrepareError(
                "the merged output would lose action group(s) "
                f"{', '.join(sorted(phantom))} — their column is missing from at least one "
                "source. Merge the sources that share an action space, separately.")
        for name in phantom:
            groups.pop(name)
        if phantom:
            print(f"  meta: dropped {section} group(s) with no column in the output: "
                  f"{', '.join(sorted(phantom))}")
    return modality


# --------------------------------------------------------------------------- #
# Optional stages — each one asks the data whether it applies
# --------------------------------------------------------------------------- #

@dataclass
class Stages:
    """Which optional stages run, decided from the sources' own schema.

    A derived column is produced only if EVERY source can produce it: a column that
    exists for some episodes and not others is exactly the "foreign field" this pipeline
    exists to avoid.
    """

    smpl: dict[str, str] = field(default_factory=dict)          # source name -> SMPL column
    hands: dict[str, dict] = field(default_factory=dict)        # source name -> {side: slice}
    smooth: list[str] = field(default_factory=list)             # raw columns to smooth

    @classmethod
    def resolve(cls, sources: list[Source], *, do_discard: bool, do_smpl: bool,
                do_open: bool, do_smooth: bool) -> "Stages":
        self = cls()

        if do_smpl:
            columns = {s.name: s.smpl_column() for s in sources}
            if all(columns.values()):
                self.smpl = {k: v for k, v in columns.items() if v}
                print(f"  stale-frame cleaning: on ({', '.join(sorted(set(self.smpl.values())))})")
            else:
                missing = [n for n, c in columns.items() if not c]
                print(f"  stale-frame cleaning: off — no SMPL reference column "
                      f"({' / '.join(SMPL_COLUMNS)}) in {', '.join(missing)}")

        if do_open:
            slices = {s.name: s.hand_state_slices() for s in sources}
            if all(slices.values()):
                self.hands = {k: v for k, v in slices.items() if v}
                print("  hand open/close labels: on (from the left_hand/right_hand state "
                      "groups in each dataset's modality.json)")
            else:
                missing = [n for n, v in slices.items() if not v]
                print("  hand open/close labels: off — no left_hand/right_hand STATE "
                      f"groups in {', '.join(missing)}")

        if do_smooth:
            common = set.intersection(*[set(s.smooth_columns()) for s in sources]) \
                if sources else set()
            self.smooth = sorted(common)
            if self.smooth:
                print(f"  smoothing: on ({', '.join(self.smooth)}"
                      f"{' + the open labels' if self.hands else ''})")
            elif not self.hands:
                print("  smoothing: off — no hand-joint columns and no open labels to smooth")

        if not do_discard:
            print("  discard list: off (--no-discard)")
        return self


def hand_closed_label(joints: np.ndarray, threshold: float) -> np.ndarray:
    """Binary (0.0 open / 1.0 closed) per frame, shape (N, 1)."""
    flex = np.abs(joints[:, FLEXION_JOINTS]).astype(np.float32)
    closure = np.clip(flex / JOINT_MAX_ABS, 0.0, 1.0).mean(axis=1)
    return (closure > threshold).astype(np.float32)[:, None]


def contiguous_segments(valid: np.ndarray | None, length: int) -> list[tuple[int, int]]:
    """[start, end) runs of the KEPT frames that were adjacent in the source.

    `valid` are the source frame indices that survived stale-frame removal (None =
    all of them); the runs are expressed in the output's frame numbering. Wherever a
    run ends, frames were cut out of the middle of the episode and the two sides are
    seconds apart in wall time.
    """
    if valid is None or length == 0:
        return [(0, length)]
    v = np.asarray(valid)
    breaks = np.flatnonzero(np.diff(v) != 1) + 1
    bounds = [0, *breaks.tolist(), len(v)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def moving_average(arr: np.ndarray, window: int,
                   segments: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Centered moving average, applied to each contiguous segment on its own.

    Smoothing has to happen AFTER the stale frames are dropped (the zeros and the
    frozen lead-in are garbage, not signal) and must not reach across the cut: the
    frames on either side of a removed dropout can be seconds apart, and a window
    sliding over the seam averaged a pre-dropout hand pose into a post-dropout one --
    into `*_smooth`, the one column prepare adds as a training target. Symptom: trains
    fine, evaluates worse, nothing in any log.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if window <= 1:
        return arr
    out = np.empty_like(arr)
    for start, end in (segments or [(0, len(arr))]):
        out[start:end] = (pd.DataFrame(arr[start:end])
                          .rolling(window=window, center=True, min_periods=1)
                          .mean().to_numpy().astype(np.float32))
    return out


def stale_mask(smpl: np.ndarray) -> np.ndarray:
    """True where a frame should be removed: all-zero SMPL, plus the frozen lead-in.

    A dropped teleop stream writes zeros; the frames immediately before it are the last
    received pose repeated, which is indistinguishable from the operator standing still
    except that it always ends in a dropout. Walking backwards from each zero run while
    the pose does not change removes both.
    """
    n = len(smpl)
    is_zero = np.all(smpl == 0, axis=1)
    remove = is_zero.copy()
    diffs = np.zeros(n)
    diffs[1:] = np.sum(np.abs(smpl[1:] - smpl[:-1]), axis=1)
    for i in range(n):
        if is_zero[i]:
            j = i - 1
            while j >= 0 and diffs[j] == 0.0 and not is_zero[j]:
                remove[j] = True
                j -= 1
    return remove


def add_derived_columns(df: pd.DataFrame, stages: Stages, source: Source, *,
                        threshold: float, window: int,
                        segments: list[tuple[int, int]] | None = None) -> dict[str, np.ndarray]:
    """Write the derived hand columns into *df* in place; return them for the meta.

    `segments` are the contiguous runs of kept frames (see `contiguous_segments`);
    every smoothing pass stays inside one."""
    produced: dict[str, np.ndarray] = {}
    slices = stages.hands.get(source.name)
    if slices:
        for side, (column, start, end) in slices.items():
            joints = np.stack(df[column].values).astype(np.float32)[:, start:end]
            label = hand_closed_label(joints, threshold)
            df[OPEN_COLUMNS[side]] = list(label)
            produced[OPEN_COLUMNS[side]] = label
    for column in stages.smooth:
        raw = np.stack(df[column].values).astype(np.float32)
        smoothed = moving_average(raw, window, segments)
        df[column + SMOOTH_SUFFIX] = list(smoothed)
        produced[column + SMOOTH_SUFFIX] = smoothed
    if slices and stages.smooth:
        # the open labels are worth smoothing for the same reason the joints are: a
        # threshold crossing chatters frame-to-frame around the closure boundary.
        for side in slices:
            key = OPEN_COLUMNS[side]
            smoothed = moving_average(produced[key], window, segments)
            df[key + SMOOTH_SUFFIX] = list(smoothed)
            produced[key + SMOOTH_SUFFIX] = smoothed
    return produced


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #

def copy_or_filter_video(src: Path, dst: Path, valid: np.ndarray | None, fps: int,
                         hold_frames: int = 0) -> None:
    """Copy *src* to *dst*, filtering and/or appending frozen terminal frames.

    The plain-copy path is what almost every run takes (no stale frames and no terminal
    hold), and it is also why `av` is a lazy import: an env without it can still prepare
    any dataset that needs no re-encoding.
    """
    if valid is None and hold_frames == 0:
        shutil.copy2(src, dst)
        return

    import av

    with av.open(str(src)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(
            container.streams.video[0])]
    filtered = frames if valid is None else [frames[i] for i in valid if i < len(frames)]
    if not filtered:
        return
    if hold_frames:
        filtered.extend([filtered[-1]] * hold_frames)
    with av.open(str(dst), mode="w") as out:
        stream = out.add_stream("h264", rate=fps)
        stream.height, stream.width = filtered[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for img in filtered:
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def append_terminal_hold(df: pd.DataFrame, frames: int) -> pd.DataFrame:
    """Append *frames* exact copies of the final prepared row.

    Counters are rewritten later by :func:`write_dataset`. The terminal transition is
    moved to the new final row, while appended rewards are zero so an original success
    reward remains attached to the actual success frame.
    """
    if frames == 0:
        return df
    if frames < 0:
        raise ValueError("terminal hold frames must be non-negative")
    if df.empty:
        raise ValueError("cannot append a terminal hold to an empty episode")

    length = len(df)
    tail = pd.concat([df.iloc[[-1]]] * frames, ignore_index=True).copy()
    if "next.reward" in tail.columns:
        tail["next.reward"] = np.zeros(frames, dtype=df["next.reward"].dtype)
    if "next.done" in tail.columns:
        tail["next.done"] = np.zeros(frames, dtype=bool)
    out = pd.concat([df, tail], ignore_index=True)
    if "next.done" in out.columns:
        out.loc[length - 1, "next.done"] = False
        out.loc[len(out) - 1, "next.done"] = True
    return out


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def column_stats(arr: np.ndarray) -> dict:
    """The one field set both stats files use: mean/std/min/max/q01/q99 per dimension."""
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def numeric_matrix(df: pd.DataFrame, col: str) -> np.ndarray | None:
    """(N, D) float32 view of a numeric column; None for strings / ragged / non-numeric.

    Video columns are never here — the frames live in videos/, not the parquet."""
    if len(df) == 0 or isinstance(df[col].iloc[0], str):
        return None
    try:
        return np.vstack([np.asarray(x, dtype=np.float32).reshape(-1) for x in df[col]])
    except Exception:                           # ragged or non-numeric column
        return None


def per_episode_stats(df: pd.DataFrame) -> tuple[dict, dict[str, np.ndarray]]:
    """(stats for meta/episodes_stats.jsonl, {column: matrix}) for one episode.

    LeRobot v2.1 expects meta/episodes_stats.jsonl, and its loader treats a dataset
    without one as incomplete: it falls back to fetching the repo from the Hub, which on
    an offline cluster surfaces as a DNS error rather than "your meta is missing a file".
    The per-episode entries carry the same fields as stats.json plus `count`, so one
    convention covers both files (scripts/gen_episodes_stats.py and an older version
    of this writer disagreed on q01/q99). The matrices are returned so the caller can
    accumulate the dataset-wide stats.json from the same pass.
    """
    stats: dict = {}
    matrices: dict[str, np.ndarray] = {}
    for col in df.columns:
        arr = numeric_matrix(df, col)
        if arr is None:
            continue
        stats[col] = {**column_stats(arr), "count": [int(arr.shape[0])]}
        matrices[col] = arr
    return stats, matrices


# --------------------------------------------------------------------------- #
# Episode selection
# --------------------------------------------------------------------------- #

def select_episodes(source: Source, stages: Stages, *, do_discard: bool) -> tuple[list[dict], int, int]:
    """Episodes to write from one source, with their frame masks already computed.

    Reads only the SMPL column when checking for stale frames — the full parquet is
    loaded one episode at a time at write time, not here.
    """
    discarded = set(source.info.get("discarded_episode_indices") or []) if do_discard else set()
    if discarded:
        print(f"  {source.name}: discarding {len(discarded)} episode(s): {sorted(discarded)}")
    smpl_column = stages.smpl.get(source.name)

    episodes: list[dict] = []
    dropped_stale = 0
    for meta in source.episodes_meta:
        index = meta["episode_index"]
        if index in discarded:
            continue
        path = parquet_path(source.path, source.info, index)
        if not path.exists():
            print(f"  {source.name}: WARNING missing parquet for episode {index}, skipping")
            continue

        valid = None
        if smpl_column:
            smpl = np.vstack([np.asarray(x, dtype=np.float32)
                              for x in pd.read_parquet(path, columns=[smpl_column])[smpl_column]])
            mask = stale_mask(smpl)
            n_remove = int(mask.sum())
            if n_remove:
                n_zero = int(np.all(smpl == 0, axis=1).sum())
                print(f"  {source.name}: episode {index} — removing {n_remove}/{len(mask)} "
                      f"frames ({100.0 * n_remove / len(mask):.1f}%): "
                      f"{n_zero} zero + {n_remove - n_zero} frozen lead-in")
                if n_remove == len(mask):
                    print(f"  {source.name}: episode {index} entirely stale — dropping it")
                    dropped_stale += 1
                    continue
                valid = np.where(~mask)[0]

        episodes.append({
            "source": source,
            "parquet_path": path,
            "video_paths": video_paths(source.path, source.info, index),
            "valid_indices": valid,
            "meta": meta,
        })
    return episodes, len(discarded), dropped_stale


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #

def write_dataset(output: Path, episodes: list[dict], sources: list[Source], *,
                  stages: Stages, kept_columns: set[str], modality: dict,
                  tasks_meta: list[dict],
                  collection_info: dict, mixed_info: dict[str, list], fps: int,
                  threshold: float, window: int, video_workers: int,
                  hold_frames: int = 0) -> None:
    output = output_directory([source.path for source in sources], output, force=True)
    if output.exists() and any(output.iterdir()):
        raise PrepareError(f"refusing to write into a nonempty output directory: {output}")
    info = copy.deepcopy(sources[0].info)
    chunks_size = info.get("chunks_size", 1000)
    data_pattern = info.get("data_path", DEFAULT_DATA_PATH)
    video_pattern = info.get("video_path", DEFAULT_VIDEO_PATH)
    video_keys = video_keys_of(info)
    task_lookup = {t["task_index"]: t["task"] for t in tasks_meta}

    planned: set[Path] = set()
    for new_index in range(len(episodes)):
        for pattern, key in [(data_pattern, ""), *[(video_pattern, key) for key in video_keys]]:
            target = dataset_file(output, pattern, episode_chunk=new_index // chunks_size,
                                  episode_index=new_index, video_key=key, output=True)
            if target in planned:
                raise PrepareError(f"multiple output resources resolve to the same path: {target}")
            planned.add(target)

    meta_dir = output / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    videos_written = 0
    episodes_jsonl: list[dict] = []
    episodes_stats: list[dict] = []
    accumulated: dict[str, list[np.ndarray]] = {}
    written_columns: set[str] = set()
    produced_widths: dict[str, int] = {}
    futures = []

    with ThreadPoolExecutor(max_workers=video_workers) as pool:
        for new_index, episode in enumerate(episodes):
            source: Source = episode["source"]
            chunk = new_index // chunks_size

            df = pd.read_parquet(episode["parquet_path"])
            df = df.rename(columns={k: v for k, v in LEGACY_COLUMN_RENAMES.items()
                                    if k in df.columns})
            df = df[[c for c in df.columns if c in kept_columns]]

            valid = episode["valid_indices"]
            if valid is not None:
                df = df.iloc[valid].reset_index(drop=True)
            prepared_length = len(df)
            segments = contiguous_segments(valid, prepared_length)

            task_strings = episode["meta"].get("tasks", [])
            if "task_index" in df.columns:
                original = df["task_index"]
                remapped = original.map(source.task_map)
                if remapped.isna().any():
                    bad = sorted(set(original[remapped.isna()].tolist()))
                    raise PrepareError(
                        f"{episode['parquet_path']} has task_index values {bad} that are "
                        "not in its meta/tasks.jsonl")
                df["task_index"] = remapped.astype("int64")
                task_strings = [task_lookup[i] for i in sorted(df["task_index"].unique().tolist())]
                # The collectors also write the task index into the annotation column
                # the modality's `annotation.human.task_description` group names (the
                # GR00T loader reads the task string through it). It is the same number
                # and has to move with task_index, or a merge leaves the two pointing at
                # different tasks. Only a column that IS a copy of task_index is
                # remapped; an annotation with its own values (validity flags...) is not.
                for col in df.columns:
                    if (col.startswith("annotation.") and col != "task_index"
                            and pd.api.types.is_integer_dtype(df[col])
                            and bool((df[col].values == original.values).all())):
                        df[col] = df["task_index"]

            produced = add_derived_columns(df, stages, source, threshold=threshold,
                                           window=window, segments=segments)
            for key, arr in produced.items():
                produced_widths[key] = int(arr.shape[1])

            # Hold is deliberately after cleaning and smoothing: every appended row is
            # an exact copy of the final training row, not another input to a centered
            # filter that would also change the original tail.
            df = append_terminal_hold(df, hold_frames)
            length = len(df)
            df["episode_index"] = new_index
            df["index"] = np.arange(total_frames, total_frames + length)
            df["frame_index"] = np.arange(length)
            if "timestamp" in df.columns:
                df["timestamp"] = np.arange(length, dtype=np.float32) / fps

            out_parquet = dataset_file(output, data_pattern, episode_chunk=chunk,
                                       episode_index=new_index, output=True)
            out_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_parquet, index=False)
            written_columns.update(df.columns)
            ep_stats, matrices = per_episode_stats(df)
            episodes_stats.append({"episode_index": new_index, "stats": ep_stats})
            # stats.json must cover EVERY numeric column, not just the derived ones:
            # Isaac-GR00T normalises from it and KeyErrors on a missing key, and the
            # columns that matter most (observation.state, action.*) are the raw ones.
            for key, arr in matrices.items():
                accumulated.setdefault(key, []).append(arr)

            for key in video_keys:
                src_video = episode["video_paths"].get(key)
                if not (src_video and src_video.exists()):
                    raise PrepareError(f"missing declared {key} video for source episode "
                                       f"{episode['meta']['episode_index']}: {src_video}")
                videos_written += 1
                dst_video = dataset_file(output, video_pattern, video_key=key,
                                         episode_index=new_index, episode_chunk=chunk, output=True)
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                futures.append(pool.submit(copy_or_filter_video, src_video, dst_video,
                                           valid, fps, hold_frames))

            episodes_jsonl.append({"episode_index": new_index, "tasks": task_strings,
                                   "length": length})
            total_frames += length
            print(f"  [{new_index + 1}/{len(episodes)}] {source.name} "
                  f"episode {episode['meta']['episode_index']} -> {new_index} "
                  f"({length} frames)", flush=True)

    for future in futures:                          # propagate video-worker exceptions
        future.result()

    # ---- meta ------------------------------------------------------------
    features = info.setdefault("features", {})
    action_groups = modality.setdefault("action", {}) if produced_widths else None
    for key, width in produced_widths.items():
        features.setdefault(key, {"dtype": "float32", "shape": [width], "names": None})
        # Declared as ACTION groups, which is what makes them trainable targets: the
        # sonic VLA predicts hand open/close, and a model's data config selects its keys
        # out of modality.json, not out of the parquet.
        name = key.split(".", 1)[1] if "." in key else key
        action_groups.setdefault(name, {"start": 0, "end": width, "original_key": key})
    phantom = [k for k, v in features.items()
               if (v or {}).get("dtype") not in ("video", "image") and k not in written_columns]
    for key in phantom:
        features.pop(key)
    if phantom:
        print(f"  meta: dropped {len(phantom)} feature(s) with no column in the output: "
              f"{', '.join(sorted(phantom))}")

    info["total_episodes"] = len(episodes)
    info["total_frames"] = total_frames
    info["total_videos"] = videos_written          # counted, not assumed
    info["total_chunks"] = max(1, math.ceil(len(episodes) / chunks_size))
    info["total_tasks"] = len(tasks_meta)
    info["fps"] = fps
    if "discarded_episode_indices" in info:      # only if the source tracked one
        info["discarded_episode_indices"] = []
    if isinstance(info.get("splits"), dict) and "train" in info["splits"]:
        info["splits"]["train"] = f"0:{len(episodes)}"
    if collection_info or mixed_info:
        merged_info = dict(collection_info)
        if mixed_info:
            # what differed between the sources, kept rather than inherited from the
            # first one — a merged dataset should say that it is a mixture.
            merged_info["mixed_across_sources"] = mixed_info
        info["collection_info"] = merged_info
    else:
        info.pop("collection_info", None)
    if hold_frames:
        info.setdefault("collection_info", {})["terminal_hold"] = {
            "frames_appended": hold_frames,
            "note": ("final prepared row repeated; frame_index/timestamp/index advance, "
                     "next.done moves to the new final frame, and appended reward is 0"),
        }
    info["prepared_from"] = [str(s.path) for s in sources]

    (meta_dir / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=4) + "\n")
    (meta_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes_jsonl))
    (meta_dir / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes_stats))
    if tasks_meta:
        (meta_dir / "tasks.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in tasks_meta))
    stats = {key: column_stats(np.concatenate(chunks, axis=0))
             for key, chunks in accumulated.items() if chunks}
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

def build_task_table(sources: list[Source], overrides: list[str] | None) -> list[dict]:
    """Dedupe task strings across sources and fill in each source's index remap.

    Every single-task dataset uses task_index 0, so merging without a remap silently
    relabels episodes. `--task-overrides` renames a source's task at merge time; the
    sources themselves are never modified.
    """
    merged: list[str] = []
    for i, source in enumerate(sources):
        if overrides:
            if len(source.tasks_meta) != 1:
                raise PrepareError(
                    f"--task-overrides expects single-task inputs, but {source.path} has "
                    f"{len(source.tasks_meta)} task(s)")
            strings = {source.tasks_meta[0]["task_index"]: overrides[i]}
        else:
            strings = {t["task_index"]: t["task"] for t in source.tasks_meta}
        source.task_map = {}
        for old_index, task in strings.items():
            if task not in merged:
                merged.append(task)
            source.task_map[old_index] = merged.index(task)
    return [{"task_index": i, "task": t} for i, t in enumerate(merged)]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess / merge collected LeRobot datasets (controller-agnostic).")
    parser.add_argument("--dataset-paths", nargs="*", type=Path, default=None, metavar="DIR",
                        help="One or more source dataset directories.")
    parser.add_argument("--dataset-list", type=Path, default=None, metavar="FILE",
                        help="Text file with one dataset path per line (# = comment).")
    parser.add_argument("--output-path", type=Path, default=None, metavar="DIR",
                        help="Destination (never in-place). Required when merging.")
    parser.add_argument("--output-root", type=Path, default=None, metavar="DIR",
                        help="Root for the default output dir of a single input "
                             "(default: <input>/../../outputs_pre).")
    parser.add_argument("--closure-threshold", type=float, default=DEFAULT_CLOSURE_THRESHOLD,
                        help=f"Finger closure > this -> hand closed (default: "
                             f"{DEFAULT_CLOSURE_THRESHOLD})")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"Centered moving-average window in frames (default: "
                             f"{DEFAULT_WINDOW})")
    parser.add_argument("--task-overrides", nargs="+", default=None, metavar="TASK",
                        help="Replace each input's task string at merge time, one per "
                             "dataset in --dataset-paths order (sources are untouched).")
    parser.add_argument("--no-discard", action="store_true",
                        help="Keep episodes listed in discarded_episode_indices.")
    parser.add_argument("--no-smpl", action="store_true",
                        help="Skip stale-frame removal even where a SMPL column exists.")
    parser.add_argument("--no-open", action="store_true",
                        help="Skip hand open/close labelling even where the modality "
                             "declares hand state groups.")
    parser.add_argument("--no-smooth", action="store_true", help="Skip smoothing.")
    parser.add_argument("--video-workers", type=int, default=min(4, os.cpu_count() or 4),
                        metavar="N", help="Thread-pool size for video copy/encode.")
    parser.add_argument("--hold-frames", type=int, default=0, metavar="N",
                        help="Append N frozen terminal frames to every episode (default: 0, "
                             "disabled). Videos are re-encoded to match.")
    args = parser.parse_args()

    if args.hold_frames < 0:
        parser.error("--hold-frames must be >= 0")

    paths: list[Path] = list(args.dataset_paths or [])
    if args.dataset_list:
        paths += [Path(line.strip()) for line in args.dataset_list.read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
    if not paths:
        parser.error("no dataset paths given: use --dataset-paths or --dataset-list")
    for path in paths:
        if not (path / "meta" / "info.json").exists():
            parser.error(f"not a LeRobot dataset (no meta/info.json): {path}")
    if args.task_overrides and len(args.task_overrides) != len(paths):
        parser.error(f"--task-overrides needs one task per dataset "
                     f"({len(args.task_overrides)} given, {len(paths)} datasets)")

    if args.output_path is not None:
        output = args.output_path.expanduser().absolute()
    elif len(paths) == 1:
        root = args.output_root or (paths[0].resolve().parent.parent / "outputs_pre")
        output = Path(root).resolve() / paths[0].resolve().name
    else:
        parser.error("--output-path is required when merging multiple datasets")
    try:
        output = output_directory(paths, output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    print("=" * 70)
    print("  lab prepare — LeRobot dataset preparation")
    print("=" * 70)
    for path in paths:
        print(f"  input : {path}")
    print(f"  output: {output}")

    sources = [Source.load(p.resolve()) for p in paths]
    # An aborted run leaves a dataset dir with meta but no episodes. It has no columns,
    # so it would empty the intersection below and take every real column with it.
    empty = [s for s in sources if not s.episodes_meta]
    for source in empty:
        print(f"  skipping {source.name}: 0 episodes recorded")
    sources = [s for s in sources if s.episodes_meta]
    if not sources:
        raise PrepareError("none of the given datasets contains any episode.")

    print("\nChecking that the sources can be merged...")
    collection_info, mixed_info = gate_backend(sources)
    gate_schema_version(sources)
    gate_action_space(sources)
    fps = gate_fps(sources)
    kept_columns = resolve_columns(sources)
    # Before any episode is written: the merged meta is built from the first source's
    # modality, and this is where a merge that would silently lose an action is refused.
    modality = prune_modality(sources[0].modality, kept_columns,
                              video_keys=set(video_keys_of(sources[0].info)))

    print("\nDeciding which stages apply (from each dataset's own schema)...")
    stages = Stages.resolve(sources, do_discard=not args.no_discard,
                            do_smpl=not args.no_smpl, do_open=not args.no_open,
                            do_smooth=not args.no_smooth)

    tasks_meta = build_task_table(sources, args.task_overrides)

    print("\nSelecting episodes...")
    episodes: list[dict] = []
    total_in = total_discarded = total_stale = 0
    for source in sources:
        kept, discarded, stale = select_episodes(source, stages,
                                                 do_discard=not args.no_discard)
        episodes += kept
        total_in += len(source.episodes_meta)
        total_discarded += discarded
        total_stale += stale
        print(f"  {source.name}: {len(kept)} episode(s) kept "
              f"({discarded} discarded, {stale} dropped as stale)")
    if not episodes:
        raise PrepareError("no episodes left after filtering.")

    print(f"\nWriting {len(episodes)} episode(s) to {output} ...")
    with staged_output(paths, output) as staging:
        write_dataset(staging, episodes, sources, stages=stages, kept_columns=kept_columns,
                      modality=modality, tasks_meta=tasks_meta, collection_info=collection_info,
                      mixed_info=mixed_info, fps=fps, threshold=args.closure_threshold,
                      window=args.window, video_workers=args.video_workers,
                      hold_frames=args.hold_frames)

    print("\n" + "=" * 70)
    print(f"  input episodes : {total_in}")
    print(f"  discarded      : {total_discarded}")
    print(f"  dropped (stale): {total_stale}")
    print(f"  output episodes: {len(episodes)}")
    if args.hold_frames:
        print(f"  terminal hold  : {args.hold_frames} frame(s) per episode")
    print(f"  tasks          : {len(tasks_meta)}")
    print(f"  output path    : {output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
