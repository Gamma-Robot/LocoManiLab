"""Validate the PRTS column recipe against the controller's serving contract."""
from __future__ import annotations

from typing import Any


# Order and widths match the pinned SONIC policy protocol. The default columns
# are the collector's public layout; renamed columns need explicit modality slices.
_LAYOUTS = {
    "sonic": {
        "state": [("left_leg", 6), ("right_leg", 6), ("waist", 3), ("left_arm", 7),
                  ("left_hand", 7), ("right_arm", 7), ("right_hand", 7)],
        "action": [("motion_token", 64), ("left_hand", 7), ("right_hand", 7)],
        "columns": [("action.motion_token", 64), ("teleop.left_hand_joints", 7),
                    ("teleop.right_hand_joints", 7)],
    },
}


def validate_recipe(row: dict[str, Any], info: dict, modality: dict, variant: str) -> str | None:
    """Return a useful error instead of training a head the server cannot decode."""
    if row.get("state_relative_action") is not False:
        return (f"PRTS {variant} requires state_relative_action: false (absolute controller actions); "
                "relative-action recipes are not supported by the policy server")
    features = info.get("features") or {}
    if not features:
        return "PRTS needs info.json features with state/action widths to verify the policy server contract"
    layout = _LAYOUTS[variant]
    for section in ("state", "action"):
        keys = row[f"select_{section}_keys"]
        if len(set(keys)) != len(keys):
            return f"PRTS select_{section}_keys contains duplicate columns"
        columns: dict[str, tuple[int, int]] = {}
        total = 0
        for key in keys:
            feature = features.get(key) or {}
            shape = feature.get("shape")
            if (not isinstance(shape, (list, tuple)) or len(shape) != 1
                    or not isinstance(shape[0], int) or shape[0] <= 0
                    or feature.get("dtype") in ("video", "image", "string")):
                return f"PRTS cannot verify numeric vector width for dataset column {key!r}"
            columns[key] = (total, shape[0])
            total += shape[0]
        groups = layout[section]
        expected = sum(width for _, width in groups)
        if total != expected:
            return f"PRTS {variant} selected {section} width is {total}, but the policy server requires {expected}"

        defaults = layout["columns"] if section == "action" else [("observation.state", expected)]
        canonical = keys == [key for key, _ in defaults] and all(
            columns[key][1] == width for key, width in defaults)
        offset = 0
        for group, width in groups:
            declared = modality.get(section) or {}
            # Current g1 exports name the commanded hand action *_hand_joints;
            # older prepared datasets used *_hand. They share the same slices.
            name = (f"{group}_joints" if section == "action" and group in ("left_hand", "right_hand")
                    and f"{group}_joints" in declared else group)
            spec = declared.get(name)
            if not isinstance(spec, dict):
                return f"PRTS {variant} requires modality group {section}.{group}"
            explicit = any(key in spec for key in ("original_key", "start", "end"))
            if explicit:
                column = spec.get("original_key", "observation.state" if section == "state" else "action")
                start, end = spec.get("start"), spec.get("end")
                if (column not in columns or not isinstance(start, int) or not isinstance(end, int)
                        or start < 0 or end > columns[column][1] or end - start != width
                        or columns[column][0] + start != offset):
                    return (f"PRTS {variant} selected {section} columns do not preserve {section}.{group} "
                            f"at offset {offset} with width {width}; use the controller's canonical layout")
            elif not canonical:
                return (f"PRTS cannot prove renamed {section} columns preserve {section}.{group}; "
                        "modality.json must declare original_key, start and end for every selected group")
            offset += width
    return None
