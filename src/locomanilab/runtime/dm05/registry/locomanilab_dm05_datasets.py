"""OpenDM dataset registry for locomanilab's G1 SONIC conversions (loaded through OPENDM_DATA_PATH).

OpenDM imports every ``*.py`` in ``$OPENDM_DATA_PATH`` at ``opendm.dataset.register`` import
time. This module registers one dataset per ``<root>/<name>/opendm_dataset.json`` written by
``locomanilab.runtime.dm05.convert``, so paths live in the conversion output, not in code.
``LOCOMANILAB_DM05_DATA_ROOT`` (os.pathsep-separated) picks the roots;
``SOMA_DM05_DATA_ROOT`` is accepted for existing environments.

The module name is deliberately unique (``locomanilab_dm05_datasets``): OpenDM puts the registry dir on
sys.path and imports by bare module name, so a generic name could shadow a built-in one.
"""
import json
import os
from pathlib import Path

from opendm.constants.robot import RobotStateDesc
from opendm.dataset.register import register_dataset

_roots = os.environ.get("LOCOMANILAB_DM05_DATA_ROOT") or os.environ.get("SOMA_DM05_DATA_ROOT", "")
_found = {}
for _root in filter(None, _roots.split(os.pathsep)):
    for _rec_path in sorted(Path(_root).glob("*/opendm_dataset.json")):
        _rec = json.loads(_rec_path.read_text())
        _found[_rec["name"]] = {
            "jsonl_dir": _rec["jsonl_dir"],
            "image_dir": _rec["image_dir"],
            "image_keys": _rec["image_keys"],
            "image_prompts": _rec["image_prompts"],
            # plain string, not a RobotType member: G1 is not in OpenDM's enum. Training only
            # renders it into the prompt ("Robot: Unitree G1") and keys the norm stats by it.
            "robot_type": _rec["robot_type"],
            # every SONIC state dim is a joint angle; only consulted in relative-action mode,
            # which locomanilab never uses (78-D action != 43-D state).
            "state_desc": [RobotStateDesc.JOINT] * int(_rec["state_dim"]),
            "fps": _rec["fps"],
        }
register_dataset(_found)
