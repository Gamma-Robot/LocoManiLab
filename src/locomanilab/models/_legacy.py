"""Read older user-owned model checkouts without rewriting their files."""
from pathlib import Path


_CONVERTERS = {
    f"scripts/data/locomanilab_{suffix}.py": f"scripts/data/soma_{suffix}.py"
    for suffix in ("sonic_to_psi_lerobot", "amo_to_psi_lerobot", "heft_to_psi_lerobot",
                   "lerobot_v21_to_v30", "sonic_to_xr1_json")
}


def converter_path(root: Path, configured: str) -> str:
    legacy = _CONVERTERS.get(configured)
    if legacy and not (root / configured).exists() and (root / legacy).is_file():
        return legacy
    return configured
