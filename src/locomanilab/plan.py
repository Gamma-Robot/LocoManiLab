"""A serializable description of a local subprocess execution."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

_PRIVATE_ENV = re.compile(r"TOKEN|PASSWORD|PASSWD|SSHPASS|SECRET|API_KEY|PRIVATE_KEY|CREDENTIAL", re.IGNORECASE)


def public_environment(values: dict[str, str] | None) -> dict[str, str]:
    return {key: "[redacted]" if _PRIVATE_ENV.search(key) else str(value)
            for key, value in (values or {}).items()}


@dataclass(frozen=True)
class ExecutionPlan:
    argv: list[str]
    cwd: Path
    stage: str = "run"
    model: str = "locomanilab"
    environment: dict[str, str] = field(default_factory=dict)

    def display(self) -> str:
        prefix = f"  cd {shlex.quote(str(self.cwd))}"
        if self.environment:
            prefix += "\n  env: " + " ".join(
                f"{key}={value}" for key, value in public_environment(self.environment).items())
        return prefix + "\n  " + shlex.join(self.argv)

    def as_dict(self) -> dict:
        return {"argv": self.argv, "cwd": str(self.cwd), "stage": self.stage,
                "model": self.model, "environment": public_environment(self.environment)}
