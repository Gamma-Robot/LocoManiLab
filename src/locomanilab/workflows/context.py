"""Small context shared by workflow services; no model code is imported."""
from __future__ import annotations


class WorkflowContext:
    def __init__(self, ctx, runner):
        self.ctx = ctx
        self.runner = runner

    def env_setup(self, kind: str) -> str | None:
        key = {"uv": "uv_setup", "conda": "conda_setup"}.get(kind)
        path = self.ctx.path(key) if key else None
        return str(path) if path else None

    def server_argv(self, cmd, *, env_kind="none", env_name=None, venv_path=None, cwd=None, pre=None):
        return self.runner.wrap(cmd, env_kind=env_kind, env_name=env_name,
                                venv_path=venv_path, cwd=cwd, pre=pre,
                                setup=self.env_setup(env_kind))
