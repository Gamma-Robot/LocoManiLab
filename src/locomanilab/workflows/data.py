"""Model-independent data workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from locomanilab.plugin import StageResult
from locomanilab.workflows.context import WorkflowContext


class DataWorkflow(WorkflowContext):
    def prepare(self, *, datasets: list[Path], out: Path | None = None,
                merge_only: bool = False, tasks: list[str] | None = None,
                hold_frames: int = 0,
                **opts: Any) -> StageResult:
        """Preprocess / merge collected datasets — locomanilab's own script, for every backend.

        Model-independent: which stages run is decided from each dataset's own
        meta/modality.json, not from the plugin this call happened to arrive through
        (it used to shell out to sonic's gear_sonic/scripts/prepare_dataset.py, whose
        hand stage sliced sonic's 43-dim state out of whatever it was handed and
        crashed on every blob-schema backend).
        """
        import sys

        from locomanilab.common import variants as variantmod
        from locomanilab.data import prepare_script_path

        if not datasets:
            return StageResult(ok=False, message="no datasets given")
        datasets = [self.ctx.anchor(d) for d in datasets]
        out = self.ctx.anchor(out) if out is not None else None

        mixed = variantmod.same_controller([Path(d) for d in datasets])
        if mixed is not None:
            return StageResult(ok=False, message=mixed)

        env_kind, env_name, venv_path = self.ctx.data_env()
        python = sys.executable if env_kind == "none" else "python"
        if env_kind == "none" and not self.runner.dry_run:
            # No data env declared: run under the CLI's own interpreter, but only if it
            # can actually do the work — a missing pandas otherwise surfaces as a
            # traceback from a script the user never invoked by hand.
            try:
                from importlib.util import find_spec
                if find_spec("pandas") is None or find_spec("pyarrow") is None:
                    raise ImportError("pandas or pyarrow is not installed")
            except ImportError as exc:
                return StageResult(ok=False, message=(
                    f"`lab prepare` needs pandas + pyarrow and this interpreter has none "
                    f"({exc}). Either `pip install -e .[data]` into the locomanilab env, or name "
                    f"an env that has them in env.local.yaml:\n"
                    f"    conda_envs: {{data: <env name>}}\n"
                    f"    paths: {{data_venv: /path/to/.venv}}"))
            python = sys.executable

        # by absolute path: the script is part of the locomanilab package, so it is found the
        # same way whether locomanilab is an editable checkout or installed into site-packages.
        cmd = [python, str(prepare_script_path()),
               "--dataset-paths", *[str(d) for d in datasets]]
        if out is not None:
            cmd += ["--output-path", str(out)]
        if merge_only:
            cmd += ["--no-discard", "--no-smpl", "--no-open", "--no-smooth"]
        if tasks:
            cmd += ["--task-overrides", *tasks]
        if hold_frames:
            cmd += ["--hold-frames", str(hold_frames)]
        return self.runner.run(cmd, cwd=self.ctx.config_root, env_kind=env_kind, env_name=env_name,
                               venv_path=venv_path, setup=self.env_setup(env_kind),
                               label="prepare",
                               stage=out.name if out is not None else Path(datasets[0]).name)
