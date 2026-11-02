"""Model-independent replay workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from locomanilab.plugin import StageResult
from locomanilab.common.sim import runtime_checkout_error, runtime_pythonpath
from locomanilab.workflows.context import WorkflowContext


class ReplayWorkflow(WorkflowContext):
    def replay(self, *, source_runs: list[Path], randomize: str = "standard",
                variants: int = 1, seed: int = 0, episodes: str = "all",
                reroll: str = "0", out_base: Path | None = None,
                object_textures: bool = False, distractor_groups: str | None = None,
                preroll_ticks: int | None = None, keep_failed: bool = False,
                viewer: bool = False, write: bool | None = None,
                sim_conda_env: str = "tele_benchmark", **opts: Any) -> StageResult:
        """Replay collected runs, optionally in randomized scenes.

        Model-independent, like `prepare`: the work is the sim repo's
        `robocasa.scripts.replay_augment`, which replays each recorded episode
        action-for-action through the backend that collected it. With
        `randomize` set the scene's appearance is re-rolled first; physics is
        untouched either way, so the source labels stay valid and only the
        videos are re-rendered. With `write` that turns a replay into
        label-preserving augmented data; without it, it answers "does this run
        still reach success?" and leaves nothing behind.

        `randomize` is the SAME spec string `lab eval --randomize` takes (presets
        off/mild/standard/stress, or comma-separated tokens) — a spec previewed with
        `robocasa.scripts.preview_randomization` copies here unchanged. Replay draws
        from the 'train' half of the texture pool and eval from 'test', so a policy
        trained on this data is evaluated on textures it never saw.

        Each source run is replayed independently; `variants` output runs per source,
        seeded `seed`..`seed+variants-1`. Outputs default to a SIBLING lane
        (`outputs_sim/sonic` -> `outputs_sim/sonic_replay`) so generated runs never
        mix with the originals `lab prepare` would glob alongside them.
        """
        sim_repo = self.ctx.sim_repo()
        if not sim_repo.exists():
            return StageResult(ok=False, message=(
                "replay needs the humanoid_benchmark submodule: run "
                f"`git submodule update --init external/humanoid_benchmark` ({sim_repo})"))
        runtime = self.ctx.repo_path("g1_backend")
        if not self.runner.dry_run and (problem := runtime_checkout_error(runtime)):
            return StageResult(ok=False, message=problem)
        results = []
        for run in source_runs:
            run = self.ctx.anchor(run).resolve()
            cmd = ["python", "-m", "robocasa.scripts.replay_augment",
                   "--source-run", str(run),
                   "--randomize", randomize,
                   "--variants", str(variants), "--seed", str(seed),
                   "--episodes", episodes,
                   "--reroll", str(reroll)]
            if out_base is not None:
                cmd += ["--out-base", str(self.ctx.anchor(out_base).resolve())]
            if object_textures:
                cmd.append("--randomize-object-textures")
            if distractor_groups:
                cmd += ["--distractor-groups", distractor_groups]
            if preroll_ticks is not None:
                cmd += ["--preroll-ticks", str(preroll_ticks)]
            if keep_failed:
                cmd.append("--keep-failed")
            if viewer:
                cmd.append("--viewer")
            if write is not None:
                cmd.append("--write" if write else "--no-write")
            res = self.runner.run(
                cmd, cwd=sim_repo, env_kind="conda", env_name=sim_conda_env,
                setup=self.env_setup("conda"),
                # The interactive window needs a real GL context; egl is
                # headless-only and the viewer comes up black under it.
                extra_env={"PYTHONPATH": runtime_pythonpath(
                               sim_repo=sim_repo, runtime_repo=runtime),
                           **({} if viewer else {"MUJOCO_GL": "egl"})},
                label="replay", stage=run.name)
            results.append(res)
            if not res.ok:
                # A failed source run is worth stopping on: the usual cause is a
                # bad spec or a run without the fixed-task sidecars, and both
                # would fail identically for every run that follows.
                return StageResult(ok=False, message=(
                    f"replay failed on {run.name}: {res.message}"))
        if not results:
            return StageResult(ok=False, message="no source runs given")
        return results[-1] if len(results) == 1 else StageResult(
            ok=True, message=f"replayed {len(results)} source run(s)")
