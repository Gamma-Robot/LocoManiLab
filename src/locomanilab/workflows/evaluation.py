"""Policy-server evaluation workflows shared by every model adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from locomanilab.plugin import StageResult

def evaluate(plugin, *, prompt: str | None = None, port: int = 5550,
         env_name: str = "TabletopPickPlace", episodes: int = 1, max_steps: int = 60,
         action_horizon: int = 64, render: str = "video", video_dir: str | None = None,
         sim_env: str = "tele_benchmark", inference_rate: float | None = None,
         scene_id: int | None = None, backend: str = "sonic",
         **opts: Any) -> StageResult:
    """Closed-loop sim eval: this model's policy server + locomanilab's sim client.

    The selected backend and checkpoint determine the controller action contract;
    model-specific action-space options belong to the policy server.
    """
    from locomanilab.common import sim as simmod

    if video_dir is not None:
        video_dir = str(plugin.ctx.anchor(video_dir))
    if opts.get("results_json") is not None:
        opts["results_json"] = str(plugin.ctx.anchor(opts["results_json"]))
    if opts.get("policy_config") is not None:
        opts["policy_config"] = str(plugin.ctx.input_path(opts["policy_config"]))
    refused = plugin.refuse_unsupported("eval", plugin._sim_server, opts)
    if refused:
        return refused
    if plugin.runner.dry_run:
        # Readiness probes source user setup scripts and execute an interpreter.
        # Planning must remain purely static, including on an unprovisioned host.
        support = simmod.SimSupport(ok=True, sim_repo=plugin.ctx.sim_repo(), conda_env=sim_env,
                                   runtime_repo=plugin.ctx.repo_path("g1_backend"))
    else:
        support = simmod.detect(plugin.ctx, conda_env=sim_env, backend=backend,
                                policy_config=opts.get("policy_config"))
    if not support.ok:
        return StageResult(ok=False, message="sim eval not supported here:\n" + support.summary())
    server = plugin._sim_server(stage="eval", prompt=prompt, port=port, backend=backend, **opts)
    if isinstance(server, StageResult):
        return server
    rate = inference_rate if inference_rate is not None else server.inference_rate
    extra_client_args = simmod.client_extra_args(
        rate, scene_id, robot=opts.get("robot"), results_json=opts.get("results_json"),
        genuine_min_steps=opts.get("genuine_min_steps"),
        token_ema_alpha=opts.get("token_ema_alpha"),
        inference_backend=opts.get("inference_backend"),
        fixed_scene=opts.get("fixed_scene", False),
        scene_seed=opts.get("scene_seed"),
        policy_config=opts.get("policy_config"),
        reset_jitter=opts.get("reset_jitter"),
        reset_object_jitter=opts.get("reset_object_jitter"),
        jitter_seed=opts.get("jitter_seed"),
        randomize=opts.get("randomize"),
        randomize_seed=opts.get("randomize_seed"),
        render_scale=opts.get("render_scale"))
    result = simmod.orchestrate(
        support=support, server_argv=server.argv, server_cwd=server.cwd,
        runner=plugin.runner, port=port, client_env_name=sim_env, env_name=env_name,
        prompt=(server.prompt or "test"), episodes=episodes, max_steps=max_steps,
        action_horizon=action_horizon, render=render, video_dir=video_dir,
        backend=backend,
        conda_setup=plugin.env_setup("conda"), dry_run=plugin.runner.dry_run,
        extra_client_args=extra_client_args)
    if plugin.runner.dry_run:
        result.message += "; backend readiness was not checked (static plan only)"
    return result

def infer_check(plugin, *, dataset: Path, episode: int = 0, frames: int | None = None,
                out: str | None = None, prompt: str | None = None, port: int = 5550,
                sim_env: str = "tele_benchmark", **opts: Any) -> StageResult:
    """Offline open-loop check: predicted vs GT actions on a training dataset.

    Same server as `eval`, driven by locomanilab/runtime/infer_client.py instead of the
    sim. Any plugin with a `_sim_server` gets this for free.
    """
    from locomanilab.common import infer_check as ic

    dataset = plugin.ctx.input_path(dataset)
    refused = plugin.refuse_unsupported("infer-check", plugin._sim_server, opts)
    if refused:
        return refused
    server = plugin._sim_server(stage="infer-check", prompt=prompt, port=port, **opts)
    if isinstance(server, StageResult):
        return server
    out = out or f"runs/infer_check/{plugin.name}_ep{episode}.png"
    return ic.orchestrate(server_argv=server.argv, server_cwd=server.cwd, runner=plugin.runner,
                          port=port, client_env_name=sim_env,
                          conda_setup=plugin.env_setup("conda"), dataset=dataset,
                          episode=episode, frames=frames, out=out,
                          prompt=(server.prompt or "test"), dry_run=plugin.runner.dry_run,
                          out_root=plugin.ctx.config_root)
