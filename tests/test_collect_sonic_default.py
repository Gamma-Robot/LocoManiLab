"""The collect CLI resolves SONIC's default before selecting its dataset lane."""
from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from locomanilab import cli


def _sonic_export(ctx, variant: str):
    directory = (
        ctx.repo_path("g1_backend")
        / "checkpoints"
        / "sonic"
        / variant
        / "g1"
    )
    directory.mkdir(parents=True)
    for name in ("policy.yaml", "model_encoder.onnx", "model_decoder.onnx"):
        (directory / name).write_text("")
    return directory / "policy.yaml"


def _invoke_collect(monkeypatch, ctx, *args):
    captured = {}

    class Plugin:
        def collect(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(ok=True, message="ok", log_path=None)

    monkeypatch.setattr(cli.data, "_workflow", lambda _dry_run: (Plugin(), ctx))
    result = CliRunner().invoke(cli.app, ["collect", "--sim", "--dry-run", *args])
    assert result.exit_code == 0, result.output
    return captured


def test_collect_defaults_sonic_to_v1_1_lane(monkeypatch, ctx):
    manifest = _sonic_export(ctx, "v1_1")

    captured = _invoke_collect(monkeypatch, ctx)

    assert captured["policy_config"] == str(manifest)
    assert captured["out_dir"].parent.name == "sonic_v1_1"


def test_collect_can_explicitly_select_legacy_low_latency_lane(monkeypatch, ctx):
    manifest = _sonic_export(ctx, "low_latency")

    captured = _invoke_collect(monkeypatch, ctx, "--sonic-ckpt", "low_latency")

    assert captured["policy_config"] == str(manifest)
    assert captured["out_dir"].parent.name == "sonic"
