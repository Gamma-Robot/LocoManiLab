"""Application options shared by the public command groups."""
from __future__ import annotations

from pathlib import Path
from contextvars import ContextVar

import typer
from typer.core import TyperGroup

from locomanilab import __version__
from locomanilab.config import ConfigError

CLI_OPTIONS: ContextVar[dict] = ContextVar("locomanilab_cli_options", default={})


class UserFacingGroup(TyperGroup):
    """Turn expected filesystem/configuration failures into concise CLI errors."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (ConfigError, OSError, ValueError) as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None

app = typer.Typer(
    cls=UserFacingGroup,
    add_completion=False, invoke_without_command=True,
    help="Humanoid data, policy, and evaluation workflows. Start with `lab init`.",
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_show_locals=False,
)


@app.callback()
def options(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", help="Local YAML configuration (or LOCOMANILAB_CONFIG)."),
    workspace: Path | None = typer.Option(None, "--workspace", help="Workspace root (or LOCOMANILAB_WORKSPACE)."),
    version: bool = typer.Option(False, "--version", is_eager=True, help="Show the installed version."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    ctx.ensure_object(dict)
    ctx.obj.update(config=config, workspace=workspace)
    CLI_OPTIONS.set({"config": config, "workspace": workspace})
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def main() -> None:
    app()
