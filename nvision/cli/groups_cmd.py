"""CLI subcommands for preset run groups defined in ``nvision.sim.run_groups``."""

from __future__ import annotations

import inspect
import typing
from typing import Annotated

import typer

from nvision.cli import defaults as cli_defaults
from nvision.cli import options as cli_options
from nvision.cli.app_instance import app
from nvision.cli.run import run
from nvision.sim import run_groups as sim_run_groups
from nvision.tools.paths import ARTIFACTS_ROOT

groups_app = typer.Typer(
    help="Run preset simulation groups (see nvision.sim.run_groups.RunGroup).",
    pretty_exceptions_show_locals=False,
)
app.add_typer(groups_app, name="groups")


@app.command("run-single")
def run_single(
    generator: Annotated[str, typer.Argument(help="Generator name (e.g. NVCenter-lorentzian)")],
    noise: Annotated[str, typer.Argument(help="Noise name (e.g. NoNoise, Gauss(0.01), SNR(20))")],
    strategy: Annotated[str, typer.Argument(help="Strategy name (e.g. Bayesian-SBED-NoSweep)")],
    repeats: cli_options.RepeatsOption = 1,
    loc_timeout_s: cli_options.LocTimeoutOption = cli_defaults.DEFAULT_LOC_TIMEOUT_S,
    no_cache: cli_options.NoCacheOption = False,
    purge: cli_options.PurgeOption = False,
    dry_run: cli_options.DryRunOption = False,
    runners: cli_options.RunnersOption = 1,
    no_progress: cli_options.NoProgressOption = False,
    open_browser: cli_options.OpenBrowserOption = False,
) -> int:
    """Run a single (generator, noise, strategy) combination.

    Accepts any noise descriptor (e.g. ``Gauss(0.15)``) even if it is not in
    the active preset noise grid.  The noise is built dynamically from the
    descriptor string via :func:`~nvision.sim.combinations._parse_noise`.
    """
    # Use single_run=True which routes through resolve() for an exact triple lookup.
    # This supports any noise value (Gauss(sigma)) regardless of the active
    # .env grid configuration.
    return run(
        out=ARTIFACTS_ROOT,
        repeats=repeats,
        loc_timeout_s=loc_timeout_s,
        filter_generator=generator,
        filter_noise=noise,
        filter_strategy=strategy,
        single_run=True,
        all_experiments=True,
        no_cache=no_cache,
        purge=purge,
        dry_run=dry_run,
        runners=runners,
        no_progress=no_progress,
        open_browser=open_browser,
    )


@groups_app.command("list")
def list_groups(
    names_only: bool = typer.Option(False, "--names-only", help="Print only the group names (one per line)"),
    aliases: bool = typer.Option(False, "--aliases", help="Also list the former group names that still resolve"),
) -> None:
    """List preset groups with their task count (generators x noises x strategies) and a description."""
    groups = sim_run_groups.run_groups()
    if aliases:
        groups = [*groups, *(sim_run_groups.get_run_group(n) for n in sim_run_groups.alias_names())]
    width = max(len(g.name) for g in groups)
    for group in groups:
        if names_only:
            typer.echo(group.name)
        else:
            typer.echo(f"{group.name:<{width}}  {group.task_count:>6} tasks  {group.description}")


@groups_app.command("count")
def count_group(
    group_name: Annotated[
        str,
        typer.Argument(help="Preset group name (same values as `nvision groups list`)."),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help='Emit {"count": N, "capped": M} instead of a bare integer.'),
    ] = False,
    max_shards: Annotated[
        int,
        typer.Option("--max-shards", help="Cap the reported count at this value."),
    ] = 20,
) -> None:
    """Print the (generator x noise x strategy) combination count for a run group.

    Used to size cross-pod sharding (e.g. Cloud Run Jobs `--tasks`) before a run
    starts, without executing anything. Plain mode prints exactly one line (the
    capped integer) to stdout, so a calling shell script can do
    `SHARD_COUNT=$(nv groups count <name>)`. On an unknown group name, prints
    the error to stderr and exits non-zero -- nothing is printed to stdout, so
    a caller can't accidentally deploy with an empty/zero task count.
    """
    try:
        group = sim_run_groups.get_run_group(group_name)
    except KeyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    count = group.task_count
    capped = min(count, max_shards) if max_shards > 0 else count

    if json_output:
        import json

        typer.echo(json.dumps({"count": count, "capped": capped}))
    else:
        typer.echo(str(capped))


# --- Top-level ``nvision run`` alias ------------------------------------------


@app.command()
def run_all(
    repeats: cli_options.RepeatsOption = cli_defaults.DEFAULT_REPEATS,
    loc_timeout_s: cli_options.LocTimeoutOption = cli_defaults.DEFAULT_LOC_TIMEOUT_S,
    no_cache: cli_options.NoCacheOption = False,
    purge: cli_options.PurgeOption = False,
    dry_run: cli_options.DryRunOption = False,
    runners: cli_options.RunnersOption = cli_defaults.DEFAULT_RUNNERS,
    open_browser: cli_options.OpenBrowserOption = cli_defaults.DEFAULT_OPEN_BROWSER,
    no_progress: cli_options.NoProgressOption = False,
    retry_failed: cli_options.RetryFailedOption = False,
    resume: cli_options.ResumeOption = False,
    resume_since: cli_options.ResumeSinceOption = None,
    resume_run: cli_options.ResumeRunOption = None,
) -> int:
    """Run all experiments (alias for ``nvision groups run voigt-grid``)."""
    return run(
        out=ARTIFACTS_ROOT,
        repeats=repeats,
        loc_timeout_s=loc_timeout_s,
        run_group=sim_run_groups.default_run_group().name,
        no_cache=no_cache,
        purge=purge,
        dry_run=dry_run,
        runners=runners,
        open_browser=open_browser,
        no_progress=no_progress,
        retry_failed=retry_failed,
        resume=resume,
        resume_since=resume_since,
        resume_run=resume_run,
    )


# --- ``groups run <name>`` / ``groups <name>``: ``nv run --run-group <name>`` with the same flags -------------


def _run_signature() -> tuple[inspect.Signature, dict[str, object]]:
    """``run``'s signature without ``--run-group`` (the group is given as the command/argument), plus its resolved
    type hints, so the group commands accept exactly the flags ``nv run`` does."""
    hints = typing.get_type_hints(run, include_extras=True)
    params = [p.replace(annotation=hints[n]) for n, p in inspect.signature(run).parameters.items() if n != "run_group"]
    return inspect.Signature(params, return_annotation=hints.get("return", inspect.Signature.empty)), hints


def _with_run_flags(func, *leading: inspect.Parameter):
    """Give ``func`` the CLI signature of ``run`` (minus ``--run-group``), optionally preceded by ``leading`` params."""
    sig, hints = _run_signature()
    func.__signature__ = sig.replace(parameters=[*leading, *sig.parameters.values()])
    func.__annotations__ = {
        **{p.name: p.annotation for p in leading},
        **{n: h for n, h in hints.items() if n != "run_group"},
    }
    return func


_GROUP_NAME_PARAM = inspect.Parameter(
    "group_name",
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    annotation=Annotated[
        str, typer.Argument(help="Preset group (same values as `nv groups list`, e.g. voigt-grid, all).")
    ],
)


def run_preset(group_name: str, **run_flags):
    """``nv groups run <group>``: identical to ``nv run --run-group <group>`` (every ``nv run`` flag applies)."""
    return run(run_group=sim_run_groups.get_run_group(group_name).name, **run_flags)


_with_run_flags(run_preset, _GROUP_NAME_PARAM)
groups_app.command("run", help="Run a preset group; same as `nv run --run-group <group>` and takes the same flags.")(
    run_preset
)


def _register_group_alias(group_name: str) -> None:
    """Register ``groups <group_name>`` as shorthand for ``groups run <group_name>``."""

    def alias(**run_flags):
        return run_preset(group_name, **run_flags)

    alias.__doc__ = f"Same as `nv run --run-group {group_name}` (and `groups run {group_name}`); takes the same flags."
    groups_app.command(group_name)(_with_run_flags(alias))


for _group_name in (*(g.name for g in sim_run_groups.run_groups()), *sim_run_groups.alias_names()):
    _register_group_alias(_group_name)
