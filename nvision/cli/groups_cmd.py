"""CLI subcommands for preset run groups defined in ``nvision.sim.run_groups``."""

from __future__ import annotations

from typing import Annotated

import typer

from nvision.cli import defaults as cli_defaults
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
    noise: Annotated[str, typer.Argument(help="Noise name (e.g. NoNoise, Gauss(0.01))")],
    strategy: Annotated[str, typer.Argument(help="Strategy name (e.g. Bayesian-SBED-NoSweep)")],
    repeats: Annotated[int, typer.Option("--repeats", help="Number of repeats")] = 1,
    loc_max_steps: Annotated[
        int,
        typer.Option("--loc-max-steps", help="Max steps for Bayesian locator measurement loop"),
    ] = cli_defaults.DEFAULT_LOC_MAX_STEPS,
    loc_timeout_s: Annotated[
        int,
        typer.Option("--loc-timeout", help="Timeout in seconds for a single locator run"),
    ] = cli_defaults.DEFAULT_LOC_TIMEOUT_S,
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching for this run"),
    runners: int = typer.Option(
        1,
        "--runners",
        min=1,
        help=(
            "Number of runner processes. 1 = live logs/progress in main thread; "
            ">1 = subprocesses with reliable Ctrl-C but silent until done."
        ),
    ),
    no_progress: bool = typer.Option(
        False, "--no-progress", help="Disable Rich progress UI; print plain logs to terminal"
    ),
    open_browser: bool = typer.Option(
        False, "--open/--no-open", help="Open results in browser after run"
    ),
) -> int:
    """Run a single (generator, noise, strategy) combination."""
    return run(
        out=ARTIFACTS_ROOT,
        repeats=repeats,
        loc_max_steps=loc_max_steps,
        loc_timeout_s=loc_timeout_s,
        filter_generator=generator,
        filter_noise=noise,
        filter_strategy=strategy,
        all_experiments=True,
        no_cache=no_cache,
        runners=runners,
        no_progress=no_progress,
        open_browser=open_browser,
    )


def _run_named_group(
    group_name: str,
    *,
    all_experiments: bool = False,
    repeats_override: int | None = None,
    no_cache: bool = False,
    runners: int = cli_defaults.DEFAULT_RUNNERS,
    open_browser: bool = False,
) -> None:
    """Execute a named :class:`~nvision.sim.run_groups.RunGroup` via :func:`nvision.cli.run.run`."""
    group = sim_run_groups.get_run_group(group_name)
    run(
        out=ARTIFACTS_ROOT,
        repeats=repeats_override if repeats_override is not None else 5,
        run_group=group.name,
        no_cache=no_cache,
        ignore_cache_strategy=None,
        all_experiments=all_experiments,
        runners=runners,
        open_browser=open_browser,
    )


@groups_app.command("list")
def list_groups(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Include one-line descriptions"),
) -> None:
    """List preset group names (from :func:`~nvision.sim.run_groups.run_groups`)."""
    for group in sim_run_groups.run_groups():
        if verbose and group.description:
            typer.echo(f"{group.name}\t{group.description}")
        else:
            typer.echo(group.name)


@groups_app.command("run")
def run_preset(
    group_name: Annotated[
        str,
        typer.Argument(
            ...,
            help="Preset group (same values as `nvision groups list`, e.g. all, sweep_only).",
        ),
    ],
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this run"),
    all_experiments: bool = typer.Option(
        False,
        "--all",
        help="Run full combination grid (disables category/strategy filtering)",
    ),
    no_cache: bool | None = typer.Option(
        None,
        "--no-cache/--cache",
        help="Cache mode override (default: no-cache for specific groups, cache for 'all').",
    ),
    runners: int = typer.Option(
        4,
        "--runners",
        min=1,
        help="Number of runner processes passed to `nvision run` for this group.",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open results in browser after run",
    ),
) -> None:
    """Run any registered preset group by name (single entry point for all groups)."""
    effective_no_cache = (group_name != "all") if no_cache is None else no_cache
    _run_named_group(
        group_name,
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache=effective_no_cache,
        runners=runners,
        open_browser=open_browser,
    )


# --- Top-level ``nvision run`` alias ------------------------------------------


@app.command()
def run_all(
    repeats: int = typer.Option(cli_defaults.DEFAULT_REPEATS, "--repeats", help="Number of repeats per scenario"),
    loc_max_steps: int = typer.Option(
        cli_defaults.DEFAULT_LOC_MAX_STEPS,
        "--loc-max-steps",
        help="Max steps for Bayesian locator measurement loop",
    ),
    loc_timeout_s: int = typer.Option(
        cli_defaults.DEFAULT_LOC_TIMEOUT_S,
        "--loc-timeout",
        help="Timeout in seconds for a single locator run",
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching for this run"),
    runners: int = typer.Option(
        cli_defaults.DEFAULT_RUNNERS,
        "--runners",
        min=1,
        help="Number of runner processes (use 1 for sequential execution).",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open results in browser after run",
    ),
) -> int:
    """Run all experiments (alias for ``nvision groups run all``)."""
    return run(
        out=ARTIFACTS_ROOT,
        repeats=repeats,
        loc_max_steps=loc_max_steps,
        loc_timeout_s=loc_timeout_s,
        run_group="all",
        no_cache=no_cache,
        runners=runners,
        open_browser=open_browser,
    )


# --- Shorthand aliases (same as ``groups run <name>``) -------------------------


@groups_app.command("sweep-only")
def sweep_only(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this run"),
    no_cache: bool = typer.Option(True, "--no-cache/--cache", help="Disable cache for this run"),
    runners: int = typer.Option(
        cli_defaults.DEFAULT_RUNNERS, "--runners", min=1, help="Number of runner processes passed to `nvision run`."
    ),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open results in browser after run"),
) -> None:
    """Alias for ``groups run sweep_only``."""
    _run_named_group(
        "sweep_only", repeats_override=repeats, no_cache=no_cache, runners=runners, open_browser=open_browser
    )


@groups_app.command("sweep-then-bayesian")
def sweep_then_bayesian(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this run"),
    no_cache: bool = typer.Option(True, "--no-cache/--cache", help="Disable cache for this run"),
    runners: int = typer.Option(
        cli_defaults.DEFAULT_RUNNERS, "--runners", min=1, help="Number of runner processes passed to `nvision run`."
    ),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open results in browser after run"),
) -> None:
    """Alias for ``groups run sweep-then-bayesian``."""
    _run_named_group(
        "sweep_then_bayesian", repeats_override=repeats, no_cache=no_cache, runners=runners, open_browser=open_browser
    )


@groups_app.command("bayesian-only")
def bayesian_only(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this run"),
    no_cache: bool = typer.Option(True, "--no-cache/--cache", help="Disable cache for this run"),
    runners: int = typer.Option(
        cli_defaults.DEFAULT_RUNNERS, "--runners", min=1, help="Number of runner processes passed to `nvision run`."
    ),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open results in browser after run"),
) -> None:
    """Alias for ``groups run bayesian-only``."""
    _run_named_group(
        "bayesian_only", repeats_override=repeats, no_cache=no_cache, runners=runners, open_browser=open_browser
    )


@groups_app.command("bayesian-clean")
def bayesian_clean(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this run"),
    no_cache: bool = typer.Option(True, "--no-cache/--cache", help="Disable cache for this run"),
    runners: int = typer.Option(
        cli_defaults.DEFAULT_RUNNERS, "--runners", min=1, help="Number of runner processes passed to `nvision run`."
    ),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open results in browser after run"),
) -> None:
    """Alias for ``groups run bayesian-clean``."""
    _run_named_group(
        "bayesian_clean", repeats_override=repeats, no_cache=no_cache, runners=runners, open_browser=open_browser
    )


@groups_app.command("demo")
def demo_group(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this run"),
    no_cache: bool = typer.Option(False, "--no-cache/--cache", help="Disable cache for this run"),
    runners: int = typer.Option(
        cli_defaults.DEFAULT_RUNNERS, "--runners", min=1, help="Number of runner processes passed to `nvision run`."
    ),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open results in browser after run"),
) -> None:
    """Alias for ``groups run demo``."""
    _run_named_group("demo", repeats_override=repeats, no_cache=no_cache, runners=runners, open_browser=open_browser)


@groups_app.command("smc-only")
def smc_only(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this run"),
    no_cache: bool = typer.Option(True, "--no-cache/--cache", help="Disable cache for this run"),
    runners: int = typer.Option(
        cli_defaults.DEFAULT_RUNNERS, "--runners", min=1, help="Number of runner processes passed to `nvision run`."
    ),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open results in browser after run"),
) -> None:
    """Alias for ``groups run smc-only``."""
    _run_named_group(
        "smc_only", repeats_override=repeats, no_cache=no_cache, runners=runners, open_browser=open_browser
    )
