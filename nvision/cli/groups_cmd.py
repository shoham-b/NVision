"""CLI subcommands for preset run groups defined in ``nvision.sim.run_groups``."""

from __future__ import annotations

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
    noise: Annotated[str, typer.Argument(help="Noise name (e.g. NoNoise, Gauss(0.01))")],
    strategy: Annotated[str, typer.Argument(help="Strategy name (e.g. Bayesian-SBED-NoSweep)")],
    repeats: cli_options.RepeatsOption = 1,
    loc_timeout_s: cli_options.LocTimeoutOption = cli_defaults.DEFAULT_LOC_TIMEOUT_S,
    no_cache: cli_options.NoCacheOption = False,
    dry_run: cli_options.DryRunOption = False,
    runners: cli_options.RunnersOption = 1,
    no_progress: cli_options.NoProgressOption = False,
    open_browser: cli_options.OpenBrowserOption = False,
    gcp: cli_options.GcpOption = cli_defaults.DEFAULT_GCP,
    gcp_bucket: cli_options.GcpBucketOption = cli_defaults.DEFAULT_GCP_BUCKET,
) -> int:
    """Run a single (generator, noise, strategy) combination.

    Accepts any noise descriptor (e.g. ``Gauss(0.15)``, ``Poisson(5000)``) even
    if it is not in the active preset noise grid.  The noise is built dynamically
    from the descriptor string via :func:`~nvision.sim.combinations._parse_noise`.
    """
    # Use single_run=True which routes through resolve() for an exact triple lookup.
    # This supports any noise value (Gauss(sigma), Poisson(scale)) regardless
    # of the active .env grid configuration.
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
        dry_run=dry_run,
        runners=runners,
        no_progress=no_progress,
        open_browser=open_browser,
        gcp=gcp,
        gcp_bucket=gcp_bucket,
    )


def _run_named_group(
    group_name: str,
    *,
    all_experiments: bool = False,
    repeats_override: int | None = None,
    no_cache: bool = False,
    dry_run: bool = False,
    runners: int = cli_defaults.DEFAULT_RUNNERS,
    open_browser: bool = False,
    loc_timeout_s: int = cli_defaults.DEFAULT_LOC_TIMEOUT_S,
    no_progress: bool = False,
    gcp: bool = cli_defaults.DEFAULT_GCP,
    gcp_bucket: str | None = cli_defaults.DEFAULT_GCP_BUCKET,
) -> None:
    """Execute a named :class:`~nvision.sim.run_groups.RunGroup` via :func:`nvision.cli.run.run`."""
    group = sim_run_groups.get_run_group(group_name)
    run(
        out=ARTIFACTS_ROOT,
        repeats=repeats_override if repeats_override is not None else cli_defaults.DEFAULT_REPEATS,
        run_group=group.name,
        no_cache=no_cache,
        dry_run=dry_run,
        loc_timeout_s=loc_timeout_s,
        no_progress=no_progress,
        ignore_cache_strategy=None,
        all_experiments=all_experiments,
        runners=runners,
        open_browser=open_browser,
        gcp=gcp,
        gcp_bucket=gcp_bucket,
    )


@groups_app.command("list")
def list_groups(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Include one-line descriptions"),
) -> None:
    """List preset group names (from :func:`~nvision.sim.run_groups.run_groups`)."""
    for group in sim_run_groups.run_groups():
        group_cmd_name = group.name.replace("_", "-")
        if verbose and group.description:
            typer.echo(f"{group_cmd_name}\t{group.description}")
        else:
            typer.echo(group_cmd_name)


@groups_app.command("run")
def run_preset(
    group_name: Annotated[
        str,
        typer.Argument(
            ...,
            help="Preset group (same values as `nvision groups list`, e.g. all, sweep_only).",
        ),
    ],
    repeats: cli_options.RepeatsOption = cli_defaults.DEFAULT_REPEATS,
    all_experiments: bool = typer.Option(
        False,
        "--all",
        help="Run full combination grid (disables category/strategy filtering)",
    ),
    no_cache: cli_options.NoCacheOption = False,
    dry_run: cli_options.DryRunOption = False,
    runners: cli_options.RunnersOption = cli_defaults.DEFAULT_RUNNERS,
    open_browser: cli_options.OpenBrowserOption = cli_defaults.DEFAULT_OPEN_BROWSER,
    gcp: cli_options.GcpOption = cli_defaults.DEFAULT_GCP,
    gcp_bucket: cli_options.GcpBucketOption = cli_defaults.DEFAULT_GCP_BUCKET,
    loc_timeout_s: cli_options.LocTimeoutOption = cli_defaults.DEFAULT_LOC_TIMEOUT_S,
    no_progress: cli_options.NoProgressOption = False,
) -> None:
    """Run any registered preset group by name (single entry point for all groups)."""
    _run_named_group(
        group_name,
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache=no_cache,
        dry_run=dry_run,
        runners=runners,
        open_browser=open_browser,
        loc_timeout_s=loc_timeout_s,
        no_progress=no_progress,
        gcp=gcp,
        gcp_bucket=gcp_bucket,
    )


# --- Top-level ``nvision run`` alias ------------------------------------------


@app.command()
def run_all(
    repeats: cli_options.RepeatsOption = cli_defaults.DEFAULT_REPEATS,
    loc_timeout_s: cli_options.LocTimeoutOption = cli_defaults.DEFAULT_LOC_TIMEOUT_S,
    no_cache: cli_options.NoCacheOption = False,
    dry_run: cli_options.DryRunOption = False,
    runners: cli_options.RunnersOption = cli_defaults.DEFAULT_RUNNERS,
    open_browser: cli_options.OpenBrowserOption = cli_defaults.DEFAULT_OPEN_BROWSER,
    gcp: cli_options.GcpOption = cli_defaults.DEFAULT_GCP,
    gcp_bucket: cli_options.GcpBucketOption = cli_defaults.DEFAULT_GCP_BUCKET,
    no_progress: cli_options.NoProgressOption = False,
) -> int:
    """Run all experiments (alias for ``nvision groups run lorentzian-sbed``)."""
    return run(
        out=ARTIFACTS_ROOT,
        repeats=repeats,
        loc_timeout_s=loc_timeout_s,
        run_group="lorentzian-sbed",
        no_cache=no_cache,
        dry_run=dry_run,
        runners=runners,
        open_browser=open_browser,
        gcp=gcp,
        gcp_bucket=gcp_bucket,
        no_progress=no_progress,
    )


# --- Shorthand aliases (same as ``groups run <name>``) -------------------------


@groups_app.command("lorentzian-sbed")
def lorentzian_sbed(
    repeats: cli_options.RepeatsOption = cli_defaults.DEFAULT_REPEATS,
    no_cache: cli_options.NoCacheOption = False,
    dry_run: cli_options.DryRunOption = False,
    runners: cli_options.RunnersOption = cli_defaults.DEFAULT_RUNNERS,
    open_browser: cli_options.OpenBrowserOption = cli_defaults.DEFAULT_OPEN_BROWSER,
    gcp: cli_options.GcpOption = cli_defaults.DEFAULT_GCP,
    gcp_bucket: cli_options.GcpBucketOption = cli_defaults.DEFAULT_GCP_BUCKET,
    loc_timeout_s: cli_options.LocTimeoutOption = cli_defaults.DEFAULT_LOC_TIMEOUT_S,
    no_progress: cli_options.NoProgressOption = False,
) -> None:
    """Alias for ``groups run lorentzian-sbed``."""
    _run_named_group(
        "lorentzian-sbed",
        repeats_override=repeats,
        no_cache=no_cache,
        dry_run=dry_run,
        runners=runners,
        open_browser=open_browser,
        loc_timeout_s=loc_timeout_s,
        no_progress=no_progress,
        gcp=gcp,
        gcp_bucket=gcp_bucket,
    )
