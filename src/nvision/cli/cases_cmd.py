"""CLI subcommands for preset run cases defined in ``nvision.sim.cases``."""

from __future__ import annotations

import typer

from nvision.cli.main import app
from nvision.sim import cases as sim_cases
from nvision.tools.paths import ARTIFACTS_ROOT

cases_app = typer.Typer(
    help="Run preset simulation cases (see nvision.sim.cases.RunCase).",
    pretty_exceptions_show_locals=False,
)
app.add_typer(cases_app, name="cases")


def _run_named_case(
    case_name: str,
    *,
    all_experiments: bool = False,
    repeats_override: int | None = None,
    no_cache_override: bool | None = None,
) -> None:
    """Execute a named :class:`~nvision.sim.cases.RunCase` via :func:`nvision.cli.run.run`."""
    from nvision.cli.run import run

    case = sim_cases.get_run_case(case_name)
    # A preset with no filters (e.g. "all") should bypass run.py defaults.
    effective_all_experiments = all_experiments or (case.filter_category is None and case.filter_strategy is None)
    run(
        out=ARTIFACTS_ROOT,
        repeats=repeats_override if repeats_override is not None else case.repeats,
        loc_max_steps=case.loc_max_steps,
        loc_timeout_s=case.loc_timeout_s,
        no_cache=case.no_cache if no_cache_override is None else no_cache_override,
        ignore_cache_strategy=None,
        filter_category=case.filter_category,
        filter_strategy=case.filter_strategy,
        filter_generator=case.filter_generator,
        all_experiments=effective_all_experiments,
        no_progress=case.no_progress,
        require_cache=case.require_cache,
        log_level=case.log_level,
    )


@cases_app.command("list")
def list_cases(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Include one-line descriptions"),
) -> None:
    """List preset case names (from :func:`~nvision.sim.cases.run_cases`)."""
    for case in sim_cases.run_cases():
        if verbose and case.description:
            typer.echo(f"{case.name}\t{case.description}")
        else:
            typer.echo(case.name)


@cases_app.command("run")
def run_preset(
    case_name: str = typer.Argument(
        ...,
        help="Case name (including `all`): same as first column from `nvision cases list`",
    ),
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this run"),
    all_experiments: bool = typer.Option(
        False,
        "--all",
        help="Run full combination grid (disables category/strategy filtering)",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable cache for this run (overrides case default).",
    ),
) -> None:
    """Run any registered preset by name (single entry point for all cases)."""
    _run_named_case(
        case_name,
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache_override=True if no_cache else None,
    )


# --- Shorthand aliases (same as ``cases run <name>``) ---------------------------


@cases_app.command("nvcenter")
def nvcenter_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
    all_experiments: bool = typer.Option(False, "--all", help="Run full combination grid"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this run"),
) -> None:
    """Alias for ``cases run nvcenter``."""
    _run_named_case(
        "nvcenter",
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache_override=True if no_cache else None,
    )


@cases_app.command("nvcenter-bayes-eig")
def nvcenter_bayes_eig_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
    all_experiments: bool = typer.Option(False, "--all", help="Run full combination grid"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this run"),
) -> None:
    """Alias for ``cases run nvcenter_bayes_eig``."""
    _run_named_case(
        "nvcenter_bayes_eig",
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache_override=True if no_cache else None,
    )


@cases_app.command("nvcenter-bayes-ucb")
def nvcenter_bayes_ucb_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
    all_experiments: bool = typer.Option(False, "--all", help="Run full combination grid"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this run"),
) -> None:
    """Alias for ``cases run nvcenter_bayes_ucb``."""
    _run_named_case(
        "nvcenter_bayes_ucb",
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache_override=True if no_cache else None,
    )


@cases_app.command("nvcenter-bayes-maxvar")
def nvcenter_bayes_maxvar_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
    all_experiments: bool = typer.Option(False, "--all", help="Run full combination grid"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this run"),
) -> None:
    """Alias for ``cases run nvcenter_bayes_maxvar``."""
    _run_named_case(
        "nvcenter_bayes_maxvar",
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache_override=True if no_cache else None,
    )


@cases_app.command("nvcenter-bayes-utility")
def nvcenter_bayes_utility_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
    all_experiments: bool = typer.Option(False, "--all", help="Run full combination grid"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this run"),
) -> None:
    """Alias for ``cases run nvcenter_bayes_utility``."""
    _run_named_case(
        "nvcenter_bayes_utility",
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache_override=True if no_cache else None,
    )
