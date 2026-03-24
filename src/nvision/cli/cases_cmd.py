"""CLI subcommands for preset run cases defined in ``nvision.sim.cases``."""

from __future__ import annotations

from typing import Annotated

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
    case_name: sim_cases.RunCaseName,
    *,
    all_experiments: bool = False,
    repeats_override: int | None = None,
    no_cache: bool = False,
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
        no_cache=no_cache,
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
    case_name: Annotated[
        sim_cases.RunCaseName,
        typer.Argument(
            ...,
            help="Preset case (same values as `nvision cases list`, e.g. all, nvcenter).",
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
        help="Cache mode override (default: no-cache for specific cases, cache for 'all').",
    ),
) -> None:
    """Run any registered preset by name (single entry point for all cases)."""
    effective_no_cache = (case_name != sim_cases.RunCaseName.ALL) if no_cache is None else no_cache
    _run_named_case(
        case_name,
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache=effective_no_cache,
    )


# --- Shorthand aliases (same as ``cases run <name>``) ---------------------------


@cases_app.command("nvcenter")
def nvcenter_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
    all_experiments: bool = typer.Option(False, "--all", help="Run full combination grid"),
    no_cache: bool = typer.Option(True, "--no-cache/--cache", help="Disable cache for this run"),
) -> None:
    """Alias for ``cases run nvcenter``."""
    _run_named_case(
        sim_cases.RunCaseName.NVCENTER,
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache=no_cache,
    )


@cases_app.command("nvcenter-bayes-sbed")
def nvcenter_bayes_sbed_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
    all_experiments: bool = typer.Option(False, "--all", help="Run full combination grid"),
    no_cache: bool = typer.Option(True, "--no-cache/--cache", help="Disable cache for this run"),
) -> None:
    """Alias for ``cases run nvcenter_bayes_sbed``."""
    _run_named_case(
        sim_cases.RunCaseName.NVCENTER_BAYES_SBED,
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache=no_cache,
    )


@cases_app.command("nvcenter-bayes-ucb")
def nvcenter_bayes_ucb_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
    all_experiments: bool = typer.Option(False, "--all", help="Run full combination grid"),
    no_cache: bool = typer.Option(True, "--no-cache/--cache", help="Disable cache for this run"),
) -> None:
    """Alias for ``cases run nvcenter_bayes_ucb``."""
    _run_named_case(
        sim_cases.RunCaseName.NVCENTER_BAYES_UCB,
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache=no_cache,
    )


@cases_app.command("nvcenter-bayes-maxvar")
def nvcenter_bayes_maxvar_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
    all_experiments: bool = typer.Option(False, "--all", help="Run full combination grid"),
    no_cache: bool = typer.Option(True, "--no-cache/--cache", help="Disable cache for this run"),
) -> None:
    """Alias for ``cases run nvcenter_bayes_maxvar``."""
    _run_named_case(
        sim_cases.RunCaseName.NVCENTER_BAYES_MAXVAR,
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache=no_cache,
    )


@cases_app.command("nvcenter-bayes-utility")
def nvcenter_bayes_utility_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
    all_experiments: bool = typer.Option(False, "--all", help="Run full combination grid"),
    no_cache: bool = typer.Option(True, "--no-cache/--cache", help="Disable cache for this run"),
) -> None:
    """Alias for ``cases run nvcenter_bayes_utility``."""
    _run_named_case(
        sim_cases.RunCaseName.NVCENTER_BAYES_UTILITY,
        all_experiments=all_experiments,
        repeats_override=repeats,
        no_cache=no_cache,
    )


if __name__ == "__main__":
    run_preset(
        case_name=sim_cases.RunCaseName.NVCENTER_BAYES_SBED,
        repeats=5,
        all_experiments=True,
        no_cache=True,
    )
