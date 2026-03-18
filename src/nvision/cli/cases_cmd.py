from __future__ import annotations

import typer

from nvision.cli.main import app
from nvision.core.paths import ARTIFACTS_ROOT
from nvision.sim import cases as sim_cases

cases_app = typer.Typer(
    help="Run specific preset simulation cases (no parameters).",
    pretty_exceptions_show_locals=False,
)
app.add_typer(cases_app, name="cases")


def _run_named_case(
    case_name: str,
    *,
    all_experiments: bool = False,
    repeats_override: int | None = None,
) -> None:
    """Run a named preset case."""
    from nvision.cli.run import run

    case = sim_cases.get_run_case(case_name)
    run(
        out=ARTIFACTS_ROOT,
        repeats=repeats_override if repeats_override is not None else case.repeats,
        seed=case.seed,
        loc_max_steps=case.loc_max_steps,
        loc_timeout_s=case.loc_timeout_s,
        no_cache=case.no_cache,
        ignore_cache_strategy=None,
        filter_category=case.filter_category,
        filter_strategy=case.filter_strategy,
        all_experiments=all_experiments,
        no_progress=case.no_progress,
        require_cache=case.require_cache,
        log_level=case.log_level,
    )


@cases_app.command(name="list")
def list_cases() -> None:
    """List available preset cases."""
    for case in sim_cases.run_cases():
        typer.echo(case.name)


@cases_app.command(name="nvcenter")
def nvcenter_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
) -> None:
    """Run the NVCenter sweep preset with built-in parameters."""
    _run_named_case("nvcenter", repeats_override=repeats)


@cases_app.command(name="nvcenter-bayes-eig")
def nvcenter_bayes_eig_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
) -> None:
    """Run the NVCenter Bayesian EIG preset."""
    _run_named_case("nvcenter_bayes_eig", repeats_override=repeats)


@cases_app.command(name="nvcenter-bayes-ucb")
def nvcenter_bayes_ucb_case(
    repeats: int | None = typer.Option(None, "--repeats", help="Override repeats for this case"),
) -> None:
    """Run the NVCenter Bayesian UCB preset."""
    _run_named_case("nvcenter_bayes_ucb", repeats_override=repeats)


if __name__ == "__main__":
    nvcenter_case()
