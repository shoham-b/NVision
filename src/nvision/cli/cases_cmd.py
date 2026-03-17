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


@cases_app.command(name="nvcenter")
def nvcenter_case() -> None:
    """Run the NVCenter preset with built-in parameters."""
    from nvision.cli.run import run

    case = sim_cases.get_run_case("nvcenter")
    run(
        out=ARTIFACTS_ROOT,
        repeats=case.repeats,
        seed=case.seed,
        loc_max_steps=case.loc_max_steps,
        loc_timeout_s=case.loc_timeout_s,
        no_cache=case.no_cache,
        ignore_cache_strategy=None,
        filter_category=case.filter_category,
        filter_strategy=case.filter_strategy,
        all_experiments=False,
        no_progress=case.no_progress,
        require_cache=case.require_cache,
        log_level=case.log_level,
    )


if __name__ == "__main__":
    nvcenter_case()
