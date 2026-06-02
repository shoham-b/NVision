"""Shared CLI option definitions using Typer Annotated types."""

from typing import Annotated
import typer
from nvision.cli import defaults as cli_defaults

RepeatsOption = Annotated[
    int,
    typer.Option(
        "--repeats",
        help="Number of simulation repeats per scenario",
    ),
]

LocMaxStepsOption = Annotated[
    int,
    typer.Option(
        "--loc-max-steps",
        help="Max steps for Bayesian locator measurement loop",
    ),
]

LocTimeoutOption = Annotated[
    int,
    typer.Option(
        "--loc-timeout",
        help="Timeout in seconds for a single locator run",
    ),
]

NoCacheOption = Annotated[
    bool,
    typer.Option(
        "--no-cache",
        help="Disable caching for this run",
    ),
]

DryRunOption = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Do not write results to cache",
    ),
]

RunnersOption = Annotated[
    int,
    typer.Option(
        "--runners",
        min=1,
        help="Number of runner processes (use 1 for sequential execution).",
    ),
]

OpenBrowserOption = Annotated[
    bool,
    typer.Option(
        "--open/--no-open",
        help="Open results in browser after run",
    ),
]

GcpOption = Annotated[
    bool,
    typer.Option(
        "--gcp/--no-gcp",
        help="Upload results to GCP after run",
    ),
]

GcpBucketOption = Annotated[
    str | None,
    typer.Option(
        "--gcp-bucket",
        help="GCP bucket to upload results to",
    ),
]

NoProgressOption = Annotated[
    bool,
    typer.Option(
        "--no-progress",
        help="Disable progress bars / Rich progress UI",
    ),
]
