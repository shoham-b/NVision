"""Shared CLI option definitions using Typer Annotated types."""

from typing import Annotated

import typer

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

PurgeOption = Annotated[
    bool,
    typer.Option(
        "--purge/--no-purge",
        help=(
            "Delete existing cache entries (and their on-disk plot artifacts) for every "
            "combination this run is about to touch, before any task starts. Unlike "
            "--no-cache, which only bypasses reading old entries and leaves them on disk, "
            "--purge actually removes them first -- use it to guarantee a truly clean slate "
            "instead of stale entries lingering under a combination the new run also writes."
        ),
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
        help="Start the results server (and open it in a browser) as soon as the run starts, "
        "so progress can be watched live instead of only after it finishes.",
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

RetryFailedOption = Annotated[
    bool,
    typer.Option(
        "--retry-failed",
        help=(
            "Only re-run combinations that failed with MemoryError in the most recent log. "
            "When combined with a group command, restricts to failures within that group."
        ),
    ),
]
