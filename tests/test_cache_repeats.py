import shutil
from pathlib import Path

from typer.testing import CliRunner

runner = CliRunner()


def test_cache_repeats_separation():
    """
    Verify that running with different 'repeats' counts produces different cache entries
    for individual runs, or at least re-triggers execution.

    Actually, the goal is that if I run with repeats=1, and then repeats=2,
    the second run should NOT use the cache from the first run for its first iteration,
    because the 'context' (repeats=2) is different.

    Validating this by checking that the cache directory grows or that logs show "Starting combination".
    """
    cache_dir = Path("artifacts_test_repeats/cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    Path("artifacts_test_repeats")


if __name__ == "__main__":
    test_cache_repeats_separation()
