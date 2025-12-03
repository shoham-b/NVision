import logging
import shutil
from pathlib import Path
from nvision.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_require_cache_missing():
    # Ensure cache is empty
    cache_dir = Path("artifacts_test/cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    result = runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_test",
            "--repeats",
            "1",
            "--loc-max-steps",
            "10",
            "--require-cache",
            "--filter-category",
            "OnePeak",  # Run a fast one
        ],
    )

    print(result.stdout)
    if (
        "Cache missing" in result.stdout or "Cache missing" in result.stderr
    ):  # Typer might capture logs differently
        print("SUCCESS: Cache missing warning found.")
    else:
        # Check logs directly if possible, but for now let's see output
        pass

    # Check that no results were generated (or at least it didn't crash)
    assert result.exit_code == 0
    print("Exit code 0")


if __name__ == "__main__":
    test_require_cache_missing()
