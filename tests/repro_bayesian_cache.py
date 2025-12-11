import shutil
import time
from pathlib import Path

from typer.testing import CliRunner

from nvision.cli import app

runner = CliRunner()


def test_bayesian_cache():
    cache_dir = Path("artifacts_bayesian_test/cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    out_dir = Path("artifacts_bayesian_test")
    # Clean output
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print("\n--- Run 1: Bayesian (Repeats=1) ---")
    start = time.time()
    result1 = runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_bayesian_test",
            "--repeats",
            "1",
            "--loc-max-steps",
            "5",  # Keep it short
            "--filter-category",
            "NVCenter",
            "--strategy",
            "NVCenter-SequentialBayesian",
            "--log-level",
            "DEBUG",
        ],
    )
    print(f"Run 1 completed in {time.time() - start:.2f}s")
    if "CACHE SAVE ERROR" in result1.stdout:
        print("!!! CACHE SAVE ERROR DETECTED !!!")
        print(result1.stdout)
    elif result1.exit_code != 0:
        print(result1.stdout)
        print(result1.stderr)
    assert result1.exit_code == 0

    # Verify cache files exist
    files = list(cache_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"Cache files found: {len(files)}")
    assert len(files) > 0, "No cache files created for Bayesian run!"

    print("\n--- Run 2: Bayesian (Repeats=1) - Should Hit Cache ---")
    start = time.time()
    result2 = runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_bayesian_test",
            "--repeats",
            "1",
            "--loc-max-steps",
            "5",
            "--filter-category",
            "NVCenter",
            "--strategy",
            "NVCenter-SequentialBayesian",
            "--log-level",
            "DEBUG",
        ],
    )
    print(f"Run 2 completed in {time.time() - start:.2f}s")
    assert result2.exit_code == 0

    # Check output for "Cache hit"
    if "Cache hit" in result2.stdout:
        print("SUCCESS: Cache hit confirmed.")
    else:
        print("FAILURE: Cache hit NOT found in logs.")
        print("Stdout snippet:")
        print(result2.stdout[:2000] + "...")


if __name__ == "__main__":
    test_bayesian_cache()
