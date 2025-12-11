import shutil
import time
from pathlib import Path

from typer.testing import CliRunner

from nvision.cli import app

runner = CliRunner()


def test_elastic_cache():
    cache_dir = Path("artifacts_elastic_test/cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    out_dir = Path("artifacts_elastic_test")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print("\n--- Run 1: Repeats=2 ---")
    start = time.time()
    result1 = runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_elastic_test",
            "--repeats",
            "2",
            "--loc-max-steps",
            "2",
            "--log-level",
            "DEBUG",
        ],
    )
    print(f"Run 1 completed in {time.time() - start:.2f}s")
    assert result1.exit_code == 0

    print("\n--- Run 2: Repeats=4 (Should reuse 2) ---")
    # This should be faster than running 4 from scratch, but slower than 2 from scratch.
    # It should say "Partial cache hit... found 2/4"
    start = time.time()
    result2 = runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_elastic_test",
            "--repeats",
            "4",
            "--loc-max-steps",
            "2",
            "--log-level",
            "DEBUG",
        ],
    )
    print(f"Run 2 completed in {time.time() - start:.2f}s")
    assert result2.exit_code == 0
    if "Partial cache hit" in result2.stdout and "found 2/4" in result2.stdout:
        print("SUCCESS: Partial cache hit detected.")
    else:
        print("FAILURE: Partial cache hit NOT found.")
        print(result2.stdout[:1000])

    print("\n--- Run 3: Repeats=2 (Should Hit Cache completely) ---")
    # Cache now has 4 results. Requesting 2 should match fully.
    # "Cache hit ... skipping simulation"
    start = time.time()
    result3 = runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_elastic_test",
            "--repeats",
            "2",
            "--loc-max-steps",
            "2",
            "--log-level",
            "DEBUG",
        ],
    )
    print(f"Run 3 completed in {time.time() - start:.2f}s")
    assert result3.exit_code == 0
    if "Cache hit" in result3.stdout:
        print("SUCCESS: Full cache hit detected.")
    else:
        print("FAILURE: Full cache hit NOT found.")
        print(result3.stdout[:1000])

    print("\n--- Run 4: Repeats=1 (Should Hit Cache completely) ---")
    start = time.time()
    result4 = runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_elastic_test",
            "--repeats",
            "1",
            "--loc-max-steps",
            "2",
            "--log-level",
            "DEBUG",
        ],
    )
    print(f"Run 4 completed in {time.time() - start:.2f}s")
    assert result4.exit_code == 0
    if "Cache hit" in result4.stdout:
        print("SUCCESS: Full cache hit detected for subset.")
    else:
        print("FAILURE: Full cache hit NOT found for subset.")


if __name__ == "__main__":
    test_elastic_cache()
