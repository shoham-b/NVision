import shutil
import time
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from nvision.cli import app

runner = CliRunner()


def test_elastic_steps():
    cache_dir = Path("artifacts_steps_test/cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    out_dir = Path("artifacts_steps_test")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # 1. Run with 10 steps
    print("\n--- Run 1: Steps=10, Repeats=1 ---")
    start = time.time()
    result1 = runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_steps_test",
            "--repeats",
            "1",
            "--loc-max-steps",
            "10",
            "--strategy",
            "NVCenter-SimpleSequential",  # Supports it now because we added tracking to Base (hopefully)
            "--filter-category",
            "NVCenter",
            "--log-level",
            "DEBUG",
        ],
    )
    print(f"Run 1 completed in {time.time() - start:.2f}s")
    if result1.exit_code != 0:
        print("Run 1 FAILED. Output:")
        print(result1.stdout)
    assert result1.exit_code == 0

    # Verify measurements in CSV (should be 10)
    res_csv = out_dir / "locator_results.csv"
    if res_csv.exists():
        df = pl.read_csv(res_csv)
        print("Run 1 measurements:", df.get_column("measurements").to_list())

    # 2. Run with 5 steps (Should HIT 10 and SUBSET)
    print("\n--- Run 2: Steps=5 (Should reuse 10) ---")
    start = time.time()
    result2 = runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_steps_test",
            "--repeats",
            "1",
            "--loc-max-steps",
            "5",
            "--strategy",
            "NVCenter-SimpleSequential",
            "--filter-category",
            "NVCenter",
            "--log-level",
            "DEBUG",
        ],
    )
    print(f"Run 2 completed in {time.time() - start:.2f}s")
    if result2.exit_code != 0:
        print("Run 2 FAILED. Output:")
        print(result2.stdout)
    assert result2.exit_code == 0

    if "Cache hit" in result2.stdout:
        print("SUCCESS: Cache hit detected.")
    else:
        print("FAILURE: Cache hit NOT found.")
        print(result2.stdout[:1000])

    if res_csv.exists():
        df = pl.read_csv(res_csv)
        print("Run 2 measurements:", df.get_column("measurements").to_list())

    # 3. Run with 20 steps (Should MISS 10 and RERUN)
    print("\n--- Run 3: Steps=20 (Should MISS 10) ---")
    result3 = runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_steps_test",
            "--repeats",
            "1",
            "--loc-max-steps",
            "20",
            "--strategy",
            "NVCenter-SimpleSequential",
            "--filter-category",
            "NVCenter",
            "--log-level",
            "DEBUG",
        ],
    )
    if result3.exit_code != 0:
        print("Run 3 FAILED. Output:")
        print(result3.stdout)
    assert result3.exit_code == 0
    if "Cache hit" not in result3.stdout:  # Expect miss
        print("SUCCESS: Cache miss detected as expected.")
    else:
        print("FAILURE: Unexpected Cache HIT.")

    # 4. Run with 10 steps (Should HIT 20)
    print("\n--- Run 4: Steps=10 (Should reuse 20) ---")
    result4 = runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_steps_test",
            "--repeats",
            "1",
            "--loc-max-steps",
            "10",
            "--strategy",
            "NVCenter-SimpleSequential",
            "--filter-category",
            "NVCenter",
            "--log-level",
            "DEBUG",
        ],
    )
    if result4.exit_code != 0:
        print("Run 4 FAILED. Output:")
        print(result4.stdout)
    assert result4.exit_code == 0
    if "Cache hit" in result4.stdout:
        print("SUCCESS: Cache hit detected from larger set.")
    else:
        print("FAILURE: Cache hit NOT found from larger set.")


if __name__ == "__main__":
    test_elastic_steps()
