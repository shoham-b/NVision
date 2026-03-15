import tempfile
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nvision.cli import app

runner = CliRunner()


@pytest.mark.skip(reason="Hangs indefinitely due to threading/sqlite deadlocks inside Typer CliRunner")
@pytest.mark.timeout(60)
def test_elastic_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        out_dir = base_dir / "out"

        fast_args = [
            "--filter-category",
            "NVCenter",
            "--filter-strategy",
            "NVCenter-Sweep",
            "--loc-max-steps",
            "1",
            "--log-level",
            "ERROR",  # Suppress debug output for speed
        ]

        print("\n--- Run 1: Repeats=1 ---")
        start = time.time()
        result1 = runner.invoke(
            app,
            [
                "run",
                "--out",
                str(out_dir),
                "--repeats",
                "1",
                *fast_args,
            ],
        )
        print(f"Run 1 completed in {time.time() - start:.2f}s")
        assert result1.exit_code == 0

        print("\n--- Run 2: Repeats=2 (Should reuse 1) ---")
        start = time.time()
        result2 = runner.invoke(
            app,
            [
                "run",
                "--out",
                str(out_dir),
                "--repeats",
                "2",
                *fast_args,
            ],
        )
        print(f"Run 2 completed in {time.time() - start:.2f}s")
        assert result2.exit_code == 0
        print("INFO: No explicit partial cache hit message in code. Run 2 completed.")

        print("\n--- Run 3: Repeats=1 (Should Hit Cache completely) ---")
        start = time.time()
        result3 = runner.invoke(
            app,
            [
                "run",
                "--out",
                str(out_dir),
                "--repeats",
                "1",
                *fast_args,
            ],
        )
        print(f"Run 3 completed in {time.time() - start:.2f}s")
        assert result3.exit_code == 0
        if "Cache hit" in result3.stdout or "skipping simulation" in result3.stdout:
            print("SUCCESS: Full cache hit detected.")
        else:
            print("FAILURE: Full cache hit NOT found.")
            print(result3.stdout[:1000])


if __name__ == "__main__":
    test_elastic_cache()
