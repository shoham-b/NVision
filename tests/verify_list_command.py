import shutil
from pathlib import Path

from typer.testing import CliRunner

from nvision.cli import app

runner = CliRunner()


def test_list_command():
    out_dir = Path("artifacts_list_test")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # 1. List empty
    print("\n--- List Empty Cache ---")
    result1 = runner.invoke(app, ["list", "--out", "artifacts_list_test"])
    print(result1.stdout)
    # It might say "No cache directory found" or "No cached combinations".

    # 2. Populate Cache
    print("\n--- Running Simulation to Populate Cache ---")
    runner.invoke(
        app,
        [
            "run",
            "--out",
            "artifacts_list_test",
            "--repeats",
            "1",
            "--loc-max-steps",
            "2",
            "--strategy",
            "NVCenter-SimpleSequential",  # Fast
            "--filter-category",
            "NVCenter",
        ],
    )

    # 3. List Populated Cache
    print("\n--- List Populated Cache ---")
    result2 = runner.invoke(app, ["list", "--out", "artifacts_list_test"])
    print(result2.stdout)

    # Assertions
    assert "NVCenter-SimpleSequential" in result2.stdout
    assert "NVCenter" in result2.stdout
    assert "Cached Simulations" in result2.stdout


if __name__ == "__main__":
    test_list_command()
