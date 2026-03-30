import sys

try:
    print("Importing nvision.cli.main...")
    from nvision.cli.main import app

    print("App imported successfully.")

    print("Calling app() with ['run', '--help']...")
    # Mock sys.argv for Typer
    sys.argv = ["nvision", "run", "--help"]
    app()
    print("app() finished successfully.")

except SystemExit as e:
    print(f"app() exited with code: {e.code}")
except Exception:
    import traceback

    print("\n--- CRASH DETECTED ---", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
