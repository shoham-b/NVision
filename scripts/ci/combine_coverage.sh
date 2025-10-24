#!/usr/bin/env bash
set -euo pipefail

if ! compgen -G ".coverage*" > /dev/null; then
  echo "No coverage files present; skipping combine/report."
  exit 0
fi

uv run coverage combine
uv run coverage xml
uv run coverage html
uv run coverage report --fail-under=85 | tee -a "$GITHUB_STEP_SUMMARY"
