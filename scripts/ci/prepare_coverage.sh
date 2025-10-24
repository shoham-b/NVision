#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "coverage-data" ]; then
  echo "No coverage artifacts downloaded; skipping aggregation."
  exit 0
fi

# Collect .coverage files from downloaded artifacts into .coverage.N files
mapfile -t files < <(find coverage-data -type f -name ".coverage" || true)
if [ ${#files[@]} -eq 0 ]; then
  echo "No .coverage files found in artifacts; skipping aggregation."
  exit 0
fi

i=0
for f in "${files[@]}"; do
  cp "$f" ".coverage.$i"
  i=$((i+1))
done
ls -la .coverage* || true
