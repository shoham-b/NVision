#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-artifacts}
GRAPHS_DIR="$OUT_DIR/graphs"
TEMPLATE=${2:-scripts/pages_index.html}

# Generate graphs via project CLI
uv run python -m nvision --out "$OUT_DIR" --repeats 5 --seed 123 --loc-max-steps 150

# Build gallery entries
mkdir -p "$GRAPHS_DIR"
tmp_gallery="$GRAPHS_DIR/.gallery.html"
: > "$tmp_gallery"
shopt -s nullglob
for f in "$GRAPHS_DIR"/*; do
  [ -f "$f" ] || continue
  name=$(basename "$f")
  ext=${name##*.}
  case "${ext,,}" in
    png|jpg|jpeg|gif|svg|webp)
      echo "<figure><a href=\"$name\"><img src=\"$name\" alt=\"$name\"></a><figcaption>$name</figcaption></figure>" >> "$tmp_gallery" ;;
    *)
      echo "<p><a href=\"$name\">$name</a></p>" >> "$tmp_gallery" ;;
  esac
done

# Render template by injecting the gallery into {{GALLERY}}
awk '{
  if ($0 ~ /\{\{GALLERY\}\}/) {
    while ((getline line < ARGV[1]) > 0) print line; next
  } print
}' "$tmp_gallery" "$TEMPLATE" > "$GRAPHS_DIR/index.html"
