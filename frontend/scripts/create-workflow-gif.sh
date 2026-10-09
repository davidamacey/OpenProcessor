#!/bin/bash
# Build the docs-site hero GIF: a loop through Cropwright's pages, one
# frame per capability, from the public-sample-data screenshots under
# docs-site/static/img/screenshots/ (see scripts/capture_docs_screenshots.py).
# Those captures are full-page, so each frame is cropped to the top
# 1600x1000 viewport first; every frame then has the same size.
#
# Usage: scripts/create-workflow-gif.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SHOTS="$REPO_ROOT/docs-site/static/img/screenshots"
OUTPUT="$REPO_ROOT/docs-site/static/img/cropwright-workflow.gif"

command -v convert >/dev/null || { echo "error: ImageMagick 'convert' is required" >&2; exit 1; }

# frame name -> delay in centiseconds, in workflow order.
FRAMES=(
  "dashboard:250"
  "ingest:180"
  "clusters:280"
  "review:300"
  "review-regions:300"
  "classes:200"
  "export:220"
  "train:250"
  "bakeoff:250"
  "settings:200"
)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

args=()
total_cs=0
for entry in "${FRAMES[@]}"; do
  name="${entry%%:*}"
  delay="${entry##*:}"
  src="$SHOTS/${name}-1600.png"
  [[ -f "$src" ]] || { echo "error: missing $src" >&2; exit 1; }
  convert "$src" -crop 1600x1000+0+0 +repage "$WORK/${name}.png"
  args+=(-delay "$delay" "$WORK/${name}.png")
  total_cs=$((total_cs + delay))
done

convert "${args[@]}" -resize 1280x -loop 0 -layers Optimize "$OUTPUT"

echo "GIF: $OUTPUT"
echo "  frames:   ${#FRAMES[@]}"
printf '  duration: %d.%02ds\n' "$((total_cs / 100))" "$((total_cs % 100))"
echo "  size:     $(du -h "$OUTPUT" | cut -f1)"
