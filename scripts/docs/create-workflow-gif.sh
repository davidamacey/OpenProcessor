#!/bin/bash
# Build the docs-site hero GIF (docs-site/static/img/openprocessor-workflow.gif).
#
# 1. scripts/docs/capture_hero_frames.py captures same-size 1600x1000
#    frames from a running API holding PUBLIC sample data only: a terminal
#    session (detect, embed text, curation ingest status), then the
#    Swagger UI. Every call is read-only.
# 2. Two closing frames come from the committed public-data screenshots in
#    docs-site/static/img/screenshots/ (captured from Cropwright, the
#    labeling UI for this API), cropped to the same 1600x1000 viewport.
# 3. ImageMagick stitches everything, scaled to 1280 wide.
#
# Usage:
#   OP_DOCS_API=http://localhost:<port> PYTHON=.venv/bin/python \
#     scripts/docs/create-workflow-gif.sh
#
# Set OP_DOCS_MLFLOW=<url> to add an MLflow frame (only once it has runs).
# Set KEEP_FRAMES=<dir> to keep the individual frames for review.
#
# Never point OP_DOCS_API at a deployment with real data; the capture
# script refuses ports 4600-4799.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHOTS="$REPO_ROOT/docs-site/static/img/screenshots"
OUTPUT="$REPO_ROOT/docs-site/static/img/openprocessor-workflow.gif"
PYTHON="${PYTHON:-python3}"
API="${OP_DOCS_API:?set OP_DOCS_API to a public-sample-data API base URL}"

command -v convert >/dev/null || { echo "error: ImageMagick 'convert' is required" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"$PYTHON" "$SCRIPT_DIR/capture_hero_frames.py" --api "$API" --out "$WORK/frames" \
  --cache "$REPO_ROOT/cache/docs-hero" ${OP_DOCS_MLFLOW:+--mlflow "$OP_DOCS_MLFLOW"}

# Closing frames: the labeling UI, captioned.
CLOSING=("clusters" "review")
for name in "${CLOSING[@]}"; do
  src="$SHOTS/${name}-1600.png"
  [[ -f "$src" ]] || { echo "error: missing $src" >&2; exit 1; }
  convert "$src" -crop 1600x1000+0+0 +repage \
    -fill '#09090bdd' -draw 'rectangle 0,924 1600,1000' \
    -gravity south -fill '#f4f4f5' -font DejaVu-Sans -pointsize 34 \
    -annotate +0+20 'Cropwright: the labeling UI for this API' \
    "$WORK/frames/99-${name}.png"
done

# Per-frame delay in centiseconds, by frame-name pattern.
delay_for() {
  case "$1" in
    *terminal*) echo 220 ;;
    *swagger-endpoint* | *swagger-response*) echo 260 ;;
    *swagger*) echo 200 ;;
    *mlflow*) echo 220 ;;
    *) echo 260 ;;
  esac
}

args=()
total_cs=0
count=0
for f in "$WORK"/frames/*.png; do
  d="$(delay_for "$(basename "$f")")"
  args+=(-delay "$d" "$f")
  total_cs=$((total_cs + d))
  count=$((count + 1))
done
if [[ -n "${KEEP_FRAMES:-}" ]]; then
  mkdir -p "$KEEP_FRAMES"
  cp "$WORK"/frames/*.png "$KEEP_FRAMES"/
fi

convert "${args[@]}" -resize 1280x -colors 128 -loop 0 -layers Optimize "$OUTPUT"

echo "GIF: $OUTPUT"
echo "  frames:   $count"
printf '  duration: %d.%02ds\n' "$((total_cs / 100))" "$((total_cs % 100))"
echo "  size:     $(du -h "$OUTPUT" | cut -f1)"
