#!/bin/bash
# Build the docs-site hero GIF (docs-site/static/img/openprocessor-workflow.gif).
#
# Frames, in order, each with a caption bar naming what it shows:
#   1. terminal API calls (detect, embed text, curation ingest status)
#   2. Swagger UI: the endpoint groups, then one endpoint executed
#   3. Triton model status (GET /models/)
#   4. Grafana dashboards with live data      (needs OP_DOCS_GRAFANA + login)
#   5. Prometheus scrape targets              (OP_DOCS_PROMETHEUS)
#   6. MLflow runs and a run comparison       (OP_DOCS_MLFLOW)
#   7. OpenSearch Dashboards index list       (OP_DOCS_OSD)
#   8. Cropwright closing frames, from the committed screenshots in
#      docs-site/static/img/screenshots/ (captured from Cropwright, the
#      labeling UI for this API)
#
# Frames 1-2 come from scripts/docs/capture_hero_frames.py, 3-7 from
# scripts/docs/capture_backend_screens.py. Both only ever read: point them
# at a stack holding PUBLIC sample data only. They refuse ports 4600-4799.
# Run `capture_backend_screens.py traffic` for a few minutes first so the
# Grafana panels have live data. Grafana frames need OP_GRAFANA_TOKEN or
# OP_GRAFANA_USER/OP_GRAFANA_PASSWORD in the environment; without them
# they are skipped.
#
# Usage:
#   OP_DOCS_API=http://localhost:<port> \
#   OP_DOCS_PROMETHEUS=http://localhost:<port> OP_DOCS_GRAFANA=http://localhost:<port> \
#   OP_DOCS_MLFLOW=http://localhost:<port> OP_DOCS_OSD=http://localhost:<port> \
#   PYTHON=.venv/bin/python scripts/docs/create-workflow-gif.sh
#
# Set KEEP_FRAMES=<dir> to keep the captioned frames for review.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHOTS="$REPO_ROOT/docs-site/static/img/screenshots"
OUTPUT="$REPO_ROOT/docs-site/static/img/openprocessor-workflow.gif"
PYTHON="${PYTHON:-python3}"
API="${OP_DOCS_API:?set OP_DOCS_API to a public-sample-data API base URL}"
# Palette size and per-frame delay scale keep the GIF under ~1.5 MB.
COLORS="${GIF_COLORS:-96}"

command -v convert >/dev/null || { echo "error: ImageMagick 'convert' is required" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"$PYTHON" "$SCRIPT_DIR/capture_hero_frames.py" --api "$API" --out "$WORK/hero" \
  --cache "$REPO_ROOT/cache/docs-hero"

backend_args=(capture --api "$API" --out "$WORK/backend")
[[ -n "${OP_DOCS_PROMETHEUS:-}" ]] && backend_args+=(--prometheus "$OP_DOCS_PROMETHEUS")
[[ -n "${OP_DOCS_GRAFANA:-}" ]] && backend_args+=(--grafana "$OP_DOCS_GRAFANA")
[[ -n "${OP_DOCS_MLFLOW:-}" ]] && backend_args+=(--mlflow "$OP_DOCS_MLFLOW")
[[ -n "${OP_DOCS_OSD:-}" ]] && backend_args+=(--osd "$OP_DOCS_OSD")
"$PYTHON" "$SCRIPT_DIR/capture_backend_screens.py" "${backend_args[@]}"

mkdir -p "$WORK/seq"
n=0
# add_frame <png> <caption> <delay-centiseconds>
add_frame() {
  local src="$1" caption="$2" delay="$3"
  [[ -f "$src" ]] || return 0
  n=$((n + 1))
  local dst
  dst="$(printf '%s/seq/%02d.png' "$WORK" "$n")"
  convert "$src" -crop 1600x1000+0+0 +repage \
    -fill '#09090be6' -draw 'rectangle 0,932 1600,1000' \
    -gravity south -fill '#f4f4f5' -font DejaVu-Sans -pointsize 30 \
    -annotate +0+18 "$caption" "$dst"
  echo "$delay" >"$dst.delay"
}

for f in "$WORK"/hero/*-terminal.png; do
  add_frame "$f" 'REST API: detect, embed and curation status (read-only calls)' 180
done
add_frame "$WORK/backend/swagger.png" 'Swagger UI: every endpoint group at /docs' 220
add_frame "$WORK/hero/"*-swagger-endpoint.png 'Swagger UI: try any endpoint in the browser' 200
add_frame "$WORK/hero/"*-swagger-response.png 'Swagger UI: the live response' 200
add_frame "$WORK/backend/models.png" 'Model status: every Triton model behind the API' 260
for f in "$WORK"/backend/grafana-*.png; do
  add_frame "$f" 'Grafana: live Triton, GPU and host dashboards' 300
done
add_frame "$WORK/backend/prometheus-targets.png" 'Prometheus: Triton, API, node, GPU and Loki targets' 240
add_frame "$WORK/backend/mlflow-experiments.png" 'MLflow: training runs logged by the trainer' 240
add_frame "$WORK/backend/mlflow-compare.png" 'MLflow: two runs compared, differences only' 260
add_frame "$WORK/backend/opensearch-indices.png" 'OpenSearch Dashboards: the curation indexes' 240
for name in clusters review; do
  src="$SHOTS/${name}-1600.png"
  [[ -f "$src" ]] || { echo "error: missing $src" >&2; exit 1; }
  add_frame "$src" 'Cropwright: the labeling UI for this API' 280
done

args=()
total_cs=0
for f in "$WORK"/seq/*.png; do
  d="$(cat "$f.delay")"
  args+=(-delay "$d" "$f")
  total_cs=$((total_cs + d))
done
if [[ -n "${KEEP_FRAMES:-}" ]]; then
  mkdir -p "$KEEP_FRAMES"
  cp "$WORK"/seq/*.png "$KEEP_FRAMES"/
fi

convert "${args[@]}" -resize 1280x800 -colors "$COLORS" -loop 0 -layers Optimize "$OUTPUT"

echo "GIF: $OUTPUT"
echo "  frames:   $n"
printf '  duration: %d.%02ds\n' "$((total_cs / 100))" "$((total_cs % 100))"
echo "  size:     $(du -h "$OUTPUT" | cut -f1)"
