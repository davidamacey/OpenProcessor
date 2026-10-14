#!/bin/bash
# =============================================================================
# acceptance_run.sh - scripted end-to-end release acceptance run (issue #54)
# =============================================================================
# Drives an INSTALLED stack through every phase of
# docs/design/release_acceptance_plan.md that can be scripted: health, docs,
# a throwaway project, public-COCO ingest, cluster, VLM labeling, human
# confirm, holdout, export, train, bake-off, promote, inference, delete,
# metrics, log noise, VLM switch, installer lifecycle, teardown.
#
# Needs only bash, curl and jq. Writes a machine-readable JSON report and
# prints a human summary. Exit status is non-zero when any required phase
# failed. A cleanup trap ALWAYS deletes the throwaway project and the models
# this run promoted (and nothing else).
#
# Safety: the only docker calls are `docker compose -p <--project-name> ...`
# and `docker ps --filter label=com.docker.compose.project=<--project-name>`;
# no other compose project is ever addressed. Never point --base-url at a
# stack you care about: this run creates, trains and deletes a project.
#
# Public data only. Fetch ~100 license-filtered COCO images with
#   python3 scripts/datasets/fetch_coco_subset.py --out data/samples/coco_va
# and pass --images-dir data/samples/coco_va (any directory of JPEG/PNG works).
#
# Example (isolated install in ~/op-acc, compose project op-acc):
#   scripts/release/acceptance_run.sh --base-url http://localhost:4703 \
#     --project-name op-acc --install-dir ~/op-acc --models-dir ~/op-acc/models \
#     --project acc-v041 --images-dir data/samples/coco_va --report acc.json
# =============================================================================
# Option values are read by the sourced acceptance_lib.sh / acceptance_phases.sh.
# shellcheck disable=SC2034
set -uo pipefail

SCHEMA_VERSION=1
# name:required(1|0): run order. Optional phases never set a non-zero exit
# unless they were named in --only or --strict is given.
PHASES=(
    health:1 resource_links:1 docs_selfhosted:1 project_create:1 ingest:1
    detect_embed:1 cluster:1 vlm_label:1 confirm_labels:1 holdout:1 export:1
    train:1 bakeoff:0 promote:1 infer:1 model_delete:1 metrics:1 log_noise:1
    vlm_switch:0 installer_rerun:0 installer_repair:0 installer_upgrade:0
    installer_uninstall:0 teardown:1
)

BASE_URL="${OP_BASE_URL:-http://127.0.0.1:4603}"
API_PREFIX="/curation"
PROJECT_NAME=""
INSTALL_DIR=""
MODELS_DIR=""
SLUG="acc-$(date +%s)"
IMAGES_DIR=""
IMAGE_COUNT=100
VLM_CROPS=20
CONFIRM_CROPS=40
TRAIN_GPU="${ACC_TRAIN_GPU:-0}"
TRAIN_EPOCHS=2
TRAIN_TIMEOUT=3600
PHASE_TIMEOUT=900
HTTP_TIMEOUT=120
REPORT="acceptance-report.json"
ONLY=""
SKIP=""
STRICT=0
VLM_SWITCH_ID=""
VLM_SWITCH_FORCE=0
UPGRADE_VERSION=""
WORKDIR=""
LOG_REPEAT_MAX=25
POLL_S="${ACC_POLL_S:-5}"

usage() {
    sed -n '2,/^# =====.*$/p' "$0" | sed -n '2,32p' | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --base-url URL         API origin (default $OP_BASE_URL or http://127.0.0.1:4603)
  --project-name NAME    compose project of the stack under test (log/lifecycle phases)
  --install-dir DIR      installer-managed directory (enables installer_* phases)
  --models-dir DIR       host models dir (promote/delete verify the model dir appears/vanishes)
  --project SLUG         throwaway OpenProcessor project slug (default acc-<epoch>)
  --images-dir DIR       COCO (or any) JPEG/PNG directory for ingest
  --image-count N        images to ingest (default 100)
  --vlm-crops N          crops to VLM-label (default 20)
  --confirm-crops N      crops to human-confirm into one class for training (default 40; trainer floor is 20)
  --train-gpu IDS        CUDA_VISIBLE_DEVICES for the trainer (default $ACC_TRAIN_GPU or 0)
  --train-epochs N       probe-profile epochs (default 2)
  --train-timeout S      seconds to wait for the training run (default 3600)
  --phase-timeout S      seconds for other long polls (default 900)
  --only a,b             run only these phases (they are then all required)
  --skip a,b             skip these phases
  --vlm-switch-id ID     enable vlm_switch: `./openprocessor vlm use ID --yes`
  --vlm-switch-force     add --force to vlm use
  --upgrade-version vX   enable installer_upgrade (upgrade to vX, then --rollback)
  --strict               optional-phase failures also fail the run
  --report PATH          JSON report path (default acceptance-report.json)
  --workdir DIR          scratch/evidence directory (default: mktemp)
  --list                 print phase names and exit
EOF
}

die_usage() { echo "acceptance_run.sh: $*" >&2; exit 2; }

while (($#)); do
    case "$1" in
        --base-url) BASE_URL="${2:?}"; shift 2 ;;
        --project-name) PROJECT_NAME="${2:?}"; shift 2 ;;
        --install-dir) INSTALL_DIR="${2:?}"; shift 2 ;;
        --models-dir) MODELS_DIR="${2:?}"; shift 2 ;;
        --project) SLUG="${2:?}"; shift 2 ;;
        --images-dir) IMAGES_DIR="${2:?}"; shift 2 ;;
        --image-count) IMAGE_COUNT="${2:?}"; shift 2 ;;
        --vlm-crops) VLM_CROPS="${2:?}"; shift 2 ;;
        --confirm-crops) CONFIRM_CROPS="${2:?}"; shift 2 ;;
        --train-gpu) TRAIN_GPU="${2:?}"; shift 2 ;;
        --train-epochs) TRAIN_EPOCHS="${2:?}"; shift 2 ;;
        --train-timeout) TRAIN_TIMEOUT="${2:?}"; shift 2 ;;
        --phase-timeout) PHASE_TIMEOUT="${2:?}"; shift 2 ;;
        --only) ONLY="${2:?}"; shift 2 ;;
        --skip) SKIP="${2:?}"; shift 2 ;;
        --vlm-switch-id) VLM_SWITCH_ID="${2:?}"; shift 2 ;;
        --vlm-switch-force) VLM_SWITCH_FORCE=1; shift ;;
        --upgrade-version) UPGRADE_VERSION="${2:?}"; shift 2 ;;
        --strict) STRICT=1; shift ;;
        --report) REPORT="${2:?}"; shift 2 ;;
        --workdir) WORKDIR="${2:?}"; shift 2 ;;
        --list)
            for e in "${PHASES[@]}"; do echo "${e%%:*}"; done
            exit 0 ;;
        -h|--help) usage; exit 0 ;;
        *) die_usage "unknown option: $1" ;;
    esac
done

command -v curl >/dev/null || die_usage "curl is required"
command -v jq >/dev/null || die_usage "jq is required"
[[ "$SLUG" =~ ^[a-z0-9][a-z0-9-]{1,40}$ ]] || die_usage "--project must match [a-z0-9][a-z0-9-]{1,40}"
BASE_URL="${BASE_URL%/}"
for list in "$ONLY" "$SKIP"; do
    IFS=',' read -ra _names <<< "$list"
    for n in "${_names[@]}"; do
        [[ -z "$n" ]] && continue
        found=0
        for e in "${PHASES[@]}"; do [[ "${e%%:*}" == "$n" ]] && found=1; done
        ((found)) || die_usage "unknown phase: $n (see --list)"
    done
done

if [[ -z "$WORKDIR" ]]; then WORKDIR="$(mktemp -d)"; fi
mkdir -p "$WORKDIR/state" || die_usage "cannot create $WORKDIR"
PHASES_JSONL="$WORKDIR/phases.jsonl"
: > "$PHASES_JSONL"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"
PROJECT_API="${API_PREFIX}/projects/${SLUG}"

_ACC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/acceptance_lib.sh
source "${_ACC_DIR}/acceptance_lib.sh"
# shellcheck source=scripts/release/acceptance_phases.sh
source "${_ACC_DIR}/acceptance_phases.sh"


# ===========================================================================
# runner
# ===========================================================================
record() { # record NAME REQUIRED STATUS SECONDS MESSAGE
    local evjson
    evjson="$(jq -Rsc 'split("\n") | map(select(length > 0))' < "$EV_FILE" 2>/dev/null || echo '[]')"
    jq -nc --arg n "$1" --argjson r "$2" --arg s "$3" --argjson t "$4" --arg m "$5" --argjson e "$evjson" \
        '{name:$n, required:($r == 1), status:$s, seconds:$t, message:$m, evidence:$e}' >> "$PHASES_JSONL"
}

n_pass=0 n_fail=0 n_skip=0 req_failed=0
for entry in "${PHASES[@]}"; do
    name="${entry%%:*}"
    required="${entry##*:}"
    EV_FILE="$WORKDIR/ev.${name}"
    : > "$EV_FILE"
    if ! is_selected "$name"; then
        echo "SKIP: not selected" > "$EV_FILE"
        record "$name" "$required" skip 0 "not selected (--only)"
        n_skip=$((n_skip + 1))
        continue
    fi
    if is_skipped "$name"; then
        echo "SKIP: --skip" > "$EV_FILE"
        record "$name" "$required" skip 0 "skipped by --skip"
        n_skip=$((n_skip + 1))
        continue
    fi
    [[ -n "$ONLY" ]] && required=1
    t0=$(date +%s.%N)
    # background + wait so a signal interrupts a long poll and the trap runs at once
    ( "ph_${name}" ) >> "$WORKDIR/phase.${name}.out" 2>&1 &
    CHILD=$!
    wait "$CHILD"
    rc=$?
    CHILD=""
    secs="$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }')"
    last="$(tail -n 1 "$EV_FILE" 2>/dev/null | sed 's/^\(FAIL\|SKIP\): //')"
    case "$rc" in
        0)  status=pass;  n_pass=$((n_pass + 1)); last="ok" ;;
        77) status=skip;  n_skip=$((n_skip + 1)) ;;
        *)  status=fail;  n_fail=$((n_fail + 1))
            if ((required == 1 || STRICT == 1)); then req_failed=$((req_failed + 1)); fi
            [[ -s "$WORKDIR/phase.${name}.out" ]] && ev "output: $(tail -c 300 "$WORKDIR/phase.${name}.out" | tr '\n' ' ')" ;;
    esac
    record "$name" "$required" "$status" "$secs" "$last"
    printf '%-4s %-18s %6ss  %s\n' "$(echo "$status" | tr '[:lower:]' '[:upper:]')" "$name" "$secs" "$last"
done

# the trap deletes anything teardown did not (and is idempotent)
cleanup_resources || true

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EXIT_CODE=0
((req_failed > 0)) && EXIT_CODE=1
mkdir -p "$(dirname "$REPORT")"
jq -s --argjson sv "$SCHEMA_VERSION" --arg base "$BASE_URL" --arg proj "$SLUG" --arg cp "$PROJECT_NAME" \
    --arg s "$STARTED_AT" --arg f "$FINISHED_AT" --argjson secs "$(( $(date +%s) - START_EPOCH ))" \
    --argjson p "$n_pass" --argjson fl "$n_fail" --argjson sk "$n_skip" --argjson rf "$req_failed" \
    --argjson ec "$EXIT_CODE" --argjson cleaned "$([[ "$(st_get cleaned)" == 1 ]] && echo 1 || echo 0)" --arg ver "$(cat "$WORKDIR/state/version" 2>/dev/null || true)" \
    '{schema_version:$sv, base_url:$base, project:$proj, compose_project:$cp, started_at:$s, finished_at:$f,
      total_seconds:$secs, cleanup_complete:($cleaned == 1), exit_code:$ec,
      summary:{passed:$p, failed:$fl, skipped:$sk, required_failed:$rf}, phases:.}' \
    "$PHASES_JSONL" > "$REPORT" || { echo "acceptance_run.sh: cannot write report ${REPORT}" >&2; EXIT_CODE=1; }

echo
echo "acceptance: ${n_pass} passed, ${n_fail} failed (${req_failed} gating), ${n_skip} skipped; report: ${REPORT}"
((EXIT_CODE == 0)) && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit "$EXIT_CODE"
