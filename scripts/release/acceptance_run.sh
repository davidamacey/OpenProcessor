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

# ---------------------------------------------------------------------------
# helpers: state, evidence, HTTP, polling
# ---------------------------------------------------------------------------
st_set() { printf '%s' "$2" > "$WORKDIR/state/$1"; }
st_get() { [[ -f "$WORKDIR/state/$1" ]] && cat "$WORKDIR/state/$1" || true; }
st_add() { printf '%s\n' "$2" >> "$WORKDIR/state/$1"; }

EV_FILE=""
ev() { printf '%s\n' "$*" >> "$EV_FILE"; }
fail() { ev "FAIL: $*"; exit 1; }
skip() { ev "SKIP: $*"; exit 77; }

HTTP_CODE=000
BODY=""
# api METHOD PATH [extra curl args...]: sets HTTP_CODE and BODY (a file).
api() {
    local method="$1" path="$2"
    shift 2
    BODY="$WORKDIR/body.$RANDOM$RANDOM"
    HTTP_CODE="$(curl -sS -o "$BODY" -w '%{http_code}' --max-time "$HTTP_TIMEOUT" \
        -X "$method" "$@" "${BASE_URL}${path}" 2>>"$WORKDIR/curl.err")" || HTTP_CODE=000
}
api_json() { api "$1" "$2" -H 'Content-Type: application/json' --data "${3:-{\}}"; }
snippet() { head -c 300 "$BODY" 2>/dev/null | tr '\n' ' '; }
# expect CODE...: fail unless HTTP_CODE is one of the listed codes
expect() {
    local c
    for c in "$@"; do [[ "$HTTP_CODE" == "$c" ]] && return 0; done
    fail "HTTP ${HTTP_CODE} (wanted $*): $(snippet)"
}
jqb() { jq -r "$1" "$BODY" 2>/dev/null; }
# poll TIMEOUT_S CMD...: run CMD every POLL_S until it returns 0.
poll() {
    local timeout="$1" deadline
    shift
    deadline=$(( $(date +%s) + timeout ))
    while true; do
        "$@" && return 0
        (( $(date +%s) >= deadline )) && return 1
        sleep "$POLL_S"
    done
}

is_selected() {
    local n="$1"
    if [[ -n "$ONLY" ]]; then
        [[ ",${ONLY}," == *",${n},"* ]] || return 1
    fi
    return 0
}
is_skipped() { [[ ",${SKIP}," == *",$1,"* ]]; }

# ---------------------------------------------------------------------------
# cleanup: always delete the throwaway project and the models this run promoted
# ---------------------------------------------------------------------------
cleanup_resources() {
    [[ "$(st_get cleaned)" == 1 ]] && return 0
    local rc=0 m
    local models
    models="$(st_get promoted_models)"
    if [[ -n "$models" ]]; then
        while IFS= read -r m; do
            [[ -z "$m" ]] && continue
            api DELETE "${PROJECT_API}/models/${m}?force=true"
            case "$HTTP_CODE" in 200|404) ;; *) rc=1 ;; esac
        done <<< "$models"
    fi
    if [[ "$(st_get project_created)" == 1 ]]; then
        api DELETE "${PROJECT_API}?confirm=${SLUG}&force=true"
        case "$HTTP_CODE" in
            200|404) ;;
            *)
                # a project must be archived-or-idle to delete; one retry via archive
                api POST "${PROJECT_API}/archive"
                api DELETE "${PROJECT_API}?confirm=${SLUG}&force=true"
                case "$HTTP_CODE" in 200|404) ;; *) rc=1 ;; esac
                ;;
        esac
    fi
    ((rc == 0)) && st_set cleaned 1
    return "$rc"
}

on_exit() {
    local code=$?
    trap - EXIT INT TERM
    cleanup_resources || echo "acceptance_run.sh: WARNING: cleanup of project '${SLUG}' did not complete" >&2
    exit "$code"
}
CHILD=""
on_signal() {
    echo "acceptance_run.sh: interrupted" >&2
    if [[ -n "$CHILD" ]]; then
        pkill -TERM -P "$CHILD" 2>/dev/null || true
        kill -TERM "$CHILD" 2>/dev/null || true
    fi
    exit 130
}
trap on_exit EXIT
trap on_signal INT TERM

# ===========================================================================
# phases. Each returns 0 pass, 1 fail (via fail), 77 skip (via skip).
# ===========================================================================
ph_health() {
    api GET /health; expect 200
    ev "GET /health -> 200 version=$(jqb '.version // "?"')"
    api GET "${API_PREFIX}/health"; expect 200
    ev "GET ${API_PREFIX}/health -> 200"
}

ph_resource_links() {
    api GET "${API_PREFIX}/projects/default/settings"; expect 200
    local n bad
    n="$(jqb '(.resource_links // []) | length')"
    [[ "$n" =~ ^[0-9]+$ && "$n" -gt 0 ]] || fail "settings served no resource_links"
    ev "resource_links: ${n} entries"
    bad="$(jqb '[.resource_links[] | select(.kind == "docs") | select((.url // "") | startswith("/") | not) | .url] | join(",")')"
    [[ -z "$bad" ]] || fail "docs links must be path-relative, got: ${bad}"
    ev "docs links are path-relative"
}

ph_docs_selfhosted() {
    local p assets ext host
    for p in /docs /redoc; do
        api GET "$p"; expect 200
        # any absolute http(s) URL in src/href that is not this origin is external
        host="${BASE_URL#*://}"
        ext="$(grep -Eo '(src|href)="https?://[^"]+"' "$BODY" | grep -Fv "$host" | head -3 | tr '\n' ' ')"
        [[ -z "$ext" ]] || fail "$p references external assets: ${ext}"
        ev "$p -> 200, no external URLs"
        assets="$(grep -Eo '(src|href)="/docs-assets/[^"]+"' "$BODY" | sed -E 's/^[a-z]+="//;s/"$//' | sort -u)"
        local a
        for a in $assets; do
            case "$a" in *.js|*.css) api GET "$a"; expect 200; ev "$a -> 200" ;; esac
        done
    done
    api GET /openapi.json; expect 200
    [[ "$(jqb '(.paths // {}) | length')" -gt 0 ]] || fail "openapi.json has no paths"
    ev "/openapi.json -> 200 with paths"
    ext="$(grep -Eo 'https?://[^"]+' "$BODY" | grep -Ev "$(printf '%s' "${BASE_URL#*://}" | sed 's/[.[\*^$]/\\&/g')|example\.|localhost|127\.0\.0\.1" | grep -E '\.(js|css)$' | head -3 | tr '\n' ' ')"
    [[ -z "$ext" ]] || fail "openapi.json references external scripts: ${ext}"
}

ph_project_create() {
    api GET "${PROJECT_API}"
    [[ "$HTTP_CODE" == 404 ]] || fail "project '${SLUG}' already exists (HTTP ${HTTP_CODE}); refusing to reuse and later delete it"
    api_json POST "${API_PREFIX}/projects" \
        "$(jq -nc --arg s "$SLUG" '{slug:$s, display_name:("Acceptance " + $s), description:"throwaway acceptance run"}')"
    expect 201
    st_set project_created 1
    ev "created project ${SLUG}"
}

# image list: --images-dir, else the repo's public fixture
collect_images() {
    local dir="$IMAGES_DIR" f
    local -a all=()
    if [[ -z "$dir" ]]; then
        dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tests/fixtures"
        ev "no --images-dir: using repo fixtures in ${dir} (fetch COCO: python3 scripts/datasets/fetch_coco_subset.py --out data/samples/coco_va)"
    fi
    [[ -d "$dir" ]] || fail "images dir not found: ${dir}"
    while IFS= read -r f; do all+=("$f"); done < <(
        find "$dir" -maxdepth 2 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort | head -n "$IMAGE_COUNT")
    ((${#all[@]})) || fail "no images in ${dir}"
    printf '%s\n' "${all[@]}"
}

ph_ingest() {
    local -a imgs=() args=()
    local f sent=0 base_total
    while IFS= read -r f; do imgs+=("$f"); done < <(collect_images)
    api GET "${PROJECT_API}/ingest/status"; expect 200
    base_total="$(jqb '.total // 0')"
    local i=0 n=${#imgs[@]}
    while ((i < n)); do
        args=()
        local -a chunk=("${imgs[@]:i:10}")
        for f in "${chunk[@]}"; do args+=(-F "images=@${f}"); done
        api POST "${PROJECT_API}/ingest/upload" -F source=acceptance "${args[@]}"
        expect 200
        sent=$((sent + ${#chunk[@]}))
        i=$((i + 10))
    done
    st_set ingested "$sent"
    ev "uploaded ${sent} images"
    _ingest_drained() {
        api GET "${PROJECT_API}/ingest/status"
        [[ "$HTTP_CODE" == 200 ]] && (( $(jqb '.total // 0') - base_total >= sent ))
    }
    poll "$PHASE_TIMEOUT" _ingest_drained || fail "ingest status never reached ${sent} new images"
    ev "ingest status total reached +${sent}"
}

ph_detect_embed() {
    local img
    img="$(collect_images | head -1)" || fail "no image"
    api POST /detect -F "image=@${img}"; expect 200
    ev "POST /detect -> 200, detections=$(jqb '(.detections // []) | length')"
    api POST /embed/image -F "image=@${img}"; expect 200
    local dims
    dims="$(jqb '(.embedding // []) | length')"
    [[ "$dims" == 512 ]] || fail "embedding dims ${dims}, wanted 512"
    ev "POST /embed/image -> 512-dim"
    _embedded() {
        api GET "${PROJECT_API}/crops?limit=1"
        [[ "$HTTP_CODE" == 200 ]] || return 1
        [[ "$(jqb '.n_unembedded // 0')" == 0 && "$(jqb '.total // 0')" -gt 0 ]]
    }
    poll "$PHASE_TIMEOUT" _embedded || fail "crops did not finish embedding: $(snippet)"
    ev "project crops embedded: total=$(jqb '.total')"
}

ph_cluster() {
    api POST "${PROJECT_API}/classes/seed_from_detector" -H 'Content-Type: application/json' --data '{}'
    expect 200
    ev "classes seeded from detector"
    api POST "${PROJECT_API}/pipeline/auto_label?run_vlm=false&train_clusters=true"
    expect 200
    _clustered() {
        api GET "${PROJECT_API}/pipeline/auto_label/status"
        [[ "$HTTP_CODE" == 200 ]] || return 1
        case "$(jqb '.status')" in running|queued|started|pending) return 1 ;; esac
    }
    poll "$PHASE_TIMEOUT" _clustered || fail "auto_label never finished"
    [[ "$(jqb '.status')" != error && "$(jqb '.status')" != failed ]] || fail "auto_label: $(jqb '.error // "error"')"
    api GET "${PROJECT_API}/clusters?max_clusters=50"; expect 200
    ev "auto_label $(jqb '.status // "done"')"
}

ph_vlm_label() {
    api GET "${PROJECT_API}/crops?limit=${VLM_CROPS}"; expect 200
    local ids
    ids="$(jqb "[.crops[]? | (.crop_id // .id)] | .[0:${VLM_CROPS}]")"
    [[ "$(jq 'length' <<< "$ids")" -gt 0 ]] || fail "no crops to label"
    st_set crop_ids "$ids"
    api_json POST "${PROJECT_API}/vlm/label_batch" "$(jq -nc --argjson i "$ids" '{crop_ids:$i}')"
    expect 200
    ev "vlm/label_batch on $(jq 'length' <<< "$ids") crops -> 200: $(snippet)"
}

ph_confirm_labels() {
    local ids cid class_id n=0
    ids="$(st_get crop_ids)"
    [[ -n "$ids" ]] || fail "no crop ids (vlm_label did not run in this session)"
    api GET "${PROJECT_API}/classes"; expect 200
    class_id="$(jqb '[.classes[]? | (.class_id // .id)] | .[0] // empty')"
    [[ -n "$class_id" ]] || fail "project has no classes to confirm against"
    for cid in $(jq -r '.[]' <<< "$ids"); do
        api GET "${PROJECT_API}/crops/${cid}"
        local cls
        cls="$(jqb '(.class_id // empty)')"
        [[ "$cls" =~ ^[0-9]+$ ]] || cls="$class_id"
        api_json PUT "${PROJECT_API}/crops/${cid}/label" "$(jq -nc --argjson c "$cls" '{class_id:$c, label_source:"human"}')"
        expect 200
        n=$((n + 1))
    done
    api GET "${PROJECT_API}/crops?label_source=human&label_validated=true&limit=1"; expect 200
    [[ "$(jqb '.total // 0')" -ge "$n" ]] || fail "only $(jqb '.total // 0') human-validated crops, wanted >= ${n}"
    ev "human-confirmed ${n} labels; API reports $(jqb '.total') validated"
}

ph_holdout() {
    api_json POST "${PROJECT_API}/test_holdout/freeze?force=true" '{"percent":10}'
    expect 200
    ev "holdout frozen: n_frozen=$(jqb '.n_frozen') classes=$(jqb '.n_classes_covered')"
}

ph_export() {
    api_json POST "${PROJECT_API}/export/yolo" '{"version_tag":"acceptance"}'
    expect 200
    ev "export/yolo -> 200: $(snippet)"
    local a
    for a in class_registry.json data.yaml manifest.json; do
        api GET "${PROJECT_API}/export/registry/${a}"; expect 200
        ev "registry artifact ${a} -> 200"
    done
}

train_spec() {
    jq -nc --arg gpu "$TRAIN_GPU" --argjson ep "$TRAIN_EPOCHS" \
        '{profile:"probe", model_size:"n", cuda_visible_devices:$gpu, hyperparameters:{epochs:$ep}, submitted_by:"acceptance"}'
}

ph_train() {
    api_json POST "${PROJECT_API}/train/preflight" "$(train_spec)"; expect 200
    local blocked
    blocked="$(jqb '.blocked')"
    ev "preflight blocked=${blocked}: $(jqb '.summary // ""')"
    [[ "$blocked" == false ]] || fail "preflight blocked the run: $(snippet)"
    api_json POST "${PROJECT_API}/train/start" "$(train_spec)"; expect 201
    local job
    job="$(jqb '.job_id')"
    [[ -n "$job" && "$job" != null ]] || fail "no job_id: $(snippet)"
    st_set train_job "$job"
    ev "training job ${job} started (probe, ${TRAIN_EPOCHS} epochs, GPU ${TRAIN_GPU})"
    _trained() {
        api GET "${PROJECT_API}/train/status/${job}"
        [[ "$HTTP_CODE" == 200 ]] || return 1
        case "$(jqb '.state')" in finished|failed|cancelled|skipped) return 0 ;; esac
        return 1
    }
    poll "$TRAIN_TIMEOUT" _trained || fail "training ${job} did not finish in ${TRAIN_TIMEOUT}s"
    [[ "$(jqb '.state')" == finished ]] || fail "training ended $(jqb '.state'): $(jqb '.error // ""')"
    ev "training finished; map50=$(jqb '.eval.map50 // "n/a"')"
}

ph_bakeoff() {
    api GET "${PROJECT_API}/bakeoff/trained_models"
    [[ "$HTTP_CODE" == 200 ]] || skip "bake-off not available (HTTP ${HTTP_CODE})"
    local model ds
    model="$(jqb '[.models[]? // .[]? | (.name // .model // .id // .)] | .[0] // empty' 2>/dev/null)"
    [[ -n "$model" ]] || skip "no trained model to bake off"
    api GET "${PROJECT_API}/bakeoff/eval_datasets"
    ds="$(jqb '[.datasets[]? // .[]? | (.name // .id // .)] | .[0] // empty' 2>/dev/null)"
    [[ -n "$ds" ]] || skip "no eval dataset"
    api_json POST "${PROJECT_API}/bakeoff/run" "$(jq -nc --arg m "$model" --arg d "$ds" '{models:[$m], datasets:[$d]}')"
    expect 200
    local job
    job="$(jqb '.job_id')"
    _bake() {
        api GET "${PROJECT_API}/bakeoff/status/${job}"
        [[ "$HTTP_CODE" == 200 ]] || return 1
        case "$(jqb '.state // .status')" in done|finished|failed|error|cancelled) return 0 ;; esac
        return 1
    }
    poll "$PHASE_TIMEOUT" _bake || fail "bake-off ${job} did not finish"
    case "$(jqb '.state // .status')" in done|finished) ;; *) fail "bake-off ended $(jqb '.state // .status')" ;; esac
    ev "bake-off ${job} finished"
}

promote_once() { # promote_once FORCE(true|false): leaves final status in BODY/HTTP_CODE
    local force="$1" job="$2" name="$3" pid
    api_json POST "${PROJECT_API}/train/promote/${job}?wait=false" \
        "$(jq -nc --arg n "$name" --argjson f "$force" '{triton_name:$n, force:$f, overwrite:true}')"
    case "$HTTP_CODE" in
        200) return 0 ;;
        202) ;;
        *) return 1 ;;
    esac
    pid="$(jqb '.promote_id // empty')"
    [[ -n "$pid" ]] || return 1
    _promoted() {
        api GET "${PROJECT_API}/train/promote/${job}/jobs/${pid}"
        [[ "$HTTP_CODE" == 200 ]] || return 1
        case "$(jqb '.status')" in done|failed) return 0 ;; esac
        return 1
    }
    poll "$PHASE_TIMEOUT" _promoted || { ev "promote job ${pid} never finished"; return 1; }
    ev "promote job ${pid} phases ended in: $(jqb '.status')"
    [[ "$(jqb '.status')" == "done" ]]
}

ph_promote() {
    local job name
    job="$(st_get train_job)"
    [[ -n "$job" ]] || fail "no training job (train did not run in this session)"
    name="acc_$(printf '%s' "$SLUG" | tr -c 'a-z0-9\n' '_')"
    # register for cleanup BEFORE promoting so a half-built model is still removed
    st_add promoted_models "$name"
    st_set promoted_name "$name"
    if promote_once false "$job" "$name"; then
        ev "promoted ${name} without force"
    else
        ev "gate/promote failed without force ($(jqb '.error // .detail // ""' | head -c 200)); retrying force=true (undertrained probe model)"
        promote_once true "$job" "$name" || fail "forced promote failed: $(snippet)"
        ev "promoted ${name} with force=true"
    fi
    if [[ -n "$MODELS_DIR" ]]; then
        [[ -d "${MODELS_DIR}/${name}" ]] || fail "promoted model dir ${MODELS_DIR}/${name} not found"
        ev "model dir ${MODELS_DIR}/${name} present"
    fi
}

ph_infer() {
    local name img n
    name="$(st_get promoted_name)"
    [[ -n "$name" ]] || fail "no promoted model (promote did not run in this session)"
    img="$(collect_images | head -1)" || fail "no image"
    for n in 1 2 3 4 5; do   # first inference may cold-start the engine
        api POST "/detect?model_name=${name}" -F "image=@${img}"
        [[ "$HTTP_CODE" == 200 ]] && break
        sleep "$POLL_S"
    done
    expect 200
    ev "POST /detect?model_name=${name} -> 200, detections=$(jqb '(.detections // []) | length')"
}

ph_model_delete() {
    local name
    name="$(st_get promoted_name)"
    [[ -n "$name" ]] || fail "no promoted model (promote did not run in this session)"
    api DELETE "${PROJECT_API}/models/${name}?force=true"; expect 200
    ev "DELETE models/${name} -> 200"
    st_set promoted_models ""
    if [[ -n "$MODELS_DIR" ]]; then
        [[ ! -d "${MODELS_DIR}/${name}" ]] || fail "model dir ${MODELS_DIR}/${name} still present after delete"
        ev "model dir removed"
    fi
}

ph_metrics() {
    api GET /metrics; expect 200
    local m missing=() warn=()
    for m in op_ingest_images_total op_queue_depth op_embedding_state_items; do
        grep -q "^${m}" "$BODY" || missing+=("$m")
    done
    grep -Eq "^op_[a-z_]+\{[^}]*project=\"${SLUG}\"" "$BODY" || missing+=("any op_* series labelled project=\"${SLUG}\"")
    for m in op_queue_oldest_item_age_seconds op_worker_up op_worker_last_heartbeat_timestamp_seconds op_opensearch_shards; do
        grep -q "^${m}" "$BODY" || warn+=("$m")
    done
    ((${#warn[@]})) && ev "note: series absent (not required): ${warn[*]}"
    ((${#missing[@]} == 0)) || fail "missing metric series: ${missing[*]}"
    ev "required op_* series present, incl. per-project label"
}

ph_log_noise() {
    [[ -n "$PROJECT_NAME" ]] || skip "needs --project-name"
    command -v docker >/dev/null || skip "docker not available"
    local logs="$WORKDIR/stack.log" top
    docker compose -p "$PROJECT_NAME" logs --no-color --since "$STARTED_AT" > "$logs" 2>&1 \
        || fail "docker compose -p ${PROJECT_NAME} logs failed: $(head -c 200 "$logs")"
    local tb
    tb="$(grep -c 'Traceback (most recent call last)' "$logs" || true)"
    # normalise digits/uuids/timestamps, then count the loudest repeated line
    top="$(sed -E 's/^[^|]*\| *//; s/[0-9a-f]{8}-[0-9a-f-]{27}/<id>/g; s/[0-9]+/N/g' "$logs" \
        | grep -Ei 'warn|error|repair' | sort | uniq -c | sort -rn | head -1 || true)"
    ev "log lines=$(wc -l < "$logs") tracebacks=${tb} loudest: ${top:-none}"
    local cnt="${top%% *}"
    cnt="${cnt// /}"
    [[ -z "$cnt" || ! "$cnt" =~ ^[0-9]+$ ]] && cnt=0
    ((cnt <= LOG_REPEAT_MAX)) || fail "a warning/error line repeated ${cnt}x (> ${LOG_REPEAT_MAX}): ${top}"
    ((tb == 0)) || fail "${tb} Python traceback(s) in the stack logs (see ${logs})"
}

op_cli() { "${INSTALL_DIR}/openprocessor" "$@"; }

ph_vlm_switch() {
    [[ -n "$VLM_SWITCH_ID" ]] || skip "pass --vlm-switch-id <catalog id> to enable"
    [[ -n "$INSTALL_DIR" ]] || skip "needs --install-dir"
    local -a flags=(--yes)
    ((VLM_SWITCH_FORCE)) && flags+=(--force)
    op_cli vlm use "$VLM_SWITCH_ID" "${flags[@]}" >> "$WORKDIR/vlm_use.log" 2>&1 \
        || fail "vlm use ${VLM_SWITCH_ID} failed: $(tail -c 300 "$WORKDIR/vlm_use.log")"
    op_cli vlm status >> "$WORKDIR/vlm_use.log" 2>&1 || fail "vlm status failed"
    api GET /health; expect 200
    ev "vlm use ${VLM_SWITCH_ID} ${flags[*]} ok; API still healthy"
}

need_install_dir() {
    [[ -n "$INSTALL_DIR" ]] || skip "needs --install-dir"
    [[ -x "${INSTALL_DIR}/openprocessor" ]] || fail "${INSTALL_DIR}/openprocessor not found/executable"
}

stack_ids() {
    [[ -n "$PROJECT_NAME" ]] || { echo ""; return; }
    docker ps -q --filter "label=com.docker.compose.project=${PROJECT_NAME}" | sort | tr '\n' ' '
}

ph_installer_rerun() {
    need_install_dir
    local before after
    before="$(stack_ids)"
    op_cli upgrade --yes >> "$WORKDIR/installer.log" 2>&1 \
        || fail "re-run failed: $(tail -c 300 "$WORKDIR/installer.log")"
    after="$(stack_ids)"
    [[ "$before" == "$after" ]] || fail "re-run recreated containers (before: ${before} after: ${after})"
    ev "re-run is a no-op: ${after:-<no project-name given; container identity not checked>}"
}

ph_installer_repair() {
    need_install_dir
    op_cli repair --yes >> "$WORKDIR/installer.log" 2>&1 \
        || fail "repair failed: $(tail -c 300 "$WORKDIR/installer.log")"
    api GET /health; expect 200
    ev "repair exited 0; API healthy"
}

ph_installer_upgrade() {
    need_install_dir
    [[ -n "$UPGRADE_VERSION" ]] || skip "pass --upgrade-version vX.Y.Z to enable upgrade/rollback"
    op_cli upgrade --version "$UPGRADE_VERSION" --yes >> "$WORKDIR/installer.log" 2>&1 \
        || fail "upgrade to ${UPGRADE_VERSION} failed: $(tail -c 300 "$WORKDIR/installer.log")"
    api GET /health; expect 200
    ev "upgraded to ${UPGRADE_VERSION}; API healthy"
    "${INSTALL_DIR}/setup-openprocessor.sh" --dir "$INSTALL_DIR" --rollback --yes >> "$WORKDIR/installer.log" 2>&1 \
        || fail "rollback failed: $(tail -c 300 "$WORKDIR/installer.log")"
    api GET /health; expect 200
    ev "rolled back; API healthy"
}

ph_installer_uninstall() {
    need_install_dir
    [[ -n "$PROJECT_NAME" ]] || skip "needs --project-name to prove only this install was removed"
    # destructive and ends the stack: runs only when asked for by name
    [[ -n "$ONLY" && ",${ONLY}," == *",installer_uninstall,"* ]] || skip "destructive: select with --only installer_uninstall"
    op_cli uninstall --yes >> "$WORKDIR/installer.log" 2>&1 \
        || fail "uninstall failed: $(tail -c 300 "$WORKDIR/installer.log")"
    [[ -z "$(stack_ids)" ]] || fail "containers of ${PROJECT_NAME} remain after uninstall"
    ev "uninstall removed the ${PROJECT_NAME} containers"
}

ph_teardown() {
    cleanup_resources || fail "cleanup incomplete (see stderr)"
    ev "project ${SLUG} and promoted models removed"
    if [[ "$(st_get project_created)" == 1 ]]; then
        api GET "${PROJECT_API}"
        [[ "$HTTP_CODE" == 404 ]] || fail "project ${SLUG} still resolves (HTTP ${HTTP_CODE})"
        ev "project lookup -> 404"
    fi
}

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
