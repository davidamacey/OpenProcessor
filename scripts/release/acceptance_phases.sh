#!/bin/bash
# =============================================================================
# acceptance_phases.sh - the scripted acceptance phases (ph_<name>)
# =============================================================================
# Sourced by acceptance_run.sh, which owns the options, WORKDIR and the
# phase runner. Not runnable on its own.

# Variables are assigned by acceptance_run.sh before this file is sourced.
# shellcheck disable=SC2154

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

# first image path; reads all of collect_images so it never dies of SIGPIPE
first_image() { collect_images | sed -n 1p; }

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
    img="$(first_image)" || fail "no image"
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
    api POST "${PROJECT_API}/classes/seed_from_detector" -H 'Content-Type: application/json' --data '{"dry_run":false}'
    expect 200
    ev "classes seeded from detector: $(jqb '(.created // []) | length') created"
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

# A project calls no VLM until it activates one, so the phase opts in to the
# deployment's endpoint first (the OCC value is what the project reports now).
ph_vlm_label() {
    local current
    api GET "${PROJECT_API}/vlm/endpoints/active"; expect 200
    current="$(jqb '{name: .active.name, revision: .active.revision}')"
    api_json POST "${PROJECT_API}/vlm/endpoints/env/activate" \
        "$(jq -nc --argjson c "$current" '{revision:null, expected_active:$c}')"
    expect 200
    ev "vlm activated for the project: env"
    api GET "${PROJECT_API}/crops?limit=${VLM_CROPS}"; expect 200
    local ids
    ids="$(jqb "[.crops[]? | (.crop_id // .id)] | .[0:${VLM_CROPS}]")"
    [[ "$(jq 'length' <<< "$ids")" -gt 0 ]] || fail "no crops to label"
    st_set crop_ids "$ids"
    api_json POST "${PROJECT_API}/vlm/label_batch" "$(jq -nc --argjson i "$ids" '{crop_ids:$i}')"
    expect 200
    ev "vlm/label_batch on $(jq 'length' <<< "$ids") crops -> 200: $(snippet)"
}

# The trainer refuses classes under the per-class floor (20 validated crops),
# so the run confirms CONFIRM_CROPS crops into ONE class and trains a
# single-class subset. The labels are arbitrary: this proves the pipeline
# runs end to end, not that the model is any good.
ph_confirm_labels() {
    local ids cid class_id n=0
    ids="$(st_get crop_ids)"
    [[ -n "$ids" ]] || fail "no crop ids (vlm_label did not run in this session)"
    api GET "${PROJECT_API}/classes"; expect 200
    class_id="$(jqb '[.classes[]? | (.class_id // .id)] | .[0] // empty')"
    [[ -n "$class_id" ]] || fail "project has no classes to confirm against"
    api GET "${PROJECT_API}/crops?limit=${CONFIRM_CROPS}"; expect 200
    ids="$(jqb "[.crops[]? | (.crop_id // .id)] | .[0:${CONFIRM_CROPS}]")"
    for cid in $(jq -r '.[]' <<< "$ids"); do
        api_json PUT "${PROJECT_API}/crops/${cid}/label" "$(jq -nc --argjson c "$class_id" '{class_id:$c, label_source:"human"}')"
        expect 200
        n=$((n + 1))
    done
    st_set confirm_class "$class_id"
    api GET "${PROJECT_API}/crops?label_source=human&label_validated=true&limit=1"; expect 200
    [[ "$(jqb '.total // 0')" -ge "$n" ]] || fail "only $(jqb '.total // 0') human-validated crops, wanted >= ${n}"
    ev "human-confirmed ${n} crops as class ${class_id}; API reports $(jqb '.total') validated"
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
    local cls
    cls="$(st_get confirm_class)"
    jq -nc --arg gpu "$TRAIN_GPU" --argjson ep "$TRAIN_EPOCHS" --argjson cls "${cls:-null}" \
        '{profile:"probe", model_size:"n", cuda_visible_devices:$gpu, hyperparameters:{epochs:$ep}, submitted_by:"acceptance"}
         + (if $cls == null then {} else {include_classes:[$cls], single_cls:true} end)'
}

ph_train() {
    api_json POST "${PROJECT_API}/train/preflight" "$(train_spec)"; expect 200
    local blocked
    blocked="$(jqb '.blocked')"
    ev "preflight blocked=${blocked}: $(jqb '.summary // ""')"
    [[ "$blocked" == false ]] || fail "preflight blocked the run: $(jqb '[.checks[] | select(.severity == "block") | "\(.name): \(.message)"] | join("; ")' | head -c 600)"
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
    local trainjob
    trainjob="$(st_get train_job)"
    [[ -n "$trainjob" ]] || skip "no training run in this session"
    api GET "${PROJECT_API}/bakeoff/trained_models"
    [[ "$HTTP_CODE" == 200 ]] || skip "bake-off not available (HTTP ${HTTP_CODE})"
    [[ "$(jq -r --arg j "$trainjob" '[.models[]? | select(.run_id == $j)] | length' "$BODY")" -gt 0 ]] \
        || skip "trained run ${trainjob} not offered for bake-off"
    # the run's own training export is always a valid eval dataset (run:<job_id>)
    api_json POST "${PROJECT_API}/bakeoff/run" \
        "$(jq -nc --arg j "$trainjob" '{models:[{source:"run", run_id:$j}], datasets:[{id:("run:" + $j)}]}')"
    expect 200 202
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
    local job name final
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
    # the API namespaces the Triton model as <project>__<name>; use what it reports
    final="$(jq -r '.triton_name // .result.triton_name // empty' "$BODY" 2>/dev/null)"
    if [[ -n "$final" && "$final" != "$name" ]]; then
        ev "API served the model as ${final} (requested ${name})"
        name="$final"
        st_add promoted_models "$name"
        st_set promoted_name "$name"
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
    img="$(first_image)" || fail "no image"
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
