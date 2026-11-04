#!/bin/bash
# =============================================================================
# acceptance_phases_import.sh - the dataset_import and combine acceptance phases
# =============================================================================
# Sourced by acceptance_run.sh after acceptance_phases.sh (it uses registry_state from there). Both
# phases write class ids by name through the class registry, which the registry document made a
# concurrent-write surface; nothing else scripted covers them. Counts come from the project stats
# route (GET /curation/projects/{project}/stats), so the phases need only the API.
# shellcheck disable=SC2154

# project_count SLUG KEY: a count from the stats route, e.g. project_count my-proj items
project_count() {
    api GET "${API_PREFIX}/projects/$1/stats"; expect 200
    jqb ".counts.$2"
}

# Throwaway projects (the import target, the combine target) are removed with the run's project.
track_extra_project() {
    st_add extra_projects "$1"
}

# create_extra_project SLUG: a throwaway project, recorded for cleanup before it exists.
create_extra_project() {
    track_extra_project "$1"
    api GET "${API_PREFIX}/projects/$1"
    [[ "$HTTP_CODE" == 404 ]] || fail "project '$1' already exists (HTTP ${HTTP_CODE}); refusing to reuse and later delete it"
    api_json POST "${API_PREFIX}/projects" \
        "$(jq -nc --arg s "$1" '{slug:$s, display_name:("Acceptance " + $s), description:"throwaway acceptance run"}')"
    expect 201
}

# import_dataset PROJECT SOURCE_JSON: preview, then import accepting the name-based suggestions;
# waits for the job and leaves its id in IMPORT_ID.
import_dataset() {
    local project="$1" source="$2" req status
    local base="${API_PREFIX}/projects/${project}"
    req="$(jq -nc --argjson s "$source" '{source:$s, accept_suggestions:true}')"
    api_json POST "${base}/datasets/preview" "$req"; expect 200
    [[ "$(jqb '.blocking')" == false ]] || fail "preview is blocked: $(snippet)"
    ev "preview -> ${project}: $(jqb '[.classes[] | .dataset_class] | join(",")')"
    api_json POST "${base}/datasets/imports" \
        "$(jq -nc --argjson r "$req" --arg k "$(jqb '.import_key')" '$r + {expected_import_key:$k}')"
    expect 200 202
    IMPORT_ID="$(jqb '.import_id')"
    _import_done() {
        api GET "${base}/datasets/imports/${IMPORT_ID}"; expect 200
        status="$(jqb '.status')"
        case "$status" in failed|cancelled) fail "import ${IMPORT_ID} ${status}: $(snippet)" ;; esac
        [[ "$status" == completed ]]
    }
    poll "$PHASE_TIMEOUT" _import_done || fail "import ${IMPORT_ID} did not complete in ${PHASE_TIMEOUT}s (status ${status})"
}

ph_dataset_import() {
    local export_dir before after fresh want got first_import
    api GET "${PROJECT_API}/export/datasets"; expect 200
    export_dir="$(jqb '(.datasets // []) | map(select(.kind == "yolo")) | .[0].export_dir // empty')"
    [[ -n "$export_dir" ]] || skip "no multi-class export to import (the export phase did not run)"
    before="$(registry_state)"

    # 1. the run's own export imported back into the run's project: classes map by NAME, so the
    #    registry (ids and names) is exactly what it was
    import_dataset "$SLUG" "$(jq -nc --arg p "$export_dir" '{path:$p, format:"auto"}')"
    first_import="$IMPORT_ID"
    after="$(registry_state)"
    [[ "$before" == "$after" ]] || fail "importing the project's own export changed its classes (before: ${before} after: ${after})"
    ev "own-export import ${first_import}: class ids and names unchanged"

    # 2. the same files again are the same job
    import_dataset "$SLUG" "$(jq -nc --arg p "$export_dir" '{path:$p, format:"auto"}')"
    [[ "$IMPORT_ID" == "$first_import" ]] || fail "a repeat import made a new job (${first_import} then ${IMPORT_ID})"
    ev "repeat import reused ${first_import}"

    # 3. into an empty project every class is created, by name, once; undo gives them up again
    fresh="${SLUG}-imp"
    create_extra_project "$fresh"
    import_dataset "$fresh" "$(jq -nc --arg p "$export_dir" '{path:$p, format:"auto"}')"
    want="$(jq -c '[.[] | select(.deprecated | not) | .class_name]' <<< "$before")"
    api GET "${API_PREFIX}/projects/${fresh}/classes"; expect 200
    [[ "$(jq -c --argjson want "$want" '
        [.classes[] | select(.deprecated | not) | .class_name] as $got
        | ($got | length) > 0 and ($got == ($got | unique)) and ($got - $want | length) == 0' "$BODY")" == true ]] \
        || fail "the fresh project's classes are not the export's, once each: $(snippet) (export: ${want})"
    ev "fresh-project import ${IMPORT_ID}: classes created by name"
    api_json POST "${API_PREFIX}/projects/${fresh}/datasets/imports/${IMPORT_ID}/undo" '{"dry_run":true}'; expect 200
    api_json POST "${API_PREFIX}/projects/${fresh}/datasets/imports/${IMPORT_ID}/undo" \
        '{"dry_run":false,"deprecate_created_classes":true}'
    expect 200 202
    _fresh_empty() {
        api GET "${API_PREFIX}/projects/${fresh}/classes"; expect 200
        [[ "$(jqb '[.classes[] | select(.deprecated | not)] | length')" == 0 ]]
    }
    poll "$PHASE_TIMEOUT" _fresh_empty || fail "undo left active classes the import created: $(snippet)"
    ev "undo deprecated the classes the import created"
}

ph_combine() {
    local second a b req preview sha job status
    second="${SLUG}-src2"
    api GET "${PROJECT_API}/export/datasets"; expect 200
    local export_dir
    export_dir="$(jqb '(.datasets // []) | map(select(.kind == "yolo")) | .[0].export_dir // empty')"
    [[ -n "$export_dir" ]] || skip "no multi-class export to build a second source from"
    create_extra_project "$second"
    import_dataset "$second" "$(jq -nc --arg p "$export_dir" '{path:$p, format:"auto"}')"
    a="$(registry_state)"
    api GET "${API_PREFIX}/projects/${second}/classes"; expect 200
    b="$(jq -S -c '[.classes[] | {class_id, class_name, deprecated}] | sort_by(.class_id)' "$BODY")"

    req="$(jq -nc --arg a "$SLUG" --arg b "$second" --arg t "${SLUG}-combined" '
        {target:{slug:$t, display_name:("Acceptance " + $t)},
         sources:[{project:$a, include:{label_states:"all"}}, {project:$b, include:{label_states:"all"}}],
         dedup:"none"}')"
    api_json POST "${API_PREFIX}/projects/combine/preview" "$req"; expect 200
    [[ "$(jqb '.ok')" == true ]] || fail "combine preview refused: $(snippet)"
    sha="$(jqb '.preview_sha')"
    preview="$(jqb '.suggested_mapping' | jq -c .)"
    track_extra_project "${SLUG}-combined"
    api_json POST "${API_PREFIX}/projects/combine" \
        "$(jq -nc --argjson r "$req" --argjson m "$preview" --arg s "$sha" '$r + {class_mapping:$m, expected_preview_sha:$s}')"
    expect 200 202
    job="$(jqb '.job_id')"
    _combined() {
        api GET "${API_PREFIX}/projects/combine/${job}"; expect 200
        status="$(jqb '.status')"
        case "$status" in failed|cancelled) fail "combine ${job} ${status}: $(snippet)" ;; esac
        [[ "$status" == completed ]]
    }
    poll "$PHASE_TIMEOUT" _combined || fail "combine ${job} did not complete in ${PHASE_TIMEOUT}s (status ${status})"

    # the target owns its ids: every source class arrives by NAME, once
    api GET "${API_PREFIX}/projects/${SLUG}-combined/classes"; expect 200
    [[ "$(jqb '[.classes[] | select(.deprecated | not) | .class_name] | (length == (unique | length))')" == true ]] \
        || fail "the combined project has two active classes with one name: $(snippet)"
    local want got
    want="$( { jq -r '.[] | select(.deprecated | not) | .class_name' <<< "$a"; jq -r '.[] | select(.deprecated | not) | .class_name' <<< "$b"; } | sort -u | tr '\n' ' ')"
    got="$(jqb '[.classes[] | select(.deprecated | not) | .class_name] | sort | join(" ")')"
    [[ "${got% }" == "${want% }" ]] || fail "combined classes (${got}) are not the union of the sources' (${want})"
    ev "combined ${job}: $(project_count "${SLUG}-combined" items) items; one class per name"

    # both sources are exactly as they were
    [[ "$(registry_state)" == "$a" ]] || fail "combine changed the source project ${SLUG}'s classes"
    api GET "${API_PREFIX}/projects/${second}/classes"; expect 200
    [[ "$(jq -S -c '[.classes[] | {class_id, class_name, deprecated}] | sort_by(.class_id)' "$BODY")" == "$b" ]] \
        || fail "combine changed the source project ${second}'s classes"
    ev "both sources' registries are unchanged"
}
