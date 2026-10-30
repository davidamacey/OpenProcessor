#!/bin/bash
# =============================================================================
# acceptance_phases_review.sh - cluster-card, label-operation and loop checks
# =============================================================================
# Sourced by acceptance_run.sh after acceptance_phases.sh; ph_cluster calls the
# check_cluster_* functions, ph_label_ops and ph_loop are phases of their own.
# Not runnable on its own.

# Variables are assigned by acceptance_run.sh before this file is sourced.
# shellcheck disable=SC2154

# crops_total QUERY: the total of GET /crops with the query (one item per page)
crops_total() {
    api GET "${PROJECT_API}/crops?${1}&limit=1"; expect 200
    jqb '.total // 0'
}

# the served sort and a flag filter hold, and the cards carry label_agreement only
check_cluster_list() {
    local sort flag n total sorted
    api GET "${PROJECT_API}/clusters?max_clusters=1000"; expect 200
    n="$(jqb '(.items // []) | length')"
    total="$(jqb '.total // -1')"
    [[ "$(jqb '(.sort_options // []) | length')" -gt 0 ]] || fail "clusters served no sort_options"
    if [[ "$(jqb '.truncated')" == false ]]; then
        [[ "$total" == "$n" ]] || fail "total ${total} != ${n} served cards"
    fi
    ev "GET clusters: ${n} cards, total=${total}, $(jqb '.sort_options | length') sort options, $(jqb '(.flag_vocabulary // []) | length') flags"
    if ((n == 0)); then
        ev "note: no clusters formed on this image set; card, sort, flag and facets checks skipped"
        return 0
    fi
    [[ "$(jqb '[.items[] | has("label_agreement")] | all')" == true ]] || fail "a cluster card lacks label_agreement"
    [[ "$(jqb '[.items[] | has("label_purity")] | any')" == false ]] || fail "a cluster card still serves label_purity"
    ev "cards carry label_agreement and no label_purity"
    sort="$(jqb '[.sort_options[].id] | (if index("size_asc") then "size_asc" else (map(select(. != "size_desc")) | .[0]) end) // empty')"
    if [[ -n "$sort" ]]; then
        api GET "${PROJECT_API}/clusters?max_clusters=1000&sort=${sort}"; expect 200
        [[ "$(jqb '.sort')" == "$sort" ]] || fail "sort=${sort} served as '$(jqb '.sort')'"
        case "$sort" in
            size_asc) sorted="$(jqb '[.items[].size] as $s | $s == ($s | sort)')" ;;
            *) sorted=true ;;
        esac
        [[ "$sorted" == true ]] || fail "cards are not in ${sort} order"
        ev "sort=${sort} served in order"
    fi
    flag="$(jqb '(.flag_vocabulary // [])[0].id // empty')"
    if [[ -z "$flag" ]]; then
        ev "note: flag_vocabulary is empty; flag filter not checked"
        return 0
    fi
    api GET "${PROJECT_API}/clusters?max_clusters=1000&flag=${flag}"; expect 200
    [[ "$(jqb '.total')" == "$(jqb '.items | length')" ]] || fail "flag=${flag}: total $(jqb '.total') != $(jqb '.items | length') cards"
    [[ "$(jq -r --arg f "$flag" '[.items[] | (.flags | index($f)) != null] | all' "$BODY")" == true ]] \
        || fail "flag=${flag} served a card without that flag"
    (( $(jqb '.total') <= total )) || fail "flag filter widened the list"
    ev "flag=${flag}: $(jqb '.total') of ${total} cards, each carrying it"
}

# a cluster's facet counts equal the matching filtered /crops totals
check_cluster_facets() {
    local cid facets v t want tab
    api GET "${PROJECT_API}/clusters?max_clusters=1000"; expect 200
    cid="$(jqb '[.items[] | select(.size > 0)] | .[0].cluster_id // empty')"
    [[ -n "$cid" ]] || return 0
    api GET "${PROJECT_API}/clusters/${cid}/facets"; expect 200
    facets="$(cat "$BODY")"
    fct() { jq -r "$1" <<< "$facets"; }
    t="$(crops_total "cluster_id=${cid}")"
    [[ "$(fct .total)" == "$t" ]] || fail "facets total $(fct .total) != /crops total ${t} (cluster ${cid})"
    for v in true false; do
        want="$(fct ".validation.$([[ "$v" == true ]] && echo validated || echo unvalidated)")"
        t="$(crops_total "cluster_id=${cid}&label_validated=${v}")"
        [[ "$want" == "$t" ]] || fail "facets validation(${v}) ${want} != /crops total ${t}"
    done
    v="$(fct '.detector_classes[0].value // empty')"
    if [[ -n "$v" ]]; then
        want="$(fct '.detector_classes[0].count')"
        t="$(crops_total "cluster_id=${cid}&detector_class_name=$(jq -rn --arg v "$v" '$v|@uri')")"
        [[ "$want" == "$t" ]] || fail "facets detector class '${v}' ${want} != /crops total ${t}"
    fi
    v="$(fct '.class_sources[0].value // empty')"
    if [[ -n "$v" ]]; then
        want="$(fct '.class_sources[0].count')"
        t="$(crops_total "cluster_id=${cid}&class_source=$(jq -rn --arg v "$v" '$v|@uri')")"
        [[ "$want" == "$t" ]] || fail "facets class source '${v}' ${want} != /crops total ${t}"
    fi
    for tab in $(fct '.review_tabs | keys[]'); do
        want="$(jq -r --arg t "$tab" '.review_tabs[$t]' <<< "$facets")"
        t="$(crops_total "cluster_id=${cid}&review_tab=${tab}")"
        [[ "$want" == "$t" ]] || fail "facets review tab '${tab}' ${want} != /crops total ${t}"
    done
    ev "cluster ${cid} facets equal the filtered /crops totals (total, validation, detector class, class source, review tabs)"
}

# state of the items a label operation must NOT touch: validated or holdout crops
# outside the chosen ones (one page of /crops; compared before and after)
guard_snapshot() { # guard_snapshot CHOSEN_IDS_JSON
    api GET "${PROJECT_API}/crops?page_size=500"; expect 200
    jq -cS --argjson ex "$1" '[.crops[]?
        | select((.test_holdout // false) or (.label_validated // false))
        | select(.crop_id as $c | ($ex | index($c)) == null)
        | {key: .crop_id, value: [.class_id, .label_validated, .class_source]}] | from_entries' "$BODY"
}

# the chosen crops' class state as {id: [class_id, label_validated, class_source]}
chosen_state() { # chosen_state IDS_JSON
    api GET "${PROJECT_API}/crops?ids=$(jq -r 'join(",")' <<< "$1")"; expect 200
    jq -cS '[.crops[]? | {key: .crop_id, value: [.class_id, .label_validated, .class_source]}] | from_entries' "$BODY"
}

# Batch-label crops, find the operation by id, dry-run then really undo it, and
# prove a second undo is refused and that holdout and validated items never moved.
ph_label_ops() {
    local target ids n op_id guard0 guard1 before after
    api GET "${PROJECT_API}/label_ops?limit=1"
    [[ "$HTTP_CODE" != 404 ]] || skip "label operations not served by this build (GET label_ops -> 404)"
    expect 200
    api GET "${PROJECT_API}/classes"; expect 200
    target="$(jqb '[.classes[]? | (.class_id // .id)] | .[0] // empty')"
    [[ -n "$target" ]] || fail "project has no classes to label into"
    api GET "${PROJECT_API}/crops?label_validated=false&page_size=500"; expect 200
    ids="$(jq -c --argjson t "$target" --argjson n "$LABEL_OPS_CROPS" \
        '[.crops[]? | select((.test_holdout // false) | not) | select(.class_id != $t) | .crop_id] | .[0:$n]' "$BODY")"
    n="$(jq 'length' <<< "$ids")"
    ((n >= 2)) || skip "fewer than 2 unvalidated non-holdout crops to relabel (${n})"
    before="$(chosen_state "$ids")"
    guard0="$(guard_snapshot "$ids")"

    api_json PUT "${PROJECT_API}/crops/batch_label" \
        "$(jq -nc --argjson i "$ids" --argjson c "$target" '{crop_ids:$i, class_id:$c, label_source:"human"}')"
    expect 200
    op_id="$(jqb '.op_id // empty')"
    [[ -n "$op_id" ]] || fail "batch_label returned no op_id: $(snippet)"
    ev "batch_label of ${n} crops -> op_id ${op_id}, updated=$(jqb '.updated')"

    api GET "${PROJECT_API}/label_ops?limit=50"; expect 200
    [[ "$(jq -r --arg o "$op_id" '[.items[] | select(.op_id == $o and .undoable == true)] | length' "$BODY")" == 1 ]] \
        || fail "label_ops does not list ${op_id} as undoable: $(snippet)"
    ev "label_ops lists ${op_id} as undoable"

    # a dry run (the default body) counts and writes nothing
    api_json POST "${PROJECT_API}/label_ops/${op_id}/undo" '{}'; expect 200
    [[ "$(jqb '.dry_run')" == true ]] || fail "undo without dry_run=false was not a dry run: $(snippet)"
    ev "undo dry run: would restore $(jqb '.restored') of $(jqb '.matched')"
    [[ "$(chosen_state "$ids")" != "$before" ]] || fail "labels were already back after only a dry run"
    api GET "${PROJECT_API}/label_ops?limit=50"; expect 200
    [[ "$(jq -r --arg o "$op_id" '[.items[] | select(.op_id == $o and .undoable == true and .undone == false)] | length' "$BODY")" == 1 ]] \
        || fail "the dry run changed ${op_id}: $(snippet)"
    guard1="$(guard_snapshot "$ids")"
    [[ "$guard1" == "$guard0" ]] || fail "holdout or validated items changed after the batch label / dry run"

    api_json POST "${PROJECT_API}/label_ops/${op_id}/undo" '{"dry_run":false}'; expect 200
    [[ "$(jqb '.restored')" -ge 1 ]] || fail "real undo restored nothing: $(snippet)"
    ev "undo restored $(jqb '.restored'), kept_changed=$(jqb '.kept_changed') kept_holdout=$(jqb '.kept_holdout')"
    after="$(chosen_state "$ids")"
    [[ "$after" == "$before" ]] || fail "items are not back after undo: before=${before} after=${after}"
    [[ "$(guard_snapshot "$ids")" == "$guard0" ]] || fail "holdout or validated items changed by the undo"
    ev "chosen items restored; holdout and validated items unchanged"

    api_json POST "${PROJECT_API}/label_ops/${op_id}/undo" '{"dry_run":false}'; expect 409
    grep -q already_undone "$BODY" || fail "second undo is 409 but not already_undone: $(snippet)"
    ev "second undo -> 409 already_undone"
}

# Auto-label loop: a dry run plans without queueing, and the status route has its shape.
# Skips when the routes are not in this build or the project has no VLM.
ph_loop() {
    local before
    api GET "${PROJECT_API}/vlm/endpoints/active"
    [[ "$HTTP_CODE" == 200 ]] || skip "no VLM endpoint route (HTTP ${HTTP_CODE})"
    [[ -n "$(jqb '.active.name // empty')" ]] || skip "no VLM active for the project"
    api GET "${PROJECT_API}/assist/loops"
    case "$HTTP_CODE" in 404|405) skip "auto-label loop routes not in this build (HTTP ${HTTP_CODE})" ;; esac
    expect 200
    [[ "$(jqb '.loops | type')" == array ]] || fail "loop list has no loops array: $(snippet)"
    before="$(jqb '.loops | length')"
    api_json POST "${PROJECT_API}/assist/loops" "$(jq -nc --argjson n "$LOOP_BUDGET_CROPS" '{budget:{max_vlm_crops:$n}}')"
    case "$HTTP_CODE" in
        200) ;;
        409|422) skip "loop dry run refused (HTTP ${HTTP_CODE}): $(snippet)" ;;
        *) expect 200 ;;
    esac
    [[ "$(jqb '.loop_id')" == null ]] || fail "dry run returned a loop_id (it queued a loop): $(snippet)"
    [[ "$(jqb '(.plan | type) + (.request | type)')" == objectobject ]] || fail "dry run has no plan/request objects: $(snippet)"
    ev "loop dry run: status=$(jqb '.status') eligible_items=$(jqb '.plan.eligible_items // "n/a"') fits_budget=$(jqb '.plan.fits_budget // "n/a"')"
    api GET "${PROJECT_API}/assist/loops"; expect 200
    [[ "$(jqb '.loops | length')" == "$before" ]] || fail "dry run created a loop"
    ev "no loop was queued (still ${before})"
}
