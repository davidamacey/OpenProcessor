#!/usr/bin/env bash
# =============================================================================
# Fake trainer for the live verification harness.
#
# The API <-> trainer protocol is a pure shared-volume file protocol
# (src/services/training/jobs.py):
#
#   API     writes  <jobs_dir>/<job_id>.job.json      to start a run
#   trainer writes  <jobs_dir>/<job_id>.status.json   while it runs
#   API     writes  <jobs_dir>/<job_id>.cancel        to cancel
#   trainer appends <jobs_dir>/<job_id>.run.log       for the log-tail route
#   trainer writes  <jobs_dir>/<job_id>.manifest.json at the end
#
# This script implements the trainer half and nothing else: it never
# touches a GPU, never loads a model, and never reads OpenSearch. It
# exists so the seven training write endpoints can be exercised against a
# real counterparty instead of a mock.
#
# Lifecycle per job: queued -> running (N epochs, one per poll) -> finished.
# A `.cancel` sentinel observed in any non-terminal state -> cancelled.
# =============================================================================
set -euo pipefail

JOBS_DIR="${OP_TRAIN_JOBS_DIR:-/jobs}"
POLL_SECONDS="${FAKE_TRAINER_POLL_SECONDS:-1}"
TOTAL_EPOCHS="${FAKE_TRAINER_EPOCHS:-3}"
# Per-job progress counters live outside the shared volume so the tests
# only ever see protocol files there.
PROGRESS_DIR="/tmp/fake-trainer-progress"

mkdir -p "$JOBS_DIR" "$PROGRESS_DIR"

now_iso() {
    date -u +"%Y-%m-%dT%H:%M:%S+00:00"
}

write_status() {
    # write_status <job_id> <state> <current_epoch> [extra_json]
    local job_id="$1" state="$2" epoch="$3" extra="${4:-}"
    local ts
    ts="$(now_iso)"
    local tmp="${JOBS_DIR}/${job_id}.status.json.tmp"
    {
        printf '{\n'
        printf '  "job_id": "%s",\n' "$job_id"
        printf '  "state": "%s",\n' "$state"
        printf '  "started_at": "%s",\n' "$ts"
        printf '  "current_epoch": %s,\n' "$epoch"
        printf '  "total_epochs": %s,\n' "$TOTAL_EPOCHS"
        printf '  "heartbeat_at": "%s",\n' "$ts"
        if [ -n "$extra" ]; then
            printf '%s,\n' "$extra"
        fi
        printf '  "trainer": "fake"\n'
        printf '}\n'
    } > "$tmp"
    mv "$tmp" "${JOBS_DIR}/${job_id}.status.json"
    printf '[%s] %s state=%s epoch=%s\n' "$ts" "$job_id" "$state" "$epoch" \
        >> "${JOBS_DIR}/${job_id}.run.log"
}

write_manifest() {
    local job_id="$1"
    local tmp="${JOBS_DIR}/${job_id}.manifest.json.tmp"
    cat > "$tmp" <<EOF
{
  "job_id": "${job_id}",
  "trainer": "fake",
  "finished_at": "$(now_iso)",
  "lineage": {"dataset_sha": null, "class_remap": null},
  "eval": {"map50": 0.5, "map50_95": 0.25}
}
EOF
    mv "$tmp" "${JOBS_DIR}/${job_id}.manifest.json"
}

terminal_state() {
    # A job whose status.json already reports a terminal state is done.
    local job_id="$1"
    local status_file="${JOBS_DIR}/${job_id}.status.json"
    [ -f "$status_file" ] || return 1
    grep -qE '"state": "(finished|failed|cancelled|skipped)"' "$status_file"
}

process_job() {
    local job_id="$1"
    local progress_file="${PROGRESS_DIR}/${job_id}"

    if terminal_state "$job_id"; then
        return 0
    fi

    if [ -f "${JOBS_DIR}/${job_id}.cancel" ]; then
        write_status "$job_id" "cancelled" "$(cat "$progress_file" 2>/dev/null || echo 0)"
        return 0
    fi

    if [ ! -f "$progress_file" ]; then
        echo 0 > "$progress_file"
        write_status "$job_id" "queued" 0
        return 0
    fi

    local epoch
    epoch="$(cat "$progress_file")"
    epoch=$((epoch + 1))
    echo "$epoch" > "$progress_file"

    if [ "$epoch" -le "$TOTAL_EPOCHS" ]; then
        write_status "$job_id" "running" "$epoch"
    else
        write_status "$job_id" "finished" "$TOTAL_EPOCHS" \
            '  "checkpoint_path": "/tmp/fake-run/weights/best.pt",
  "eval": {"map50": 0.5, "map50_95": 0.25}'
        write_manifest "$job_id"
    fi
}

printf 'fake-trainer watching %s (poll=%ss, epochs=%s)\n' \
    "$JOBS_DIR" "$POLL_SECONDS" "$TOTAL_EPOCHS"

while true; do
    shopt -s nullglob
    for job_file in "${JOBS_DIR}"/*.job.json; do
        base="$(basename "$job_file")"
        process_job "${base%.job.json}"
    done
    shopt -u nullglob
    sleep "$POLL_SECONDS"
done
