#!/bin/bash
# Cropwright release orchestrator. Local, not CI — CI minutes are exhausted and
# this is the standard workflow elsewhere in this org. Modelled on
# a similar local release pipeline elsewhere in this org, trimmed to what
# a single static-frontend image needs: no backend/GPU/multi-component legs, no
# criteria-lib/rehearse/promote/bump machinery.
#
# DESIGN RULES (same as the source this is modelled on)
#   * Every stage is independently runnable, skippable, and resumable via the
#     ledger under .release/<version>/ (gitignored) — a release that dies
#     partway through does not restart from zero.
#   * Nothing reaches the outside world implicitly. tag / publish / finish are
#     the only stages that do; each announces itself and refuses without --yes.
#   * A failing gate is overridable only with --force-<stage> "<reason>" — the
#     reason is mandatory and recorded in the ledger. No bare --force.
#
# EXIT CODES (stable)
#   0  stage passed
#   1  a gate failed (fix and re-run)
#   2  misuse: bad arguments
#   3  precondition unmet (dirty worktree, builder unreachable, no login)
#   4  aborted by the operator (declined a confirmation)
#
# Usage:
#   ./scripts/release.sh status [version]
#   ./scripts/release.sh reset <version>
#   ./scripts/release.sh preflight 0.1.0
#   ./scripts/release.sh run 0.1.0 --skip scan
#   ./scripts/release.sh run 0.1.0 --from verify --dry-run
#   ./scripts/release.sh publish 0.1.0 --yes
#   ./scripts/release.sh scan 0.1.0 --force-scan "known upstream CVE, tracked in #NN"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${BLUE}[release]${NC} $*" >&2; }
ok()   { echo -e "${GREEN}[release] OK${NC} $*" >&2; }
warn() { echo -e "${YELLOW}[release] !${NC} $*" >&2; }
err()  { echo -e "${RED}${BOLD}[release] X${NC} $*" >&2; }

EXIT_GATE=1; EXIT_MISUSE=2; EXIT_PRECONDITION=3; EXIT_ABORT=4

STAGES=(preflight verify build scan smoke tag publish finish)
EXTERNAL_STAGES="tag publish finish"

usage() { sed -n '2,29p' "$0" | sed 's/^# \?//'; }

stage_script() {
    case "$1" in
        preflight) echo "10-preflight.sh" ;;
        verify)    echo "30-verify.sh" ;;
        build)     echo "40-build.sh" ;;
        scan)      echo "50-scan.sh" ;;
        smoke)     echo "60-smoke.sh" ;;
        tag)       echo "70-tag.sh" ;;
        publish)   echo "80-publish.sh" ;;
        finish)    echo "90-finish.sh" ;;
        *)         echo "" ;;
    esac
}

# ── ledger ──────────────────────────────────────────────────────────────────
# Local run-state only, never an artifact/image/tag. .release/ is gitignored.

ledger_dir() { echo "$REPO_ROOT/.release/${1:?version}"; }

ledger_record() {
    local version="$1" stage="$2" status="$3" detail="${4:-}"
    local dir; dir="$(ledger_dir "$version")/steps"
    mkdir -p "$dir"
    {
        echo "status=$status"
        echo "when=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "operator=${USER:-unknown}"
        echo "sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
        [[ -n "$detail" ]] && echo "detail=$detail"
    } > "$dir/$stage"
    return 0
}

ledger_status() {
    local version="$1" stage="$2"
    local f; f="$(ledger_dir "$version")/steps/$stage"
    [[ -f "$f" ]] && grep -m1 '^status=' "$f" | cut -d= -f2 || echo "pending"
}

cmd_status() {
    local version="${1:?status needs a version}"
    echo -e "${BOLD}Release $version${NC}"
    printf '  %-10s %s\n' "STAGE" "STATUS"
    for stage in "${STAGES[@]}"; do
        local st; st="$(ledger_status "$version" "$stage")"
        printf '  %-10s %s\n' "$stage" "$st"
    done
}

cmd_reset() {
    local version="${1:?reset needs a version}"
    local dir; dir="$(ledger_dir "$version")"
    [[ -d "$dir" ]] || { ok "no ledger for $version"; return 0; }
    warn "this clears the release ledger for $version (artifacts/images/tags untouched)"
    if [[ "${ASSUME_YES:-false}" != "true" ]]; then
        read -r -p "Clear the ledger for $version? [y/N] " reply
        [[ "$reply" == "y" || "$reply" == "Y" ]] || { err "aborted"; return $EXIT_ABORT; }
    fi
    rm -rf "${dir:?}/steps"
    ok "ledger cleared for $version"
}

# ── stage dispatch ───────────────────────────────────────────────────────────

run_stage() {
    local version="$1" stage="$2"
    local script_name; script_name="$(stage_script "$stage")"
    [[ -n "$script_name" ]] || { err "unknown stage '$stage'"; return $EXIT_MISUSE; }
    local script="$SCRIPT_DIR/release/$script_name"
    [[ -x "$script" ]] || { err "missing or non-executable: $script"; return $EXIT_MISUSE; }

    if [[ " $EXTERNAL_STAGES " == *" $stage "* ]]; then
        warn "stage '$stage' changes state OUTSIDE this repository"
        if [[ "${ASSUME_YES:-false}" != "true" ]]; then
            read -r -p "Proceed with '$stage' for $version? [y/N] " reply
            [[ "$reply" == "y" || "$reply" == "Y" ]] || {
                ledger_record "$version" "$stage" "aborted" "operator declined confirmation"
                err "$stage aborted — nothing ran"
                return $EXIT_ABORT
            }
        fi
    fi

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        log "DRY RUN would execute: $script $version"
        return 0
    fi

    log "-- $stage --"
    local rc=0
    RELEASE_VERSION="$version" "$script" "$version" || rc=$?

    case $rc in
        0) ledger_record "$version" "$stage" "done"; ok "$stage" ;;
        "$EXIT_ABORT")
            ledger_record "$version" "$stage" "aborted" "exit=$rc"
            err "$stage aborted (exit $rc)" ;;
        *)
            if [[ -n "${FORCE_REASON[$stage]:-}" ]]; then
                ledger_record "$version" "$stage" "overridden" \
                    "exit=$rc; operator=${USER:-unknown}; reason=${FORCE_REASON[$stage]}"
                warn "$stage FAILED (exit $rc) and was overridden by ${USER:-unknown}: ${FORCE_REASON[$stage]}"
                rc=0
            else
                ledger_record "$version" "$stage" "failed" "exit=$rc"
                err "$stage failed (exit $rc)"
                [[ $rc -eq $EXIT_PRECONDITION ]] || rc=$EXIT_GATE
            fi ;;
    esac
    return $rc
}

cmd_run() {
    local version="$1"; shift
    local -a to_run=()
    local started=false
    for stage in "${STAGES[@]}"; do
        [[ -n "$FROM_STAGE" && "$started" == false && "$stage" != "$FROM_STAGE" ]] && continue
        started=true
        if [[ ",$SKIP_STAGES," == *",$stage,"* ]]; then
            log "skipping $stage (--skip)"
            ledger_record "$version" "$stage" "skipped" "--skip"
            continue
        fi
        to_run+=("$stage")
    done
    [[ ${#to_run[@]} -gt 0 ]] || { err "no stages selected"; return $EXIT_MISUSE; }
    log "stages: ${to_run[*]}"
    for stage in "${to_run[@]}"; do
        run_stage "$version" "$stage" || return $?
    done
    ok "all selected stages complete"
}

# ── arg parsing ──────────────────────────────────────────────────────────────

COMMAND="${1:-}"; shift || true
SKIP_STAGES=""; FROM_STAGE=""; DRY_RUN=false; ASSUME_YES=false
POSITIONAL=()
declare -A FORCE_REASON=()

while (( $# > 0 )); do
    case "$1" in
        --skip)    SKIP_STAGES="$2"; shift 2 ;;
        --from)    FROM_STAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --yes)     ASSUME_YES=true; shift ;;
        --force-*)
            _fstage="${1#--force-}"
            [[ -n "$(stage_script "$_fstage")" ]] || { err "--force-${_fstage}: not a stage"; exit $EXIT_MISUSE; }
            [[ $# -ge 2 && -n "${2:-}" && "${2:0:1}" != "-" ]] || {
                err "--force-${_fstage} requires a reason string"; exit $EXIT_MISUSE; }
            FORCE_REASON["$_fstage"]="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*)        err "unknown option: $1"; exit $EXIT_MISUSE ;;
        *)         POSITIONAL+=("$1"); shift ;;
    esac
done
export DRY_RUN ASSUME_YES

case "$COMMAND" in
    ""|help|-h|--help) usage; exit 0 ;;
    status) cmd_status "${POSITIONAL[0]:?status needs a version}" ;;
    reset)  cmd_reset "${POSITIONAL[0]:?reset needs a version}" ;;
    run)
        [[ ${#POSITIONAL[@]} -ge 1 ]] || { err "run needs a version, e.g. run 0.1.0"; exit $EXIT_MISUSE; }
        cmd_run "${POSITIONAL[0]}" ;;
    preflight|verify|build|scan|smoke|tag|publish|finish)
        version="${POSITIONAL[0]:?$COMMAND needs a version}"
        run_stage "$version" "$COMMAND" ;;
    *)
        err "unknown command: $COMMAND"; usage; exit $EXIT_MISUSE ;;
esac
