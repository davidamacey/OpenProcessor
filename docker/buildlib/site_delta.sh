#!/bin/bash
# Build-time helper: ship an extra dependency set as a file-level delta over a runtime tree.
#
#   site_delta.sh snapshot ROOT MANIFEST      record every file and symlink under ROOT
#   site_delta.sh delta ROOT MANIFEST DEST    copy what was added since the snapshot into DEST
#
# The delta stage of the Dockerfile COPYs DEST over the runtime image's tree, so the export and
# evaluator images add only their own packages instead of repeating the runtime set. A file that
# existed at snapshot time and changed or vanished fails the build: the extra install must never
# rewrite a runtime package. Bytecode caches are ignored (they are regenerated).
set -euo pipefail

usage() {
    echo "usage: $0 snapshot ROOT MANIFEST | delta ROOT MANIFEST DEST" >&2
    exit 2
}

list_entries() {
    # One "kind<TAB>sha-or-target<TAB>path" line per file or symlink, sorted. Batched through
    # xargs: a process per file made a 50,000-file tree take minutes.
    local root="$1"
    (
        cd "$root"
        find . -name __pycache__ -prune -o -type l -printf 'L\t%l\t%p\n'
        find . -name __pycache__ -prune -o -type f -print0 | xargs -0 -r sha256sum |
            sed -E 's/^([0-9a-f]{64})  (.*)$/F\t\1\t\2/'
    ) | LC_ALL=C sort
}

cmd="${1:-}"
case "$cmd" in
    snapshot)
        [[ $# -eq 3 && -d "$2" ]] || usage
        list_entries "$2" > "$3"
        ;;
    delta)
        [[ $# -eq 4 && -d "$2" && -f "$3" ]] || usage
        root="$2" manifest="$3" dest="$4"
        now="$(mktemp)"
        trap 'rm -f "$now"' EXIT
        list_entries "$root" > "$now"
        # Lines only in the snapshot are files that changed or disappeared.
        changed="$(LC_ALL=C comm -23 <(LC_ALL=C sort "$manifest") <(LC_ALL=C sort "$now") || true)"
        if [[ -n "$changed" ]]; then
            echo "the extra install changed or removed runtime files:" >&2
            echo "$changed" | cut -f3 | head -50 >&2
            exit 1
        fi
        mkdir -p "$dest"
        added="$(mktemp)"
        trap 'rm -f "$now" "$added"' EXIT
        LC_ALL=C comm -13 <(cut -f3 "$manifest" | LC_ALL=C sort) <(cut -f3 "$now" | LC_ALL=C sort) > "$added"
        if [[ -s "$added" ]]; then
            tar -C "$root" --no-recursion -T "$added" -cf - | tar -C "$dest" -xf -
        fi
        echo "delta: $(wc -l < "$added") files -> $dest" >&2
        ;;
    *) usage ;;
esac
