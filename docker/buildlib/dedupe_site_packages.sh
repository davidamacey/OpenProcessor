#!/bin/bash
# Build-time helper: replace the entries of a duplicate site-packages that are byte-identical to
# the canonical one with absolute symlinks into the canonical RUNTIME path.
#
#   dedupe_site_packages.sh DUP_SITE CANON_BUILD_SITE CANON_RUNTIME_SITE
#
# DUP_SITE is the isolated export venv's site-packages. CANON_BUILD_SITE is where the canonical
# packages sit while building; CANON_RUNTIME_SITE is where they will sit in the final image, so
# the links dangle by design until the stages are assembled. Not --system-site-packages: the
# user site would shadow the venv's pinned ultralytics. Skipped: bytecode caches, dist-info
# metadata (the venv's installer records must stay real) and anything already a symlink.
set -euo pipefail

[[ $# -eq 3 && -d "$1" && -d "$2" ]] || { echo "usage: $0 DUP_SITE CANON_BUILD_SITE CANON_RUNTIME_SITE" >&2; exit 2; }
dup="$1" canon_build="$2" canon_runtime="${3%/}"

linked=0
for entry in "$dup"/* "$dup"/.[!.]*; do
    [[ -e "$entry" || -L "$entry" ]] || continue
    name="$(basename "$entry")"
    case "$name" in
        __pycache__ | *.dist-info) continue ;;
    esac
    [[ -L "$entry" ]] && continue
    twin="$canon_build/$name"
    [[ -e "$twin" && ! -L "$twin" ]] || continue
    if [[ -d "$entry" ]]; then
        [[ -d "$twin" ]] || continue
        diff -rq --no-dereference -x __pycache__ "$entry" "$twin" > /dev/null || continue
    else
        cmp -s "$entry" "$twin" || continue
    fi
    rm -rf "$entry"
    ln -s "$canon_runtime/$name" "$entry"
    linked=$((linked + 1))
done
echo "dedupe: $linked entries linked to $canon_runtime" >&2
