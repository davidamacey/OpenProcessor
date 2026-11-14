#!/bin/bash
# =============================================================================
# Where the database, search and queue data live (OP_DATA_DIR)
# =============================================================================
# OP_DATA_DIR in .env roots the PostgreSQL, OpenSearch and Valkey data under one host
# directory through docker-compose.data-dir.yml. This library
#   - says whether that overlay applies           (data_dir_value)
#   - creates the three directories               (data_dir_prepare)
#   - refuses to start empty over an old volume   (data_dir_refuse_empty_over_volume)
#   - reports whether a path is on a rotating disk (storage_rotational, storage_warn_if_rotational)
# Needs log_warn / log_error (colors.sh or the installer's own).

DATA_DIR_SUBDIRS=(postgres opensearch valkey)
# Compose volume key per data directory (docker-compose.yml top-level `volumes:`).
declare -A DATA_DIR_VOLUME=([postgres]=openprocessor-pg [opensearch]=opensearch_data [valkey]=openprocessor-queue)
# The OpenSearch image runs as this uid; its directory must be writable by it.
DATA_DIR_OPENSEARCH_UID=1000

# data_dir_value [ENV_FILE]: the OP_DATA_DIR line of the .env (last one wins), else empty.
data_dir_value() {
    local file="${1:-${PROJECT_DIR:-.}/.env}"
    [[ -f "$file" ]] || return 0
    awk '{ i = index($0, "=") } i > 1 && substr($0, 1, i - 1) == "OP_DATA_DIR" { v = substr($0, i + 1) } END { print v }' "$file"
}

# data_dir_prepare PROJECT_DIR PROJECT: create the directories, then refuse when they would
# hide data. Prints nothing and returns 0 when OP_DATA_DIR is not set.
data_dir_prepare() {
    local root="$1" project="$2" dir sub
    dir="$(data_dir_value "$root/.env")"
    [[ -n "$dir" ]] || return 0
    if [[ "$dir" != /* ]]; then
        log_error "OP_DATA_DIR=${dir} is not an absolute path"
        return 1
    fi
    for sub in "${DATA_DIR_SUBDIRS[@]}"; do
        mkdir -p "$dir/$sub" || { log_error "cannot create ${dir}/${sub}"; return 1; }
    done
    local os_uid os_mode
    os_uid="$(stat -c %u "$dir/opensearch")"
    os_mode="$(stat -c %a "$dir/opensearch")"
    if [[ "$os_uid" != "$DATA_DIR_OPENSEARCH_UID" && "${os_mode: -1}" != [2367] ]]; then
        log_error "${dir}/opensearch is owned by uid ${os_uid}; OpenSearch runs as uid ${DATA_DIR_OPENSEARCH_UID}."
        log_error "fix: sudo chown -R ${DATA_DIR_OPENSEARCH_UID}:${DATA_DIR_OPENSEARCH_UID} ${dir}/opensearch"
        return 1
    fi
    data_dir_refuse_empty_over_volume "$project" "$dir"
}

# data_dir_volume_has_data VOLUME IMAGE: 0 when the named volume exists and is not empty.
data_dir_volume_has_data() {
    local volume="$1" image="$2"
    docker volume ls -q 2>/dev/null | grep -Fxq "$volume" || return 1
    [[ -n "$(docker run --rm --pull=never -v "${volume}:/v:ro" --entrypoint ls "$image" -A /v 2>/dev/null)" ]]
}

# data_dir_refuse_empty_over_volume PROJECT DIR: a named volume of this project holds data and
# its new directory is empty -> refuse (return 1) with the move procedure. Never copies.
data_dir_refuse_empty_over_volume() {
    local project="$1" dir="$2" sub volume image
    local -a stale=()
    for sub in "${DATA_DIR_SUBDIRS[@]}"; do
        [[ -z "$(ls -A "$dir/$sub" 2>/dev/null)" ]] || continue
        volume="${project}_${DATA_DIR_VOLUME[$sub]}"
        image="$(_data_dir_image "$sub")"
        [[ -n "$image" ]] || continue
        if data_dir_volume_has_data "$volume" "$image"; then
            stale+=("$volume -> ${dir}/${sub}")
        fi
    done
    (( ${#stale[@]} )) || return 0
    log_error "OP_DATA_DIR=${dir} is empty but these Docker volumes already hold data, so the stack would start with empty databases:"
    printf '  %s\n' "${stale[@]}" >&2
    log_error "Nothing was started. Move the data first: stop the stack, copy each volume into its directory, verify, then start."
    log_error "Steps: docs-site/docs/operations/storage.mdx (Moving the databases to fast storage). The old volumes are kept until you remove them."
    return 1
}

# _data_dir_image SUB: the image the compose file uses for that data directory's service.
_data_dir_image() {
    local svc
    case "$1" in postgres) svc=postgres ;; valkey) svc=valkey ;; opensearch) svc=opensearch ;; esac
    dc config --images "$svc" 2>/dev/null | head -n1
}

# --- rotational detection ------------------------------------------------------------------
# SYS_ROOT is overridable so a test can point at a fake /sys.
# storage_rotational PATH: prints yes | no | unknown. Any rotating member of an md, LVM or
# device-mapper stack makes the answer yes; a device the kernel gives no flag for makes it unknown.

_rota_dev() {
    local name="$1" sys="${SYS_ROOT:-/sys}" parent s r result="no" sub
    local base="$sys/class/block/$name"
    [[ -e "$base" ]] || { echo unknown; return; }
    if [[ -e "$base/partition" ]]; then
        parent="$(basename "$(dirname "$(readlink -f "$base")")")"
        _rota_dev "$parent"; return
    fi
    if [[ -d "$base/slaves" ]] && [[ -n "$(ls -A "$base/slaves" 2>/dev/null)" ]]; then
        for s in "$base"/slaves/*; do
            sub="$(_rota_dev "$(basename "$s")")"
            [[ "$sub" == yes ]] && { echo yes; return; }
            [[ "$sub" == unknown ]] && result=unknown
        done
        echo "$result"; return
    fi
    r="$(cat "$base/queue/rotational" 2>/dev/null || true)"
    case "$r" in 1) echo yes ;; 0) echo no ;; *) echo unknown ;; esac
}

storage_rotational() {
    local path="$1" src dev
    while [[ ! -e "$path" && "$path" != / ]]; do path="$(dirname -- "$path")"; done
    src="$(findmnt -n -o SOURCE --target "$path" 2>/dev/null | head -n1)" || src=""
    src="${src%%\[*}"
    [[ "$src" == /dev/* ]] || { echo unknown; return; }
    dev="$(basename "$(readlink -f "$src")")"
    _rota_dev "$dev"
}

# storage_warn_if_rotational PATH [LABEL]: a warning, never a refusal.
storage_warn_if_rotational() {
    local path="$1" label="${2:-the data location}" state
    state="$(storage_rotational "$path")"
    case "$state" in
        yes)
            log_warn "${label} (${path}) is on a rotating disk: database commits wait on the disk, so ingest is several times slower than on an SSD."
            log_warn "Set OP_DATA_DIR in .env to a directory on SSD or NVMe (docs-site/docs/operations/storage.mdx)."
            ;;
        unknown)
            log_warn "could not determine whether ${label} (${path}) is on solid-state storage; ingest is several times slower on a rotating disk (OP_DATA_DIR moves the data)."
            ;;
    esac
    return 0
}

# storage_check_data_location ROOT: warn about wherever the data really goes: OP_DATA_DIR, else
# Docker's data root (the named volumes live there).
storage_check_data_location() {
    local root="$1" dir
    dir="$(data_dir_value "$root/.env")"
    if [[ -n "$dir" ]]; then
        storage_warn_if_rotational "$dir" "OP_DATA_DIR"
    else
        dir="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
        [[ -n "$dir" ]] && storage_warn_if_rotational "$dir" "Docker's data root"
    fi
    return 0
}
