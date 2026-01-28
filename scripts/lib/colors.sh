#!/bin/bash
# =============================================================================
# colors.sh - Logging utilities and color definitions for OpenProcessor scripts
# =============================================================================

# Avoid redefining if already sourced
[[ -n "${_COLORS_SH_LOADED:-}" ]] && return 0
_COLORS_SH_LOADED=1

# Color definitions
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export CYAN='\033[0;36m'
export MAGENTA='\033[0;35m'
export BOLD='\033[1m'
export DIM='\033[2m'
export NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-0}" == "1" ]]; then
        echo -e "${DIM}[DEBUG] $1${NC}"
    fi
}

# Print a header banner
print_header() {
    local text="$1"
    local width=60
    echo ""
    echo -e "${BOLD}$(printf '=%.0s' $(seq 1 $width))${NC}"
    echo -e "${BOLD} $text${NC}"
    echo -e "${BOLD}$(printf '=%.0s' $(seq 1 $width))${NC}"
    echo ""
}

# Print a section divider
print_section() {
    local text="$1"
    echo ""
    echo -e "${CYAN}--- $text ---${NC}"
    echo ""
}

# =============================================================================
# Progress Spinner
# =============================================================================

# Spinner characters
SPINNER_CHARS='|/-\'

# Start a spinner in the background
# Usage: start_spinner "message"
start_spinner() {
    local msg="$1"
    local pid_file="/tmp/spinner_$$_pid"

    (
        local i=0
        while true; do
            printf "\r${BLUE}[%c]${NC} %s" "${SPINNER_CHARS:i++%4:1}" "$msg"
            sleep 0.1
        done
    ) &
    echo $! > "$pid_file"
}

# Stop the spinner
# Usage: stop_spinner [success|error] "completion message"
stop_spinner() {
    local status="${1:-success}"
    local msg="$2"
    local pid_file="/tmp/spinner_$$_pid"

    if [[ -f "$pid_file" ]]; then
        kill "$(cat "$pid_file")" 2>/dev/null || true
        rm -f "$pid_file"
    fi

    # Clear the line
    printf "\r%*s\r" 80 ""

    if [[ "$status" == "success" ]]; then
        [[ -n "$msg" ]] && log_success "$msg"
    else
        [[ -n "$msg" ]] && log_error "$msg"
    fi
}

# =============================================================================
# Progress Bar
# =============================================================================

# Print a progress bar
# Usage: print_progress current total "message"
print_progress() {
    local current="$1"
    local total="$2"
    local msg="${3:-}"
    local width=40
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    printf "\r${BLUE}[${NC}"
    printf "%${filled}s" '' | tr ' ' '#'
    printf "%${empty}s" '' | tr ' ' '-'
    printf "${BLUE}]${NC} %3d%% %s" "$percent" "$msg"

    if [[ "$current" -eq "$total" ]]; then
        echo ""
    fi
}

# =============================================================================
# User Prompts
# =============================================================================

# Ask a yes/no question with default
# Usage: confirm "Question?" [Y|N]
confirm() {
    local prompt="$1"
    local default="${2:-Y}"
    local reply

    if [[ "$default" == "Y" ]]; then
        prompt="$prompt [Y/n]: "
    else
        prompt="$prompt [y/N]: "
    fi

    read -r -p "$prompt" reply
    reply="${reply:-$default}"

    [[ "$reply" =~ ^[Yy] ]]
}

# Ask user to select from options
# Usage: select_option "prompt" option1 option2 option3
select_option() {
    local prompt="$1"
    shift
    local options=("$@")
    local selection

    echo -e "${CYAN}$prompt${NC}"
    PS3="Select option: "

    select selection in "${options[@]}"; do
        if [[ -n "$selection" ]]; then
            echo "$selection"
            return 0
        fi
    done
}
