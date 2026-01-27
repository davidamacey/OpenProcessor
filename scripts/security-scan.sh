#!/bin/bash
set -e

# =============================================================================
# Security Scanning Script for OpenProcessor Docker Images
# Uses free, open-source tools to scan for vulnerabilities and security issues
# No Docker Hub/Scout subscription required
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-davidamacey}"
REPO_API="${DOCKERHUB_USERNAME}/openprocessor"
REPO_TRITON="${DOCKERHUB_USERNAME}/openprocessor-triton"
SCAN_TARGET="${1:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/security-reports}"
SEVERITY_THRESHOLD="${SEVERITY_THRESHOLD:-MEDIUM}"
FAIL_ON_CRITICAL="${FAIL_ON_CRITICAL:-true}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# Output Functions
# =============================================================================

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

# =============================================================================
# Tool Installation Functions
# =============================================================================

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

install_trivy() {
    if command_exists trivy; then
        print_info "Trivy already installed: $(trivy --version | head -1)"
        return 0
    fi

    print_warning "Trivy not found. Installing..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command_exists brew; then
            brew install trivy
        else
            print_error "Homebrew not found. Please install Trivy manually."
            return 1
        fi
    else
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    fi

    print_success "Trivy installed successfully"
}

install_grype() {
    if command_exists grype; then
        print_info "Grype already installed: $(grype version | head -1)"
        return 0
    fi

    print_warning "Grype not found. Installing..."
    curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
    print_success "Grype installed successfully"
}

install_syft() {
    if command_exists syft; then
        print_info "Syft already installed: $(syft version | head -1)"
        return 0
    fi

    print_warning "Syft not found. Installing..."
    curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
    print_success "Syft installed successfully"
}

install_hadolint() {
    if command_exists hadolint; then
        print_info "Hadolint already installed: $(hadolint --version)"
        return 0
    fi

    print_warning "Hadolint not found. Installing..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command_exists brew; then
            brew install hadolint
        else
            print_error "Homebrew not found. Please install Hadolint manually."
            return 1
        fi
    else
        HADOLINT_VERSION=$(curl -s https://api.github.com/repos/hadolint/hadolint/releases/latest | grep '"tag_name":' | sed -E 's/.*"v([^"]+)".*/\1/')
        curl -sL -o /usr/local/bin/hadolint "https://github.com/hadolint/hadolint/releases/download/v${HADOLINT_VERSION}/hadolint-Linux-x86_64"
        chmod +x /usr/local/bin/hadolint
    fi

    print_success "Hadolint installed successfully"
}

check_dockle() {
    if ! command_exists docker; then
        print_warning "Docker not found. Dockle requires Docker to run."
        return 1
    fi

    print_info "Dockle will run via Docker image (no installation needed)"
    return 0
}

# =============================================================================
# Scanning Functions
# =============================================================================

lint_dockerfile() {
    local dockerfile=$1
    local component=$2

    print_header "Linting Dockerfile: ${dockerfile}"

    local output_file="${OUTPUT_DIR}/${component}-hadolint.txt"

    if hadolint "${dockerfile}" | tee "${output_file}"; then
        print_success "Dockerfile passed Hadolint checks"
        return 0
    else
        print_warning "Dockerfile has linting issues (see ${output_file})"
        return 1
    fi
}

run_dockle() {
    local image=$1
    local component=$2

    print_header "Running Dockle (CIS Docker Benchmark) on ${image}"

    local output_file="${OUTPUT_DIR}/${component}-dockle.json"
    local abs_output_dir
    abs_output_dir=$(cd "${OUTPUT_DIR}" && pwd)

    local dockle_args=(
        --timeout 300s
        --exit-code 1
        --exit-level WARN
        --ignore CIS-DI-0005
        --ignore DKL-DI-0006
        --format json
        --output "/output/${component}-dockle.json"
    )

    if docker run --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "${abs_output_dir}:/output" \
        goodwithtech/dockle:latest \
        "${dockle_args[@]}" \
        "${image}"; then
        print_success "Dockle scan completed (see ${output_file})"
        return 0
    else
        print_warning "Dockle found issues (see ${output_file})"
        return 1
    fi
}

generate_sbom() {
    local image=$1
    local component=$2

    print_header "Generating SBOM for ${image}"

    local sbom_file="${OUTPUT_DIR}/${component}-sbom.json"

    syft "${image}" -o cyclonedx-json > "${sbom_file}"
    print_success "SBOM generated: ${sbom_file}"

    # Also generate human-readable table format
    syft "${image}" -o table > "${OUTPUT_DIR}/${component}-sbom.txt"
    print_info "Human-readable SBOM: ${OUTPUT_DIR}/${component}-sbom.txt"

    echo "${sbom_file}"
}

scan_trivy() {
    local image=$1
    local component=$2

    print_header "Scanning ${image} with Trivy"

    local json_output="${OUTPUT_DIR}/${component}-trivy.json"
    local txt_output="${OUTPUT_DIR}/${component}-trivy.txt"

    # Run Trivy scan with JSON output
    trivy image \
        --severity "${SEVERITY_THRESHOLD},HIGH,CRITICAL" \
        --format json \
        --output "${json_output}" \
        --quiet \
        "${image}"

    # Generate table format
    trivy convert --format table --output "${txt_output}" "${json_output}" 2>/dev/null || \
        trivy image --severity "${SEVERITY_THRESHOLD},HIGH,CRITICAL" --format table --output "${txt_output}" "${image}"

    print_success "Trivy reports generated:"
    print_info "  - JSON: ${json_output}"
    print_info "  - Text: ${txt_output}"

    # Check for CRITICAL vulnerabilities
    local critical_count
    local high_count
    critical_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "${json_output}")
    high_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "${json_output}")

    print_info "Found ${critical_count} CRITICAL and ${high_count} HIGH severity vulnerabilities"

    if [ "${FAIL_ON_CRITICAL}" = "true" ] && [ "${critical_count}" -gt 0 ]; then
        print_error "CRITICAL vulnerabilities found - scan failed"
        return 1
    fi

    return 0
}

scan_grype() {
    local image=$1
    local component=$2
    local sbom_file=$3

    print_header "Scanning with Grype"

    local json_output="${OUTPUT_DIR}/${component}-grype.json"
    local txt_output="${OUTPUT_DIR}/${component}-grype.txt"

    # Scan from SBOM for speed
    if [ -n "${sbom_file}" ] && [ -f "${sbom_file}" ]; then
        print_info "Scanning from SBOM for faster results..."
        grype "sbom:${sbom_file}" \
            --output json \
            --file "${json_output}"

        grype "sbom:${sbom_file}" \
            --output table \
            | tee "${txt_output}"
    else
        grype "${image}" \
            --output json \
            --file "${json_output}"

        grype "${image}" \
            --output table \
            | tee "${txt_output}"
    fi

    print_success "Grype reports generated:"
    print_info "  - JSON: ${json_output}"
    print_info "  - Text: ${txt_output}"

    # Check for CRITICAL vulnerabilities
    local critical_count
    local high_count
    critical_count=$(jq '[.matches[]? | select(.vulnerability.severity == "Critical")] | length' "${json_output}")
    high_count=$(jq '[.matches[]? | select(.vulnerability.severity == "High")] | length' "${json_output}")

    print_info "Found ${critical_count} Critical and ${high_count} High severity vulnerabilities"

    if [ "${FAIL_ON_CRITICAL}" = "true" ] && [ "${critical_count}" -gt 0 ]; then
        print_error "CRITICAL vulnerabilities found - scan failed"
        return 1
    fi

    return 0
}

# =============================================================================
# Component Scanning (Parallel Execution)
# =============================================================================

scan_component() {
    local component=$1
    local dockerfile=""
    local image=""

    case "${component}" in
        api)
            dockerfile="${PROJECT_DIR}/Dockerfile"
            image="${REPO_API}:latest"
            ;;
        triton)
            dockerfile="${PROJECT_DIR}/Dockerfile.triton"
            image="${REPO_TRITON}:latest"
            ;;
        *)
            print_error "Invalid component: ${component}"
            return 1
            ;;
    esac

    print_header "Security Scanning: ${component}"
    print_info "Image: ${image}"
    print_info "Dockerfile: ${dockerfile}"
    print_info "Running tools in PARALLEL for speed..."
    echo ""

    # Check if image exists locally
    if ! docker image inspect "${image}" >/dev/null 2>&1; then
        print_warning "Image not found locally: ${image}"
        print_info "Building image first..."

        if [ -f "${dockerfile}" ]; then
            docker build -f "${dockerfile}" -t "${image}" "${PROJECT_DIR}"
        else
            print_error "Dockerfile not found: ${dockerfile}"
            return 1
        fi
    fi

    # Create status directory for parallel job tracking
    local status_dir
    status_dir=$(mktemp -d)

    # === PARALLEL PHASE 1: Hadolint + Dockle + SBOM ===

    # Hadolint (fast - Dockerfile only)
    if [ -f "${dockerfile}" ]; then
        (
            lint_dockerfile "${dockerfile}" "${component}"
            echo $? > "${status_dir}/hadolint.status"
        ) &
    else
        echo "0" > "${status_dir}/hadolint.status"
    fi

    # Dockle (medium speed)
    (
        check_dockle && run_dockle "${image}" "${component}"
        echo $? > "${status_dir}/dockle.status"
    ) &

    # SBOM generation (needed for Grype)
    (
        generate_sbom "${image}" "${component}" > "${status_dir}/sbom_path.txt"
        echo $? > "${status_dir}/sbom.status"
    ) &

    # Wait for Phase 1
    wait
    print_info "Phase 1 complete (Hadolint, Dockle, SBOM)"

    # Get SBOM path for Grype
    local sbom_file=""
    if [ -f "${status_dir}/sbom_path.txt" ]; then
        sbom_file=$(cat "${status_dir}/sbom_path.txt")
    fi

    # === PARALLEL PHASE 2: Trivy + Grype ===
    print_info "Phase 2: Running Trivy and Grype in parallel..."

    (
        scan_trivy "${image}" "${component}"
        echo $? > "${status_dir}/trivy.status"
    ) &

    (
        scan_grype "${image}" "${component}" "${sbom_file}"
        echo $? > "${status_dir}/grype.status"
    ) &

    # Wait for Phase 2
    wait
    print_info "Phase 2 complete (Trivy, Grype)"

    # Collect results
    local exit_code=0
    for tool in hadolint dockle sbom trivy grype; do
        if [ -f "${status_dir}/${tool}.status" ]; then
            local status
            status=$(cat "${status_dir}/${tool}.status")
            if [ "$status" != "0" ]; then
                exit_code=1
            fi
        fi
    done

    # Cleanup
    rm -rf "${status_dir}"

    echo ""
    if [ ${exit_code} -eq 0 ]; then
        print_success "Security scan completed for ${component}"
    else
        print_warning "Security scan completed with issues for ${component}"
    fi

    return ${exit_code}
}

# =============================================================================
# Summary and Help
# =============================================================================

generate_summary() {
    print_header "Security Scan Summary"

    print_info "All reports saved to: ${OUTPUT_DIR}"
    echo ""

    print_info "Report files:"
    find "${OUTPUT_DIR}" -maxdepth 1 -type f -exec ls -lh {} \; | awk '{printf "  %-50s %8s\n", $9, $5}'
    echo ""
}

show_usage() {
    cat << EOF
Usage: $0 [OPTION]

Security scanning for OpenProcessor Docker images using free, open-source tools

Tools used:
  - Hadolint: Dockerfile linter
  - Dockle: Container image CIS best practices checker
  - Syft: SBOM (Software Bill of Materials) generator
  - Trivy: Comprehensive vulnerability scanner
  - Grype: Fast vulnerability scanner

Options:
    api         Scan only API image (yolo-api/FastAPI)
    triton      Scan only Triton image (inference server)
    all         Scan both images (default)
    install     Install all required tools
    help        Show this help message

Environment Variables:
    OUTPUT_DIR              Report output directory (default: ./security-reports)
    SEVERITY_THRESHOLD      Minimum severity to report (default: MEDIUM)
    FAIL_ON_CRITICAL        Fail if CRITICAL vulnerabilities found (default: true)
    DOCKERHUB_USERNAME      Docker Hub username (default: davidamacey)

Examples:
    $0                      # Scan both images
    $0 api                  # Scan only API image
    $0 install              # Install all required tools

    # Customize scanning
    OUTPUT_DIR=./reports SEVERITY_THRESHOLD=HIGH $0 all
    FAIL_ON_CRITICAL=false $0 api

Reports:
    All reports are saved to \${OUTPUT_DIR}/ with multiple formats:
    - *-hadolint.txt: Dockerfile linting results
    - *-dockle.json: CIS best practices check
    - *-sbom.json: Software Bill of Materials (CycloneDX format)
    - *-trivy.json: Trivy vulnerability scan (JSON)
    - *-trivy.txt: Trivy vulnerability scan (human-readable)
    - *-grype.json: Grype vulnerability scan (JSON)
    - *-grype.txt: Grype vulnerability scan (human-readable)

EOF
}

install_all_tools() {
    print_header "Installing Security Scanning Tools"

    install_trivy
    install_grype
    install_syft
    install_hadolint
    check_dockle

    print_success "All tools installed successfully!"
    echo ""
    print_info "Tool versions:"
    command_exists trivy && trivy --version | head -1
    command_exists grype && grype version | head -1
    command_exists syft && syft version | head -1
    command_exists hadolint && hadolint --version
    print_info "Dockle: runs via Docker image"
}

# =============================================================================
# Main
# =============================================================================

main() {
    print_header "OpenProcessor Security Scanner"
    print_info "Output directory: ${OUTPUT_DIR}"
    print_info "Severity threshold: ${SEVERITY_THRESHOLD}"
    print_info "Fail on critical: ${FAIL_ON_CRITICAL}"
    echo ""

    case "${SCAN_TARGET}" in
        install)
            install_all_tools
            exit 0
            ;;
        help|--help|-h)
            show_usage
            exit 0
            ;;
        api|triton)
            # Check required tools
            install_trivy
            install_grype
            install_syft
            install_hadolint
            check_dockle

            scan_component "${SCAN_TARGET}"
            exit_code=$?
            ;;
        all)
            # Check required tools
            install_trivy
            install_grype
            install_syft
            install_hadolint
            check_dockle

            scan_component "api"
            api_exit=$?

            scan_component "triton"
            triton_exit=$?

            exit_code=$((api_exit + triton_exit))
            ;;
        *)
            print_error "Invalid option: ${SCAN_TARGET}"
            show_usage
            exit 1
            ;;
    esac

    echo ""
    generate_summary

    if [ ${exit_code} -eq 0 ]; then
        print_success "All security scans passed!"
        exit 0
    else
        print_error "Security scans failed"
        exit 1
    fi
}

# Run main function
main
