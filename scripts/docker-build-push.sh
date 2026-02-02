#!/bin/bash
set -e

# =============================================================================
# Docker Build and Push Script for OpenProcessor
# Builds and pushes Docker images to Docker Hub with security scanning
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

# Get commit SHA for tagging
COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")

# Default to building both platforms
PLATFORMS="${PLATFORMS:-linux/amd64}"  # ARM64 not typically needed for GPU workloads
BUILD_TARGET="${1:-all}"

# Builder configuration
DEFAULT_BUILDER_NAME="openprocessor-builder"

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
# Prerequisites
# =============================================================================

check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

check_docker_login() {
    if ! docker info | grep -q "Username"; then
        print_warning "Not logged into Docker Hub. Attempting login..."
        docker login
    else
        print_success "Already logged into Docker Hub"
    fi
}

# =============================================================================
# Security Tool Database Updates
# =============================================================================

update_security_tools() {
    print_info "Updating security scanning tool databases..."

    # Update Trivy vulnerability database
    if command -v trivy &> /dev/null; then
        print_info "Updating Trivy vulnerability database..."
        trivy image --download-db-only --quiet 2>/dev/null || true
        print_success "Trivy database updated"
    else
        print_warning "Trivy not found, skipping database update"
    fi

    # Update Grype vulnerability database
    if command -v grype &> /dev/null; then
        print_info "Updating Grype vulnerability database..."
        grype db update --quiet 2>/dev/null || true
        print_success "Grype database updated"
    else
        print_warning "Grype not found, skipping database update"
    fi

    print_success "Security tool updates complete!"
}

# =============================================================================
# Security Scanning
# =============================================================================

run_security_scan() {
    local component=$1

    if [ "${SKIP_SECURITY_SCAN}" = "true" ]; then
        print_warning "Security scanning skipped (SKIP_SECURITY_SCAN=true)"
        return 0
    fi

    if [ ! -f "${SCRIPT_DIR}/security-scan.sh" ]; then
        print_warning "Security scan script not found, skipping..."
        return 0
    fi

    print_info "Running security scan for ${component}..."
    if OUTPUT_DIR="${PROJECT_DIR}/security-reports" FAIL_ON_CRITICAL="${FAIL_ON_CRITICAL:-false}" "${SCRIPT_DIR}/security-scan.sh" "${component}"; then
        print_success "Security scan passed for ${component}"
        return 0
    else
        print_warning "Security scan found issues for ${component}"
        if [ "${FAIL_ON_SECURITY_ISSUES}" = "true" ]; then
            print_error "Failing build due to security issues (FAIL_ON_SECURITY_ISSUES=true)"
            return 1
        else
            print_warning "Continuing despite security issues (set FAIL_ON_SECURITY_ISSUES=true to fail)"
            return 0
        fi
    fi
}

# =============================================================================
# Build Functions
# =============================================================================

build_api() {
    print_header "Building API Image (yolo-api)"
    print_info "Platforms: ${PLATFORMS}"
    print_info "Version: ${VERSION_FULL}"
    print_info "Tags: latest, ${VERSION_FULL}"

    cd "${PROJECT_DIR}"

    # Build and push multi-arch image
    docker buildx build \
        --platform "${PLATFORMS}" \
        --file Dockerfile \
        --tag "${REPO_API}:latest" \
        --tag "${REPO_API}:${VERSION_FULL}" \
        ${CACHE_FLAG} \
        --push \
        .

    print_success "API image built and pushed successfully"
    print_info "Tags pushed:"
    print_info "  - ${REPO_API}:latest"
    print_info "  - ${REPO_API}:${VERSION_FULL}"
}

build_triton() {
    print_header "Building Triton Image (inference server)"
    print_info "Platforms: ${PLATFORMS}"
    print_info "Version: ${VERSION_FULL}"
    print_info "Tags: latest, ${VERSION_FULL}"

    cd "${PROJECT_DIR}"

    # Build and push multi-arch image
    docker buildx build \
        --platform "${PLATFORMS}" \
        --file Dockerfile.triton \
        --tag "${REPO_TRITON}:latest" \
        --tag "${REPO_TRITON}:${VERSION_FULL}" \
        ${CACHE_FLAG} \
        --push \
        .

    print_success "Triton image built and pushed successfully"
    print_info "Tags pushed:"
    print_info "  - ${REPO_TRITON}:latest"
    print_info "  - ${REPO_TRITON}:${VERSION_FULL}"
}

build_local() {
    local component=$1

    print_header "Building ${component} Image (local only, no push)"

    cd "${PROJECT_DIR}"

    case "${component}" in
        api)
            docker build \
                --file Dockerfile \
                --tag "${REPO_API}:latest" \
                --tag "${REPO_API}:${VERSION_FULL}" \
                ${CACHE_FLAG} \
                .
            print_success "API image built locally: ${REPO_API}:latest"
            ;;
        triton)
            docker build \
                --file Dockerfile.triton \
                --tag "${REPO_TRITON}:latest" \
                --tag "${REPO_TRITON}:${VERSION_FULL}" \
                ${CACHE_FLAG} \
                .
            print_success "Triton image built locally: ${REPO_TRITON}:latest"
            ;;
    esac
}

# =============================================================================
# Parallel Security Scans
# =============================================================================

run_parallel_scans() {
    local components=("$@")

    if [ "${SKIP_SECURITY_SCAN}" = "true" ]; then
        print_warning "Security scanning skipped (SKIP_SECURITY_SCAN=true)"
        return 0
    fi

    print_header "Running Security Scans in Parallel"

    # Update security tool databases first
    update_security_tools

    # Create temp files for status tracking
    local status_dir
    status_dir=$(mktemp -d)

    for component in "${components[@]}"; do
        (
            print_info "[${component^}] Starting security scan..."
            if run_security_scan "$component"; then
                echo "0" > "${status_dir}/${component}.status"
                print_success "[${component^}] Security scan completed"
            else
                echo "1" > "${status_dir}/${component}.status"
                print_warning "[${component^}] Security scan had issues"
            fi
        ) 2>&1 | sed "s/^/[${component^}] /" &
    done

    # Wait for all scans
    wait

    # Check results
    local all_passed=true
    for component in "${components[@]}"; do
        if [ -f "${status_dir}/${component}.status" ]; then
            status=$(cat "${status_dir}/${component}.status")
            if [ "$status" != "0" ]; then
                all_passed=false
            fi
        fi
    done

    # Cleanup
    rm -rf "${status_dir}"

    if [ "$all_passed" = true ]; then
        print_success "All security scans completed successfully!"
    else
        print_warning "Some security scans had issues (see above)"
    fi
}

# =============================================================================
# Scan Only Mode
# =============================================================================

scan_only() {
    print_info "Running security scan only (no build)..."

    # Build images locally first if they don't exist
    if ! docker image inspect "${REPO_API}:latest" >/dev/null 2>&1; then
        print_info "API image not found, building locally..."
        build_local "api"
    fi

    if ! docker image inspect "${REPO_TRITON}:latest" >/dev/null 2>&1; then
        print_info "Triton image not found, building locally..."
        build_local "triton"
    fi

    run_parallel_scans "api" "triton"
}

# =============================================================================
# Usage
# =============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTION]

Build and push Docker images to Docker Hub for OpenProcessor

Options:
    api         Build and push only API image (yolo-api/FastAPI)
    triton      Build and push only Triton image (inference server)
    all         Build and push both images (default)
    local       Build locally without pushing
    scan        Security scan only (build locally, scan, no push)
    help        Show this help message

Environment Variables:
    VERSION                   Semantic version (e.g., v1.2.3) - overrides VERSION file
    DOCKERHUB_USERNAME        Docker Hub username (default: davidamacey)
    PLATFORMS                 Target platforms (default: linux/amd64)
    NO_CACHE                  Build without cache (default: false)
    SKIP_SECURITY_SCAN        Skip security scanning (default: false)
    FAIL_ON_SECURITY_ISSUES   Fail build if security issues found (default: false)
    FAIL_ON_CRITICAL          Fail scan if CRITICAL vulnerabilities found (default: false)

Examples:
    $0                      # Build and push both images
    $0 api                  # Build and push only API image
    $0 local                # Build locally without pushing
    $0 scan                 # Security scan only

    # Specify version
    VERSION=v1.0.0 $0 all

    # Build without cache
    NO_CACHE=true $0 api

    # Skip security scanning for faster builds
    SKIP_SECURITY_SCAN=true $0 all

    # Fail build if security issues found (recommended for CI)
    FAIL_ON_SECURITY_ISSUES=true FAIL_ON_CRITICAL=true $0 all

Versioning:
    The script supports semantic versioning via VERSION file or environment variable:
    - Creates tags: latest, vX.Y.Z (full version only)
    - Environment variable VERSION overrides VERSION file
    - If neither exists, defaults to v0.0.0

Security Scanning:
    After building, images are automatically scanned with:
    - Hadolint: Dockerfile linting
    - Dockle: CIS best practices
    - Trivy: Vulnerability scanning
    - Grype: Additional vulnerability scanning
    - Syft: SBOM generation

    Reports are saved to ./security-reports/

EOF
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Version management
    if [ -n "${VERSION}" ]; then
        SEMVER="${VERSION}"
    elif [ -f "${PROJECT_DIR}/VERSION" ]; then
        SEMVER=$(cat "${PROJECT_DIR}/VERSION" | tr -d '[:space:]')
    else
        SEMVER="v0.0.0"
    fi

    # Validate and normalize version
    if [[ ! "${SEMVER}" =~ ^v?[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        print_error "Invalid semantic version format: ${SEMVER}"
        print_error "Expected format: v1.2.3 or 1.2.3"
        exit 1
    fi

    # Ensure version starts with 'v'
    if [[ ! "${SEMVER}" =~ ^v ]]; then
        SEMVER="v${SEMVER}"
    fi

    VERSION_FULL="${SEMVER}"

    print_header "OpenProcessor Docker Build & Push"
    print_info "Version: ${VERSION_FULL}"
    print_info "Commit:  ${COMMIT_SHA}"
    print_info "Branch:  ${BRANCH}"
    echo ""

    # Warn if using default version
    if [ "${SEMVER}" = "v0.0.0" ]; then
        print_warning "No VERSION file or VERSION env var found, using default: ${SEMVER}"
        print_warning "Create a VERSION file or set VERSION for production builds"
        echo ""
    fi

    # Cache control
    CACHE_FLAG=""
    if [ "${NO_CACHE}" = "true" ]; then
        CACHE_FLAG="--no-cache"
        print_info "Building without cache (NO_CACHE=true)"
    fi

    # Check prerequisites
    check_docker

    # Track built components
    BUILT_COMPONENTS=()

    case "${BUILD_TARGET}" in
        api)
            check_docker_login
            # Setup buildx
            if ! docker buildx inspect "${DEFAULT_BUILDER_NAME}" > /dev/null 2>&1; then
                print_info "Creating buildx builder..."
                docker buildx create --name "${DEFAULT_BUILDER_NAME}" --use
                docker buildx inspect --bootstrap
            else
                docker buildx use "${DEFAULT_BUILDER_NAME}"
            fi
            build_api
            BUILT_COMPONENTS+=("api")
            ;;
        triton)
            check_docker_login
            # Setup buildx
            if ! docker buildx inspect "${DEFAULT_BUILDER_NAME}" > /dev/null 2>&1; then
                print_info "Creating buildx builder..."
                docker buildx create --name "${DEFAULT_BUILDER_NAME}" --use
                docker buildx inspect --bootstrap
            else
                docker buildx use "${DEFAULT_BUILDER_NAME}"
            fi
            build_triton
            BUILT_COMPONENTS+=("triton")
            ;;
        all)
            check_docker_login
            # Setup buildx
            if ! docker buildx inspect "${DEFAULT_BUILDER_NAME}" > /dev/null 2>&1; then
                print_info "Creating buildx builder..."
                docker buildx create --name "${DEFAULT_BUILDER_NAME}" --use
                docker buildx inspect --bootstrap
            else
                docker buildx use "${DEFAULT_BUILDER_NAME}"
            fi
            build_api
            build_triton
            BUILT_COMPONENTS+=("api" "triton")
            ;;
        local)
            print_info "Building locally (no push)..."
            build_local "api"
            build_local "triton"
            BUILT_COMPONENTS+=("api" "triton")
            ;;
        scan)
            scan_only
            exit 0
            ;;
        help|--help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Invalid option: ${BUILD_TARGET}"
            show_usage
            exit 1
            ;;
    esac

    # Run security scans on built components
    if [ ${#BUILT_COMPONENTS[@]} -gt 0 ]; then
        run_parallel_scans "${BUILT_COMPONENTS[@]}"
    fi

    # Switch back to default builder
    docker buildx use default 2>/dev/null || true

    print_success "Build completed successfully!"
    echo ""
    print_info "Images:"
    if [[ " ${BUILT_COMPONENTS[*]} " =~ " api " ]]; then
        print_info "  API:    ${REPO_API}:latest / ${REPO_API}:${VERSION_FULL}"
    fi
    if [[ " ${BUILT_COMPONENTS[*]} " =~ " triton " ]]; then
        print_info "  Triton: ${REPO_TRITON}:latest / ${REPO_TRITON}:${VERSION_FULL}"
    fi
    echo ""
    print_info "To pull:"
    print_info "  docker pull ${REPO_API}:latest"
    print_info "  docker pull ${REPO_TRITON}:latest"
}

# Run main function
main
