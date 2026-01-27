# Installation Guide

Complete guide for setting up triton-api on your system.

---

## TL;DR - One Line Setup

```bash
git clone https://github.com/your-org/triton-api.git && cd triton-api && ./scripts/setup.sh
```

---

## Quick Start (Recommended)

### Interactive Setup

```bash
git clone https://github.com/your-org/triton-api.git
cd triton-api
./scripts/setup.sh
```

### Non-Interactive Setup

```bash
# Accept all defaults
./scripts/setup.sh --yes

# Specify profile and GPU
./scripts/setup.sh --profile=standard --gpu=0 --yes

# Skip TensorRT export (if models already exported)
./scripts/setup.sh --skip-export --yes
```

### What Setup Does

The setup script will:
1. Check prerequisites (Docker, NVIDIA drivers)
2. Detect your GPU and select the optimal profile
3. Download required models (~500MB)
4. Export models to TensorRT (45-60 minutes first time)
5. Generate configuration files
6. Start all services
7. Run smoke tests

<!--
## Docker Hub (Coming Soon)

Pre-built images with TensorRT models will be available:

```bash
# Pull pre-built images (no export required!)
docker pull your-org/triton-api:latest
docker pull your-org/triton-api-models:latest

# Clone for configs only
git clone https://github.com/your-org/triton-api.git
cd triton-api

# Start with pre-built images
docker compose -f docker-compose.hub.yml up -d
```

This skips the 45-60 minute TensorRT export step.
-->

---

## Prerequisites

### Required Software

| Software | Minimum Version | Installation |
|----------|-----------------|--------------|
| Docker | 20.10+ | [docs.docker.com](https://docs.docker.com/engine/install/) |
| Docker Compose | v2.0+ | Included with Docker Desktop |
| NVIDIA Driver | 535+ | [nvidia.com/drivers](https://www.nvidia.com/drivers) |
| NVIDIA Container Toolkit | Latest | [See below](#nvidia-container-toolkit) |

### Hardware Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU VRAM | 6GB | 12GB+ |
| GPU Architecture | Ampere (30-series) | Ampere or newer |
| System RAM | 16GB | 32GB+ |
| CPU Cores | 8 | 16+ |
| Storage | 20GB free | 50GB+ (SSD recommended) |

### NVIDIA Container Toolkit

Install the NVIDIA Container Toolkit to enable GPU access in Docker:

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify toolkit is configured
docker info | grep -i nvidia
nvidia-smi
```

---

## GPU Profiles

The system automatically selects a profile based on your GPU's VRAM:

| Profile | VRAM Range | Models | Performance | Use Case |
|---------|------------|--------|-------------|----------|
| **minimal** | 6-8GB | Core only | ~5 RPS | RTX 3060, RTX 4060 |
| **standard** | 12-24GB | All | ~15 RPS | RTX 3080, RTX 4090 |
| **full** | 48GB+ | All | ~50 RPS | A6000, A100 |

### Profile Details

**minimal** (6-8GB GPUs):
- YOLO object detection
- YOLO face detection
- ArcFace embeddings
- MobileCLIP embeddings
- No OCR (optional add-on)
- 1 instance per model, batch size 16

**standard** (12-24GB GPUs):
- All models including OCR
- 2 instances per model, batch size 32
- Good for most production workloads

**full** (48GB+ GPUs):
- All models with maximum parallelism
- 4 instances per model, batch size 64
- High-throughput production systems

### Manually Selecting a Profile

```bash
# During setup
./scripts/setup.sh --profile=minimal

# After setup
./scripts/triton-api.sh profile minimal
./scripts/triton-api.sh restart
```

---

## Manual Installation

If you prefer to run steps individually:

### 1. Clone Repository

```bash
git clone https://github.com/your-org/triton-api.git
cd triton-api
```

### 2. Create Directories

```bash
mkdir -p pytorch_models logs cache/huggingface outputs test_results
```

### 3. Download Models

Models are downloaded from public sources (no authentication required):

```bash
# All models (~500MB)
./scripts/triton-api.sh download all

# Or essential only (~250MB)
./scripts/triton-api.sh download essential

# Check download status
./scripts/triton-api.sh download status
```

### 4. Configure Environment

Copy and edit the environment template:

```bash
cp env.template .env
# Edit .env to adjust settings
```

Or generate automatically:

```bash
./scripts/triton-api.sh profile standard
```

### 5. Start Containers (for export)

```bash
docker compose up -d triton-api yolo-api
```

### 6. Export to TensorRT

This step converts models to optimized TensorRT format:

```bash
# All models (45-60 minutes)
./scripts/triton-api.sh export all

# Or essential only (25-35 minutes)
./scripts/triton-api.sh export essential

# Check export status
./scripts/triton-api.sh export status
```

### 7. Start All Services

```bash
docker compose up -d
```

### 8. Verify Installation

```bash
# Check status
./scripts/triton-api.sh status

# Run smoke tests
./scripts/triton-api.sh test quick

# Test API directly
curl http://localhost:4603/health
```

---

## Troubleshooting

### GPU Out of Memory (OOM)

**Symptoms:** Services crash, "CUDA out of memory" errors

**Solutions:**
1. Switch to a smaller profile:
   ```bash
   ./scripts/triton-api.sh profile minimal
   ./scripts/triton-api.sh restart
   ```

2. Reduce batch size in `.env`:
   ```bash
   MAX_BATCH_SIZE=8
   ```

3. Stop other GPU processes:
   ```bash
   nvidia-smi  # Check what's using GPU
   ```

### TensorRT Export Fails

**Symptoms:** Export hangs or errors during `trtexec`

**Solutions:**
1. Check GPU memory is available:
   ```bash
   nvidia-smi  # Need 4GB+ free
   ```

2. Unload existing models first:
   ```bash
   docker compose stop triton-api
   ./scripts/triton-api.sh export all
   docker compose start triton-api
   ```

3. Check CUDA version compatibility:
   ```bash
   nvidia-smi  # Driver version
   docker compose exec triton-api nvidia-smi  # Container CUDA version
   ```

### Models Not Loading

**Symptoms:** Triton shows "UNAVAILABLE" for models

**Solutions:**
1. Check if exports exist:
   ```bash
   ./scripts/triton-api.sh export status
   ```

2. Check Triton logs:
   ```bash
   ./scripts/triton-api.sh logs triton-api
   ```

3. Verify model configs:
   ```bash
   ls -la models/*/config.pbtxt
   ```

### Docker Permission Errors

**Symptoms:** "Permission denied" when running Docker commands

**Solutions:**
1. Add user to docker group:
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

2. Check Docker socket permissions:
   ```bash
   ls -la /var/run/docker.sock
   ```

### Services Won't Start

**Symptoms:** `docker compose up` fails

**Solutions:**
1. Check port conflicts:
   ```bash
   lsof -i :4603  # API port
   lsof -i :4600  # Triton port
   ```

2. Check disk space:
   ```bash
   df -h
   ```

3. View detailed errors:
   ```bash
   docker compose up  # Without -d to see output
   ```

---

## Upgrading

### Pulling Latest Version

```bash
git pull
./scripts/triton-api.sh update
./scripts/triton-api.sh restart
```

### Rebuilding After Updates

```bash
docker compose build --no-cache
docker compose up -d
```

### Re-exporting Models

After major updates, you may need to re-export TensorRT models:

```bash
./scripts/triton-api.sh export all
./scripts/triton-api.sh restart
```

---

## Uninstallation

### Stop and Remove Containers

```bash
docker compose down -v  # -v removes volumes
```

### Remove Images

```bash
docker compose down --rmi all
```

### Clean Up Files

```bash
# Remove generated files
rm -rf pytorch_models/*.pt pytorch_models/*.onnx
rm -rf models/*/1/model.plan
rm -rf .env docker-compose.override.yml
rm -rf cache/ logs/ outputs/ test_results/
```

---

## Docker Image Building

### Production Dockerfiles

The project includes production-optimized Dockerfiles:

| File | Purpose | Base Image |
|------|---------|------------|
| `Dockerfile` | FastAPI service | `python:3.13-slim-trixie` |
| `Dockerfile.triton` | Triton server | `nvcr.io/nvidia/tritonserver:25.10-py3` |

### Building Images

```bash
# Build and push to Docker Hub
./scripts/docker-build-push.sh all

# Build locally without pushing
./scripts/docker-build-push.sh local

# Build with specific version
VERSION=v1.0.0 ./scripts/docker-build-push.sh all
```

### Security Scanning

The project includes comprehensive security scanning using free, open-source tools:

**Tools Used:**
- **Hadolint** - Dockerfile linting
- **Dockle** - CIS Docker Benchmark compliance
- **Trivy** - Vulnerability scanning
- **Grype** - Fast vulnerability scanning
- **Syft** - SBOM (Software Bill of Materials) generation

```bash
# Install security scanning tools
./scripts/security-scan.sh install

# Scan all images
./scripts/security-scan.sh all

# Scan specific image
./scripts/security-scan.sh api

# View reports
ls -la security-reports/
```

**Reports Generated:**
- `*-hadolint.txt` - Dockerfile linting results
- `*-dockle.json` - CIS best practices check
- `*-sbom.json` - Software Bill of Materials (CycloneDX)
- `*-trivy.json/txt` - Vulnerability scan results
- `*-grype.json/txt` - Additional vulnerability scan

### CI/CD Integration

For automated builds in CI/CD pipelines:

```bash
# Fail on security issues
FAIL_ON_SECURITY_ISSUES=true FAIL_ON_CRITICAL=true ./scripts/docker-build-push.sh all

# Skip scanning for faster builds
SKIP_SECURITY_SCAN=true ./scripts/docker-build-push.sh all
```

---

## Getting Help

- **Documentation:** See [README.md](README.md) and [CLAUDE.md](CLAUDE.md)
- **Issues:** Report bugs at [GitHub Issues](https://github.com/your-org/triton-api/issues)
- **Logs:** Check `./scripts/triton-api.sh logs` for debugging
