# Installation Guide

Two ways to install OpenProcessor:

- **The one-line installer** (`setup-openprocessor.sh`): a pinned release into
  its own directory, no git clone, no image build, no host Python. Use this to
  run OpenProcessor.
- **From source** (`git clone` + `scripts/setup.sh`): for development or to
  build the images yourself. See [Install from source](#install-from-source).

---

## One-line installer

```bash
curl -fsSL https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/setup-openprocessor.sh | bash
```

What it does, in order:

1. Resolves the release (latest, or `--version vX.Y.Z`), downloads that
   release's `setup-openprocessor.sh` and `SHA256SUMS`, checks the checksum and
   runs the verified copy. A truncated download is a syntax error before
   anything runs.
2. Downloads the release bundle and checks every file against `SHA256SUMS`.
3. Checks Docker, Compose, the GPUs and free disk; asks which tiers you want
   (or takes `--tiers`), and plans GPU placement.
4. Writes `.env` (mode 600): tiers as `COMPOSE_PROFILES`, ports, image digests
   from `images.lock`, the GPU plan and the OpenSearch heap. It never changes
   a value you set yourself.
5. Pulls the images and checks each one's digest against `images.lock`.
6. Exports the TensorRT engines inside the containers, starts everything and
   runs a health check.
7. Prints a summary: URLs, the OpenSearch heap and shard budget, the security
   note and the management commands.

State lives in `<dir>/.install/` (`state.json`, `install.log`, mode 600).

### Prerequisites

- Linux with Docker Engine and Docker Compose v2.
- An NVIDIA GPU with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  The installer never installs packages or changes the Docker daemon config;
  it tells you what is missing and stops.
- Disk for the tiers you pick, on both the install dir and Docker's data root
  (tier sizes: [README](README.md#tiers)), plus 20 GB headroom.
- For the `segmenter` tier: a HuggingFace token with access to SAM 3 (gated).
  Pass it with `HF_TOKEN_FILE=<path>` (or `HF_TOKEN`), or paste it at the prompt.

### Verifying the download

See [README: Verify the download yourself](README.md#verify-the-download-yourself).
`SHA256SUMS`, `images.lock` and `cropwright.lock` are **integrity** checks: they
catch a truncated or corrupted download and a release whose files disagree
with each other. They come from the same origin as the files they cover, so
they do not prove **authenticity**. Signing is follow-up work.

### Installer flags

Every flag also has an environment variable (the flag wins).

| Flag | Env var | Meaning |
|---|---|---|
| `--dir PATH` | `OP_INSTALL_DIR` | install directory (default `./openprocessor`) |
| `--project NAME` | `OP_PROJECT` | compose project name and container prefix (`[a-z0-9][a-z0-9_-]*`, default `openprocessor`) |
| `--version vX.Y.Z` | `OP_VERSION` | pinned release (default: latest published release) |
| `--branch REF` | `OP_BRANCH` | testing install from a branch head: not checksum-verified, not reproducible |
| `--release-dir DIR` | `OP_RELEASE_DIR` | install from release assets built locally by `scripts/release/build_deploy_bundle.sh` (still checksum-verified; needs `--version`) |
| `--image-tag TAG` | `OP_IMAGE_TAG` | run images by tag (prefix `OP_IMAGE_REPO`) instead of `images.lock` digests; local builds only, never pulled |
| `--tiers LIST` / `--all` | `OP_TIERS` | comma list of `core,curation,segmenter,vlm,trainer,cropwright`, or all of them |
| `--gpu-plan K=V,...` | `OP_GPU_PLAN` | override GPU placement, e.g. `triton=1,segmenter=0,vlm=2,trainer=0` |
| `--profile NAME` | `GPU_PROFILE` | Triton instance profile: `minimal`, `standard` or `full` |
| `--vlm-remote URL --vlm-model NAME` | `OP_VLM_URL`, `OP_VLM_MODEL` | use a remote OpenAI-compatible VLM instead of the local one |
| `--vlm-key-file PATH` | `OP_VLM_KEY_FILE` | API key for the remote VLM (stored in `secrets/vlm/`, never in `.env`) |
| `--vlm-model-id ID` | `OP_VLM_CATALOG_ID` | choose a VLM catalog entry explicitly |
| `--bind ADDR` | `OP_BIND_ADDRESS` | publish address for the API and backend ports (default `127.0.0.1`); see [Network access](#network-access) |
| `--local-only` | | keep Cropwright on `127.0.0.1` too |
| `--port-base N` | `OP_PORT_BASE` | move the whole 46xx port block to `N..N+12` |
| `--with-monitoring` | `OP_WITH_MONITORING` | add Prometheus, Grafana, Loki and Alloy (default-open dashboards) |
| `--sample-data` | `OP_SAMPLE_DATA` | fetch the public COCO sample after the install |
| `--skip-models` | | do not export or load models (run `./openprocessor models install` later) |
| `--no-start` | | configure and pull only; start nothing |
| `--unattended` | `OP_UNATTENDED` | never prompt (automatic when there is no terminal) |
| `--dry-run` | `OP_DRY_RUN` | print every state-changing command and run none |
| `--force` | | accept a GPU plan the hardware check refused |
| `--force-existing-dir` | | install into a non-empty directory this installer did not create (its files are backed up first) |
| `--yes` | | confirm a destructive step (uninstall, rollback, upgrade) without a prompt; `--unattended` implies it. A missing terminal alone is never consent |
| `--cpu [--control-plane-only]` | `OP_FORCE_CPU` | no GPU. Alone it explains why and exits 4; with `--control-plane-only` it installs OpenSearch, the API and Cropwright with no inference |
| `--repair` | | re-verify files, re-pull missing images, re-run failed model groups, at the installed version |
| `--rollback` | | restore the newest backup of a different version (see below) |
| `--uninstall` | | stop and remove the containers; add `--purge-volumes`, `--purge-data`, `--remove-images` to delete more |
| `--reset-hf-token` | | ask for a new HuggingFace token |

Consent variables for unattended runs:

| Variable | Allows |
|---|---|
| `OP_ALLOW_PUBLIC_BIND=1` | a non-loopback `--bind` |
| `OP_ALLOW_EXTERNAL_VLM=1` | a `--vlm-remote` URL outside private address space (crops leave the host) |
| `OP_CONFIRM_PURGE=<project>` | `--purge-volumes` / `--purge-data` |
| `OP_CONFIRM_PURGE_SECRETS=<project>` | also deleting `secrets/` during a purge |

Other variables: `OP_GH_REPO`, `CW_GH_REPO` (where to install from),
`OP_IMAGE_NAMESPACE` (Docker Hub namespace, default `davidamacey`),
`OP_ARTIFACT_BASE_URL` / `OP_RAW_BASE_URL` and the `CW_` equivalents (https
mirrors), `OP_DOCS_URL`, `OP_HEALTH_TIMEOUT` (cap in seconds on each health
wait). The header of `setup-openprocessor.sh` lists them all.

### Network access

**Cropwright is reachable on your LAN by default, for homelab or
small-business use. The API itself stays bound to 127.0.0.1. There is no login
on Cropwright — a warning is shown. Pass `--local-only` to opt out and keep
everything on 127.0.0.1.**

- Cropwright's nginx reaches the API over the Docker network, so LAN browsers
  never need the API port.
- Do not port-forward any of these ports to the public internet. For access
  beyond a trusted network, put a reverse proxy with authentication in front.
- `--bind <ip>` publishes the API and backend ports on that address. A
  non-loopback address asks you to type `expose` (unattended:
  `OP_ALLOW_PUBLIC_BIND=1`).
- A specific `--bind` address (for example `--bind 10.0.0.5`) also narrows
  Cropwright to that interface. `--bind 0.0.0.0` leaves Cropwright on all
  interfaces, and `--local-only` always wins.
- A re-run keeps the Cropwright bind you chose last time.

Details: [SECURITY.md](SECURITY.md).

### OpenSearch heap sizing

The installer sets `OPENSEARCH_HEAP` in `.env` from the host's RAM:
`clamp(floor(RAM_GiB / 8), 1, 8)` GB. `docker-compose.yml` passes it as
`-Xms`/`-Xmx`.

| Host RAM | Heap | Soft shard budget (20 per heap GB) |
|---|---|---|
| under 16 GiB | 1g | 20 |
| 16-23 GiB | 2g | 40 |
| 24-31 GiB | 3g | 60 |
| 32-39 GiB | 4g | 80 |
| 40-63 GiB | 5g-7g | 100-140 |
| 64 GiB and up | 8g (cap) | 160 |

Why 1/8 with a cap of 8 GB: OpenSearch wants at most half of its memory as
heap (the rest is page cache) and stays well below the ~31 GB
compressed-pointer limit, and this host also runs Triton, the API and the
model workers. A heap you set yourself in `.env` is never overwritten, on a
re-run or by `scripts/setup.sh --force`.

The **soft shard budget** is heap GB x `OP_SHARDS_PER_HEAP_GB` (advanced,
default 20, commented out in `env.template`). It is not a cap: going past it
only warns. The installer summary prints both, for example
`OpenSearch  : heap 4g, soft shard budget 80 (20 shards per heap GB)`. To make
room for more, raise `OPENSEARCH_HEAP` (about 1 GB per 20 shards) and
`./openprocessor restart opensearch`.

### Curation quick-config

`env.template` has a "Curation quick-config" block with every key the
curation tiers need: the ingest detector (`OP_INGEST_PRIMARY_*`), the
segmenter and VLM endpoints, the feature flags and the GPU keys. Projects,
prompt packs, region profiles, keymaps and the active VLM are not `.env`
settings: they are created and activated through the API. The installer writes exactly that block; the detector keeps
its full class vocabulary (leave the class-id filter unset). `./openprocessor sample
coco` fetches the public, license-filtered COCO sample (200 images; `--full`
for 800) into `data/samples/`. Details: [docs/CURATION.md](docs/CURATION.md).

### The `openprocessor` CLI

Every install directory has an `openprocessor` script; a source checkout has
the same script at the repo root, with `scripts/openprocessor.sh` as a shim.
It runs every compose command with the right project name, `.env` and
project directory, and refuses to act on a compose project that another
directory created. Run `./openprocessor help` for the list.

| Command | What it does |
|---|---|
| `start` / `stop` | `docker compose up -d` / `down` for this install |
| `restart [service]` | restart one service or all |
| `logs [service] [-f]` | last 100 lines of a service (50 of all), or follow with `-f` |
| `status` | containers, then API, Triton, OpenSearch and Grafana health on this install's ports, then GPU memory |
| `health` | print the API `/health` JSON |
| `models [status]` | Triton model repository index and each model's state |
| `models install [--only GROUP]` / `models repair` | export and load model groups (`preflight`, `base`, `yolo`, `mobileclip`, `faces`, `ocr`, `pe`); skips groups that are up to date and re-runs failed ones |
| `export <model>` / `download <model>` | export to TensorRT / download weights. Targets include `all`, `essential`, `status`; `export` also takes `yolo`, `scrfd`, `arcface`, `clip-image`, `clip-text`, `ocr`; `download` also takes `ocr` |
| `curation [up\|down\|logs\|status]` | start, stop, follow or list the curation workers (`status` is the default) |
| `sample coco [--full]` | fetch the public COCO sample into `data/samples/` using the API image |
| `config show` | print `.env` with every token, key, secret and password redacted |
| `gpu plan` | print the GPU keys from `.env` |
| `vlm ...` | local VLM catalog and switching, see [Choosing a VLM](#choosing-a-vlm) |
| `train-mode on\|off` | stop (and later restart) the local VLM, plus the segmenter when it shares the training GPU, around a training run |
| `upgrade [--version vX.Y.Z]`, `repair`, `uninstall [...]` | run this install's own `setup-openprocessor.sh` against this directory (installed directories only) |
| `update` | in a checkout: pull images and rebuild; in an install directory: same as `upgrade` |
| `profile [minimal\|standard\|full] [gpu_id]` | show or switch the Triton instance profile (checkout only) |
| `test [quick\|full\|visual]` | smoke test, `tests/test_full_system.py`, or visual validation (checkout only; `full` and `visual` need `.venv`) |
| `bench [quick\|full]` | build and run `benchmarks/triton_bench` (checkout only) |
| `clean [cache\|models\|exports\|all]` | remove caches, downloaded weights, or TensorRT plans (asks first) |
| `setup [flags]` | run `scripts/setup.sh` (checkout only) |
| `version` | print the CLI version |

### Choosing a VLM

Labeling and region verification call an OpenAI-compatible vision-language
model. A deployment has a **registry** of endpoints (the built-in `env`
endpoint from `OP_VLM_URL`, plus any you save through the API) and each project
**activates** one. Three ways to get a model:

1. **Local model from the catalog** (the `vlm` tier). The catalog is
   `examples/vlm/catalog.tsv`. The installer picks the best `tested` entry
   that fits the free VRAM on the VLM GPU, or you name one with
   `--vlm-model-id ID`.
2. **Remote endpoint** at install time: `--vlm-remote URL --vlm-model NAME
   [--vlm-key-file PATH]`. The key goes to `secrets/vlm/`, never `.env`. A URL
   outside private address space needs consent (`OP_ALLOW_EXTERNAL_VLM=1`
   unattended) because crops leave the host.
3. **Registered endpoint** through `POST /curation/vlm/endpoints`, activated per
   project with `POST /curation/projects/{project}/vlm/endpoints/{name}/activate`.

Host-side commands for the local model:

| Command | What it does |
|---|---|
| `./openprocessor vlm list` | catalog entries with licence, VRAM and status |
| `./openprocessor vlm status` | the model in `.env`, the endpoint, the model actually serving (probed) and the one requested through the API |
| `./openprocessor vlm use <id> [--force] [--yes]` | switch the local model: checks the GPU has room (`--force` to try anyway), checks a gated model's HuggingFace token, refuses while a training run holds the GPUs, pauses the workers, rewrites the VLM keys in `.env` (backup restored on failure), recreates the container, waits until it serves the model, probes it, then unpauses |
| `./openprocessor vlm apply [--force] [--yes]` | apply the model requested through `POST /curation/vlm/local/select` (does the same as `use`) |
| `./openprocessor vlm probe` | probe the in-compose endpoint and record what it serves |
| `./openprocessor vlm key set <slug>` | read a key from the terminal (hidden) into `secrets/vlm/<slug>`, mode 600; reference it from an endpoint as `secret:<slug>` |

Entries whose status is not `tested` are never picked automatically; `use`
accepts them with a warning. The API never restarts the container: it records
the wanted model and reports `restart_required`.

### Upgrade, repair, rollback, uninstall

Run these from the install directory.

```bash
./openprocessor upgrade                   # latest release
./openprocessor upgrade --version vX.Y.Z  # a specific release
./setup-openprocessor.sh --repair
./setup-openprocessor.sh --rollback
./setup-openprocessor.sh --uninstall [--purge-volumes] [--purge-data] [--remove-images]
```

- **Upgrade** (or re-running the installer in the same dir): backs up the
  current release files, `.env` and state to `backups/<UTC timestamp>/`,
  installs the new release, adds new `.env` keys without touching yours,
  updates the image pins, re-runs the model groups that need it, restarts and
  checks health. Changing version asks once (`--yes` to skip the prompt). A
  re-run with no `--tiers` keeps the installed tiers.
- **Repair**: the same at the installed version. Re-verifies files,
  re-pulls missing images, re-runs failed or missing model groups.
- **Rollback**: restores the **newest backup of a different version** than
  the one installed, and re-pins its images. Same-version backups (from a
  re-run or a repair) are skipped, so after upgrading v1 to v2 and re-running
  v2, rollback lands on v1. With no other version on record it restores the
  newest backup (a config rollback). The files it replaces are kept in
  `.install/rollback-undo/`. If the Triton image changed, rebuild engines
  with `./openprocessor models install`.
- **Uninstall**: stops and removes the containers of this install only.
  Volumes, `data/`, models and caches are kept unless you add
  `--purge-volumes` (the named volumes), `--purge-data` (`models/`,
  `pytorch_models/`, `cache/`, `data/`; a source root outside the install dir
  is never touched) or `--remove-images` (only this install's pinned images
  that nothing else uses). Every purge makes you type the project name
  (unattended: `OP_CONFIRM_PURGE=<project>`).

All of these act only on a directory this installer created, and only on its
own compose project; a git checkout or another install's dir is refused.

### Offline installs from a local bundle

`scripts/release/build_deploy_bundle.sh vX.Y.Z [SRC] [OUT]` builds the release
assets locally; `--release-dir OUT --version vX.Y.Z` installs from them (still
checksum-verified). For the `cropwright` tier to be offline too, stage
Cropwright's release files into the bundle:

```bash
CW_RELEASE_DIR=/path/to/cropwright-release \
  scripts/release/build_deploy_bundle.sh vX.Y.Z . dist/release-vX.Y.Z
```

`CW_RELEASE_DIR` holds Cropwright's `SHA256SUMS`, `docker-compose.yml` and
`.env.example` for the tag in `cropwright.lock`; the script checks them
against the lock and copies them to `OUT/cropwright/<tag>/`. Without them the
installer says so and downloads Cropwright from its GitHub release (verified
against `cropwright.lock` either way). Images still come from a registry
unless they are already present locally (use `--image-tag` for local builds).

### Troubleshooting

Exit codes:

| Code | Meaning | What to do |
|---|---|---|
| 1 | general failure | read the last `[ERROR]` line and `.install/install.log` |
| 2 | usage | check the flags (`--help`); a port conflict you declined also exits 2: use `--port-base N` |
| 3 | project or container-name collision | another compose project already uses the name, or the dir is a checkout / another install: pass `--project NAME` or `--dir` |
| 4 | no usable GPU, or the GPU plan was refused | fix the driver / Container Toolkit, adjust `--gpu-plan`, or `--force` |
| 5 | Docker unreachable | start Docker; add your user to the `docker` group |
| 6 | HuggingFace token missing or rejected | for `segmenter`, accept the SAM 3 license on HuggingFace, then pass the token (`HF_TOKEN_FILE`) or `--reset-hf-token` |
| 7 | verification failed | a checksum, digest or `images.lock` check failed (including a lock line whose image repo does not match its key). Re-download; if it persists the release itself is inconsistent: report it |
| 8 | health check or model setup failed | `./openprocessor logs <service>`, fix, then `./setup-openprocessor.sh --repair` |
| 9 | consent not given | re-run with `--yes`, or set the matching consent variable |

Common cases:

- **"a container cannot see GPU N"**: the NVIDIA Container Toolkit is missing or
  not configured for Docker. On WSL2 this usually means GPU passthrough is off.
  The installer never falls back to CPU silently.
- **Port in use**: interactively the installer offers the next free port;
  unattended it takes it and warns. `--port-base N` moves the whole block.
- **No GPU**: there is no CPU inference path (every model except the text
  encoder is a TensorRT engine). `--cpu --control-plane-only` gives you
  OpenSearch, the API and Cropwright for browsing an existing index or using a
  remote VLM; the API reports `degraded`.
- **Failed model groups**: the summary lists them with
  `./openprocessor models install --only <group>` to retry each one.
- **Disk**: the installer warns if it cannot check free space; the Triton
  image alone is about 30 GB.

---

## Install from source

For development, or to build the images yourself.

```bash
git clone https://github.com/davidamacey/OpenProcessor.git && cd OpenProcessor && ./scripts/setup.sh
```

### Interactive Setup

```bash
git clone https://github.com/davidamacey/OpenProcessor.git
cd OpenProcessor
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
3. Pull the API and Triton images from Docker Hub (tens of GB); if the pull
   fails it builds them from the Dockerfiles instead
4. Download required models (~500MB)
5. Export models to TensorRT (~30-60 minutes, **one-time only**)
6. Generate configuration files (the OpenSearch heap is sized from host RAM,
   as for the installer; see [OpenSearch heap sizing](#opensearch-heap-sizing))
7. Start all services
8. Run smoke tests

**First-time setup takes ~30-60 minutes** (mostly TensorRT compilation).
Subsequent starts take ~30 seconds since compiled engines are cached on disk.

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
- SCRFD face detection
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
./scripts/openprocessor.sh profile minimal
./scripts/openprocessor.sh restart
```

---

## Manual Installation

If you prefer to run steps individually:

### 1. Clone Repository

```bash
git clone https://github.com/davidamacey/OpenProcessor.git
cd OpenProcessor
```

### 2. Create Directories

```bash
mkdir -p pytorch_models logs cache/huggingface outputs test_results
```

### 3. Download Models

Models are downloaded from public sources (no authentication required):

```bash
# All models (~500MB)
./scripts/openprocessor.sh download all

# Or essential only (~250MB)
./scripts/openprocessor.sh download essential

# Check download status
./scripts/openprocessor.sh download status
```

### 4. Configure Environment

Copy and edit the environment template:

```bash
cp env.template .env
# Edit .env to adjust settings
```

Or generate automatically:

```bash
./scripts/openprocessor.sh profile standard
```

### 5. Start Containers (for export)

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d triton-server yolo-api
```

`make up` and `./openprocessor start` add the dev overlay for you and start
the whole core stack.

### 6. Export to TensorRT

This step converts models to optimized TensorRT format:

```bash
# All models (45-60 minutes)
./scripts/openprocessor.sh export all

# Or essential only (25-35 minutes)
./scripts/openprocessor.sh export essential

# Check export status
./scripts/openprocessor.sh export status
```

### 7. Start All Services

```bash
make up              # or: ./openprocessor start
make curation-up     # optional: the curation workers
```

### 8. Verify Installation

```bash
# Check status
./scripts/openprocessor.sh status

# Run smoke tests
./scripts/openprocessor.sh test quick

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
   ./scripts/openprocessor.sh profile minimal
   ./scripts/openprocessor.sh restart
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

2. The export needs the `triton-server` container running (it builds the
   engines with `trtexec` there) and frees GPU memory by unloading models
   first. Stop other GPU processes, then retry:
   ```bash
   ./scripts/openprocessor.sh start
   ./scripts/openprocessor.sh export all
   ```

3. Check CUDA version compatibility:
   ```bash
   nvidia-smi  # Driver version
   docker compose exec triton-server nvidia-smi  # Container CUDA version
   ```

### Models Not Loading

**Symptoms:** Triton shows "UNAVAILABLE" for models

**Solutions:**
1. Check if exports exist:
   ```bash
   ./scripts/openprocessor.sh export status
   ```

2. Check Triton logs:
   ```bash
   ./scripts/openprocessor.sh logs triton-server
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
./scripts/openprocessor.sh update
./scripts/openprocessor.sh restart
```

### Rebuilding After Updates

```bash
make rebuild
make up
```

### Re-exporting Models

After major updates, you may need to re-export TensorRT models:

```bash
./scripts/openprocessor.sh export all
./scripts/openprocessor.sh restart
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
| `Dockerfile.triton` | Triton server | `nvcr.io/nvidia/tritonserver:26.06-py3` |

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
- **Issues:** Report bugs at [GitHub Issues](https://github.com/davidamacey/OpenProcessor/issues)
- **Logs:** Check `./scripts/openprocessor.sh logs` for debugging
