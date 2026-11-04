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
- An NVIDIA GPU and driver, with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  The installer never installs packages or changes the Docker daemon config;
  it tells you what is missing and stops.
- Disk for the tiers you pick, on both the install dir and Docker's data root
  (tier sizes: [README](README.md#tiers)), plus 20 GB headroom.
- For the `segmenter` tier: a HuggingFace token with access to SAM 3 (gated).
  Pass it with `HF_TOKEN_FILE=<path>` (or `HF_TOKEN`), or paste it at the prompt.
  An unattended install (`--unattended`, or no terminal) without a token does not
  fail: it drops the segmenter from the recommended tiers and logs a warning, and
  an explicit `--tiers ...,segmenter` then stops at the missing token. For
  scripted installs always set `HF_TOKEN_FILE` (a file with mode 600) and name the tier.

Hardware (the same table as [Deployment and GPU sizing](docs-site/docs/deployment/gpu-sizing.mdx#hardware),
which also has the profile table):

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, 8 GB+ VRAM, Ampere or newer | A100 / A6000 / RTX 4090 class, 16 GB+ |
| RAM | 16 GB | 64 GB |
| CPU | 16 cores | 48+ cores |
| Storage | | NVMe SSD for image storage |

<details>
<summary>Installing the NVIDIA Container Toolkit (Ubuntu/Debian)</summary>

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify the toolkit is configured
docker info | grep -i nvidia
nvidia-smi
```

</details>

### Verifying the download

See [README: Verify the download yourself](README.md#verify-the-download-yourself).
`SHA256SUMS` and `images.lock` are **integrity** checks: they
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
| `--branch REF` | `OP_BRANCH` | testing install from a branch head: not checksum-verified, not reproducible. Images run by the tag `sha-<first 12 hex of the commit>`; CI does not publish those tags, so tag a local build with it or pass `--image-tag` |
| `--release-dir DIR` | `OP_RELEASE_DIR` | install from release assets built locally by `scripts/release/build_deploy_bundle.sh` (still checksum-verified; needs `--version`) |
| `--image-tag TAG` | `OP_IMAGE_TAG` | run images by tag (prefix `OP_IMAGE_REPO`) instead of `images.lock` digests; local builds only, never pulled |
| `--tiers LIST` / `--all` | `OP_TIERS` | comma list of `core,curation,segmenter,vlm,trainer,cropwright`, or all of them |
| `--gpu-plan K=V,...` | `OP_GPU_PLAN` | override GPU placement, e.g. `triton=1,segmenter=0,vlm=2,trainer=0` |
| `--profile NAME` | `GPU_PROFILE` | Triton instance profile: `minimal`, `small`, `standard` or `full` (picked automatically from the free VRAM of the Triton card) |
| `--vlm-remote URL --vlm-model NAME` | `OP_VLM_URL`, `OP_VLM_MODEL` | use a remote OpenAI-compatible VLM instead of the local one |
| `--vlm-key-file PATH` | `OP_VLM_KEY_FILE` | API key for the remote VLM (stored in `secrets/vlm/`, never in `.env`) |
| `--vlm-model-id ID` | `OP_VLM_CATALOG_ID` | choose a VLM catalog entry explicitly |
| `--bind ADDR` | `OP_BIND_ADDRESS` | publish address for the API and backend ports (default `127.0.0.1`); see [Network access](#network-access) |
| `--local-only` | | keep Cropwright on `127.0.0.1` too |
| `--port-base N` | `OP_PORT_BASE` | move the whole 46xx port block to `N..N+12` |
| `--with-monitoring` | `OP_WITH_MONITORING` | add Prometheus, Grafana, Loki and Alloy (default-open dashboards) |
| `--sample-data` | `OP_SAMPLE_DATA` | fetch the public COCO sample after the install |
| `--no-faces` | `OP_FACES_ENABLED=false` | turn the face models off: no face detection or embedding, the `faces` model group is skipped (written to `.env`) |
| `--skip-models` | | do not export or load models (run `./openprocessor models install` later); the API answers 503 until then, the install still exits 0 and says so. OpenSearch is brought up healthy first, then the rest starts without `depends_on` gating on the API; the per-model READY checks are skipped, a dead API or OpenSearch still fails the install |
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

Other variables: `OP_GH_REPO` (the GitHub `org/repo` to install from),
`OP_IMAGE_NAMESPACE` (Docker Hub namespace, default `davidamacey`),
`OP_ARTIFACT_BASE_URL` / `OP_RAW_BASE_URL` (https mirrors), `OP_DOCS_URL`,
`OP_HEALTH_TIMEOUT` (cap in seconds on each health wait). The header of
`setup-openprocessor.sh` lists them all.

### Network access

Cropwright is reachable on your LAN by default, for homelab or
small-business use. The API itself stays bound to 127.0.0.1. On a LAN bind the
installer puts a shared HTTP basic-auth login in front of Cropwright (user `cropwright`, password
in `secrets/cropwright/password`). Pass `--local-only` to opt out and keep
everything on 127.0.0.1.
`--bind <ip>` publishes the backend ports after a typed `expose` (unattended: `OP_ALLOW_PUBLIC_BIND=1`).
The login, the bind rules, `OP_UI_BIND_ADDRESS` and `OP_GATEWAY_SUBPATHS` are in
[Security and network exposure](docs-site/docs/deployment/security.mdx#lan-access-from-the-one-line-installer);
the policy is [SECURITY.md](SECURITY.md). Do not port-forward any of these ports to the public internet; use a reverse proxy with authentication for remote access.

### Control-plane-only mode (optional, not the default)

`--cpu --control-plane-only` is an installer option for a machine with no
usable NVIDIA GPU. It starts only OpenSearch and the API, plus Cropwright if you
ask for that tier. It does not start Triton, the curation workers, the VLM, the
segmenter, the trainer or monitoring, and it refuses `--tiers` that include
them. Use it to browse or manage an existing index, or when the models run on
another machine. Anything that needs local inference (ingest detection, embeddings,
exports, training) answers 503, and the API reports `degraded`. The default
install requires a GPU and builds the full stack for the tiers you pick.

### OpenSearch heap sizing

The installer sets `OPENSEARCH_HEAP` in `.env` to half of host RAM (2 to 30 GB) and never overwrites a value
you set. The table, the shard budget and the kernel setting are in
[Memory, disk and storage](docs-site/docs/deployment/sizing-and-storage.mdx#opensearch-heap-and-shards).

### Curation quick-config

`env.template` has a "Curation quick-config" block with every key the
curation tiers need: the ingest detector (`OP_INGEST_PRIMARY_*`), the
segmenter and VLM endpoints, the feature flags and the GPU keys. Projects,
prompt packs, region profiles, keymaps and the active VLM are not `.env`
settings: they are created and activated through the API. The installer writes exactly that block; the detector keeps
its full class vocabulary (leave the class-id filter unset). `./openprocessor sample
coco` fetches the public, license-filtered COCO sample (200 images; `--full`
for 800) into `data/samples/`. Details: [docs/CURATION.md](docs/CURATION.md).

#### Where your images live (the source root)

The API and the workers read images from one source root, mounted read-only at `/data/source`; its host side is
`OP_SOURCE_ROOT_HOST` (the installer writes `./data`). Which services mount it, how to change it and the
aliases are in [Basic configuration](docs-site/docs/configuration/basic.mdx). Second roots need your own
`docker-compose.local.yml` (next).

### More than one source root and your own compose overrides

`./openprocessor` and `make` load `docker-compose.local.yml` (next to `docker-compose.yml`,
ignored by git) after the stock files, when it exists. Put anything of your own in it: a second
read-only source root, a changed `shm_size`, a mounted credentials file. A file named
`docker-compose.override.yml` is not read. Example for a second root:

```yaml
# docker-compose.local.yml (next to docker-compose.yml; git ignores it)
services:
  api:
    volumes:
      - /srv/nas/archive:/data/archive:ro
  curation-detection-worker:
    volumes:
      - /srv/nas/archive:/data/archive:ro
  curation-auto-label-worker:
    volumes:
      - /srv/nas/archive:/data/archive:ro
```

and in `.env`:

```bash
OP_SOURCE_PATH_ALIASES=archive=/data/archive
```

Then `./openprocessor start` (or `make up`) recreates the services. Check the merged result with
`docker compose -p <project> -f docker-compose.yml -f docker-compose.local.yml config`.

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
| `models install [--only GROUP]` / `models repair` | export and load model groups (`preflight`, `base`, `yolo`, `mobileclip`, `faces`, `ocr`, `pe`); skips the `faces` group when `OP_FACES_ENABLED=false` and groups that are up to date (every engine stamp in `.install/precision.tsv` must match the GPU architecture and the exported ONNX, when the ONNX is present) and re-runs failed ones; engines already in `models/` are loaded into Triton and adopted if it accepts them, and only a missing, stale or refused engine is rebuilt (a GPU profile change alone does not rebuild); an unreachable Triton stops the run instead of rebuilding. An engine adopted without its ONNX is stamped with hash `-`; the first run that finds the ONNX and the model READY pins the real hash (the engine is kept, later ONNX changes rebuild it). Accepts `--yes` (never prompts; kept for scripted callers that pass it to every subcommand) |
| `faces purge [--dry-run] [--yes]` | report or delete every stored face record, person id and name, and the face FAISS files; `--yes` requires `OP_FACES_ENABLED=false` first (see [SECURITY.md](SECURITY.md#privacy-face-data)) |
| `export <model>` / `download <model>` | export to TensorRT / download weights. Targets include `all`, `essential`, `status`; `export` also takes `yolo`, `scrfd`, `arcface`, `clip-image`, `clip-text`, `ocr`; `download` also takes `ocr` |
| `curation [up\|down\|logs\|status]` | start, stop, follow or list the curation workers (`status` is the default) |
| `sample coco [--full]` | fetch the public COCO sample into `data/samples/` using the export image |
| `config show` | print `.env` with every token, key, secret and password redacted |
| `cropwright password` | set the Cropwright basic-auth login (an empty answer generates one) |
| `gpu plan` | print the GPU keys from `.env`, then where the PE image encoder runs and whether this host can honour it (see [Spreading the PE encoder over GPUs](#spreading-the-pe-encoder-over-gpus)) |
| `vlm ...` | local VLM catalog and switching, see [Choosing a VLM](#choosing-a-vlm) |
| `train-mode on\|off` | stop (and later restart) the local VLM, plus the segmenter when it shares the training GPU, around a training run |
| `upgrade [--version vX.Y.Z]`, `repair`, `uninstall [...]` | run this install's own `setup-openprocessor.sh` against this directory (installed directories only) |
| `update` | in a checkout: pull images and rebuild; in an install directory: same as `upgrade` |
| `profile [minimal\|small\|standard\|full] [gpu_id]` | show the Triton instance profile in `.env`, or switch to one: rewrites `models/*/config.pbtxt` and sets `GPU_PROFILE` and `OP_INGEST_GPU_DECODE` in `.env` (checkout only; restart `triton-server` after; see [GPU sizing](docs-site/docs/deployment/gpu-sizing.mdx#switching-profile-later)) |
| `test [quick\|full\|visual]` | smoke test, `tests/test_full_system.py`, or visual validation (checkout only; `full` and `visual` need `.venv`) |
| `clean [cache\|models\|exports\|all]` | remove caches, downloaded weights, or TensorRT plans (asks first) |
| `setup [flags]` | run `scripts/setup.sh` (checkout only) |
| `version` | print the CLI version |

### Choosing a VLM

Labeling and region verification call an OpenAI-compatible vision-language
model. A deployment has a **registry** of endpoints (the built-in `env`
endpoint from `OP_VLM_URL`, plus any you save through the API) and each project
**activates** one. A project calls no VLM until it does, so installing a VLM never
makes an ingest wait for it. Three ways to get a model:

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

`use`, `apply` and `probe` need `OP_LOCAL_VLM_ENDPOINT` in `.env` (the installer
sets it to `env` with the `vlm` tier). To add the in-compose VLM to an install
that was made without it, re-run `./setup-openprocessor.sh --tiers <your
tiers>,vlm`, or set `OP_LOCAL_VLM_ENDPOINT=env` in `.env`, keep `vlm` in
`COMPOSE_PROFILES` and run `docker compose --profile vlm up -d vlm`.

### Offline installs from a local bundle

`scripts/release/build_deploy_bundle.sh vX.Y.Z [SRC] [OUT]` builds the release
assets locally; `--release-dir OUT --version vX.Y.Z` installs from them (still
checksum-verified). Images still come from a registry
unless they are already present locally (use `--image-tag` for local builds).
A later re-run, `repair` or `./openprocessor upgrade` of that install needs only
`--release-dir OUT`: the installed version is reused, and `--version` is for an
upgrade to another release.

### Spreading the PE encoder over GPUs

The PE image encoder is about 99 percent of the GPU work of `/ingest`, and one
card tops out near 23 images per second on 640 x 480 frames. A second (or third)
card runs a second (third) copy of the encoder: list the cards in
`TRITON_GPU_ID`.

```bash
# .env: the PE encoder on GPU 0 and GPU 2; everything else stays on GPU 0
TRITON_GPU_ID=0,2
API_GPU_ID=0
# or at install time
./setup-openprocessor.sh --gpu-plan triton=0+2,vlm=1
```

Rules, all enforced:

- The ids are host indexes from `nvidia-smi`. Triton sees the listed cards
  numbered in host order, so the **lowest id is the primary card**: the
  detector, face, MobileCLIP and OCR models stay there (their configs pin
  `gpus: [ 0 ]`), and `API_GPU_ID` should name it. Every other listed card runs
  only the encoder, one instance of about 1.5 GB (`GPU_MIB_PER_INSTANCE` in
  `models/pe_image_encoder/config.pbtxt`).
- One Triton, one dynamic-batcher queue: whichever card is free takes the next
  batch, so a slower card simply takes fewer. The API and its clients are
  unchanged.
- All listed cards must share one compute capability, because they load one
  TensorRT plan. Cards of different models with the same capability (an RTX A6000
  and an RTX 3080 Ti) load it, with a TensorRT warning in the Triton log that a
  plan is not portable across device models.
- Fail closed. `./openprocessor gpu plan` exits 1 for an id the host does not
  have, a repeated id, mixed compute capabilities, or a card with less than the
  encoder's budget free; the installer refuses the same cases (`--force` only
  overrides the free-memory check) and a container asked for a missing device
  does not start. A card shared with the VLM or the segmenter is planned with the
  encoder's memory subtracted from theirs.
- `docker-compose.yml` passes `TRITON_GPU_ID` through unsplit (the NVIDIA runtime
  reads `0,2` as two devices), so no compose edit is needed.

Measured scaling is in [docs/PERFORMANCE.md](docs/PERFORMANCE.md#multi-gpu-pe-encoding-228).

---

## Install from source

For development, or to build the images yourself.

```bash
git clone https://github.com/davidamacey/OpenProcessor.git && cd OpenProcessor && ./scripts/setup.sh
```

`setup.sh` flags (`./scripts/setup.sh --help` prints them):

| Flag | Meaning |
|---|---|
| `--yes`, `-y` | skip confirmation prompts and accept the defaults |
| `--profile=NAME` | force a GPU profile: `minimal`, `small`, `standard` or `full` (see [GPU profiles](docs-site/docs/deployment/gpu-sizing.mdx#gpu-profiles)) |
| `--gpu=ID` | use this GPU (default 0) |
| `--skip-export` | skip the TensorRT export (models already exported) |
| `--skip-download` | skip the model download |
| `--skip-start` | do not start the services |
| `--force` | overwrite an existing `.env` (otherwise setup never touches one) |
| `--curation` | print the curation next steps at the end |

```bash
./scripts/setup.sh --yes                              # accept all defaults
./scripts/setup.sh --profile=standard --gpu=0 --yes   # choose profile and GPU
```

### What setup does

1. Checks Docker, Docker Compose, the NVIDIA driver and Docker's GPU runtime.
2. Detects your GPU and selects a profile from its free VRAM.
3. Creates the working directories.
4. Downloads the model weights from public sources (about 500 MB for all, about
   250 MB for the essential set). A failed download is a warning; export retries it.
5. Pulls the `api` and `triton-server` images from Docker Hub. If the pull fails
   it builds them from the Dockerfiles instead.
6. Generates the Triton model configs and `.env` (the OpenSearch heap is sized
   from host RAM, as for the installer; see [OpenSearch heap sizing](#opensearch-heap-sizing)).
7. Exports the models to TensorRT (the `minimal` profile exports the essential
   set only). The script estimates 45 to 60 minutes for all models, once; the
   engines are cached on disk.
8. Starts the services and runs a smoke test.

### Working in a checkout

Manage the stack with `make` (`make up`, `make curation-up`, `make down`,
`make status`). The `./openprocessor` CLI (and its shim
`./scripts/openprocessor.sh`) uses the project `openprocessor` and warns when `.env` has no
`COMPOSE_PROJECT_NAME` line (the `.env` that `setup.sh` writes has none); set the line to
choose another name:

```bash
echo 'COMPOSE_PROJECT_NAME=openprocessor' >> .env
```

When you run `docker compose` by hand, pass `-p <project>` so you never act on
another stack that uses the default name.

### Manual installation

To run the steps yourself instead of `setup.sh`:

1. Clone the repository and create the directories:
   `mkdir -p pytorch_models logs cache/huggingface outputs test_results`.
2. `./openprocessor download all` (or `essential`; `download status` checks it).
3. `cp env.template .env` and edit it.
4. `./openprocessor start` (or `make up`): the `triton-server` container must be
   running for the export. This also starts the rest of the core stack.
5. `./openprocessor export all` (or `essential`; `export status` checks it). Commands are in
   the [CLI table](#the-openprocessor-cli).
6. `./openprocessor start` again, then `make curation-up` if you want the curation workers.
7. Verify: `./openprocessor status`, `./openprocessor test quick`,
   `curl http://localhost:4603/health`.

---

## Troubleshooting

Exit codes of the installer:

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
- **GPU out of memory ("CUDA out of memory", services crash)**: use a smaller
  profile (`--profile minimal` on the installer; in a checkout
  `./openprocessor profile minimal`, or `scripts/setup.sh --profile=minimal --force`).
  Or move other GPU services to another card: set `SEGMENTER_GPU_ID`,
  `VLM_GPU_ID` and `API_GPU_ID` in `.env`, run `./openprocessor gpu plan`, then
  `./openprocessor start`. `nvidia-smi` shows what else holds the GPU.
- **TensorRT export hangs or fails**: the export needs the `triton-server`
  container running (it builds the engines with `trtexec` there) and frees GPU
  memory by unloading models first. Stop other GPU processes, then
  `./openprocessor start` and `./openprocessor export all`. Compare the driver
  (`nvidia-smi`) with the container's
  (`docker compose -p <project> exec triton-server nvidia-smi`).
- **Triton shows "UNAVAILABLE" for models**: `./openprocessor export status`
  shows which exports exist, `./openprocessor logs triton-server` the reason,
  and `ls -la models/*/config.pbtxt` confirms the configs are there.
- **"Permission denied" from Docker**: `sudo usermod -aG docker $USER`, then log
  out and back in. `ls -la /var/run/docker.sock` shows the socket permissions.
- **Services will not start** (`docker compose up` fails): check port conflicts
  (`lsof -i :4603` for the API, `lsof -i :4600` for Triton) and disk space
  (`df -h`). Run `docker compose -p <project> up` without `-d` to see the output.

---

## Upgrade, repair, rollback, uninstall

Run these from the install directory:

```bash
./openprocessor upgrade [--version vX.Y.Z]
./setup-openprocessor.sh --repair
./setup-openprocessor.sh --rollback
./setup-openprocessor.sh --uninstall [--purge-volumes] [--purge-data] [--remove-images]
```

What each step does, the backups it makes, the 0.6.x image-split note and the data backup are in
[Upgrade, repair and back up](docs-site/docs/operations/upgrade-and-backup.mdx). The flags are in the
[table above](#installer-flags).

### From a checkout

```bash
git pull
./openprocessor update      # pull images and rebuild
make up                     # recreate the services whose configuration changed
make curation-up            # same, for the curation workers
make dev-restart            # reload the source in api and the workers
```

The processes import their source once at start, so restart them after every
pull. `make rebuild` rebuilds the containers without cache. After major updates
you may need to re-export the TensorRT models (`./openprocessor export all`, then
`./openprocessor restart`).

To remove a checkout's stack:

```bash
./openprocessor stop                             # keeps volumes
docker compose -p openprocessor down -v          # -v removes volumes
docker compose -p openprocessor down --rmi all   # also remove the images
```

Use your own project name in place of `openprocessor` if you set
`COMPOSE_PROJECT_NAME`. Without `-p`, `docker compose` picks the project from
`.env` or the directory name, which may be a different stack. To clean up files:

```bash
rm -rf pytorch_models/*.pt pytorch_models/*.onnx
rm -rf models/*/1/model.plan
rm -rf .env docker-compose.local.yml
rm -rf cache/ logs/ outputs/ test_results/
```

---

## Getting Help

- **Documentation:** See [README.md](README.md) and [CLAUDE.md](CLAUDE.md)
- **Building and scanning release images:** see [CONTRIBUTING.md](CONTRIBUTING.md#building-and-scanning-images)
- **Issues:** Report bugs at [GitHub Issues](https://github.com/davidamacey/OpenProcessor/issues)
