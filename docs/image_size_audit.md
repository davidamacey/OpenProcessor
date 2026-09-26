# Docker image size audit

Status: audit and plan only. No Dockerfile has been changed and no image
rebuilt on this branch. Measured 2026-09-26 against the published
`davidamacey/openprocessor*:latest` images (same layer layout as the
`cutover-local` builds of main `1ecdef8d`).

## 1. Current sizes

| Image | Size | Dockerfile | Base |
|---|---:|---|---|
| `openprocessor` (API + curation workers) | 21.4 GB | `Dockerfile` | `python:3.13-slim-trixie`, multi-stage |
| `openprocessor-evaluator` | 15.0 GB | `docker/evaluator/Dockerfile` | `python:3.13-slim-trixie`, single stage |
| `openprocessor-trainer` | 7.7 GB | `docker/trainer/Dockerfile` | `python:3.12-slim-trixie`, multi-stage |
| `openprocessor-segmenter` | 7.4 GB | `docker/segmenter/Dockerfile` | `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime` |

No two images share any layer above the Debian base, so a full install pulls
all of them in full.

## 2. Layer breakdown (`docker history --no-trunc`)

### API (`openprocessor`, 21.4 GB)

| Layer | Size |
|---|---:|
| `COPY --from=builder /root/.local` (main site-packages) | 13.9 GB |
| `COPY --from=builder /opt/venv-y11` (YOLO11 export venv) | 7.08 GB |
| runtime apt (`curl jq procps libgl1 libglib2.0 libgomp1` + `upgrade`) | 279 MB |
| Debian + Python base | ~118 MB |
| `src/`, `scripts/`, `VERSION` | 4.4 MB |

Largest entries in the main site-packages (`/home/appuser/.local`):

| Package | Size | Needed by |
|---|---:|---|
| `tensorrt_libs` (tensorrt-cu13 11.1.0.106) | 4.32 GB | engine builds only (`export/`, `POST /models/{name}/export`) |
| `nvidia/*` CUDA wheels (cu13 1.7 GB, cudnn 0.9, cublas 0.8, nccl 0.24, cusparselt 0.22, nvrtc 0.22) | 4.24 GB | CUDA torch |
| `torch` | 1.18 GB | runtime (see section 3) |
| `triton` (compiler shipped with CUDA torch) | 0.90 GB | CUDA torch |
| `onnxruntime` (gpu) | 0.40 GB | runtime (PE text ONNX backend, probe models) |
| `cupy` + `cupy_backends` + `cupyx` | 0.28 GB | optional GPU clustering probe; pulled in by `nvidia-modelopt[onnx]` |
| `llvmlite` + `numba` | 0.21 GB | `umap-learn` / `hdbscan` |
| `_polars_runtime_32` | 0.17 GB | `nvidia-modelopt` dependency |
| `opencv_python.libs` + `opencv_contrib_python.libs` + `cv2` | 0.30 GB | two OpenCV wheels (paddleocr pulls contrib) |

Largest entries in `/opt/venv-y11` (7.1 GB): `tensorrt_libs` 4.32 GB, CPU
`torch` 0.77 GB, `onnxruntime` 0.40 GB, `cupy` 0.23 GB, `polars` 0.17 GB,
`scipy` 0.11 GB, OpenCV 0.27 GB.

**Duplicate measurement.** Of the 6.9 GB in the venv's site-packages, 5.96 GB
is byte-identical (compared with `diff -rq -x __pycache__`) to the same
top-level entry in the main site-packages. 60+ distributions are installed at
the exact same version in both environments, including
`tensorrt_cu13{,_libs,_bindings}==11.1.0.106`, `onnxruntime_gpu==1.24.4`,
`cupy_cuda12x==14.2.0`, `nvidia_modelopt==0.47.0`, `polars==1.44.2`,
`scipy==1.18.1`, `onnx==1.21.0`, `sympy`, `matplotlib`. Only these genuinely
differ: `torch 2.14.0+cpu` / `torchvision +cpu`, `ultralytics==8.3.253`,
`numpy`, `ml_dtypes`, `protobuf`, `pyyaml`, `opencv_python_headless`
(~0.95 GB total).

### Evaluator (15.0 GB)

| Layer | Size |
|---|---:|
| single `pip install` of `requirements.txt` + bake-off extras + `mlflow` | 14.2 GB |
| apt incl. `build-essential cmake git pkg-config` (kept in the final image) | 673 MB |
| Debian + Python base | ~118 MB |

The pip layer is the API's main dependency set installed a second time
(same top entries: `tensorrt_libs` 4.32, `nvidia` 4.24, `torch` 1.18,
`triton` 0.90 GB), system-wide instead of `--user`, so it cannot dedupe
against the API image. `perception_models` is not installed here.

### Trainer (7.7 GB)

| Layer | Size |
|---|---:|
| `COPY --from=builder /root/.local` | 7.31 GB |
| runtime apt | 279 MB |

Site-packages: `nvidia` 3.2 GB, `torch` 1.18, `triton` 0.89, `onnxruntime`
0.33, `polars` 0.17, `pyarrow` 0.14, `mlflow` 0.11 GB. Both `opencv_python`
and `opencv_python_headless` are installed (~0.2 GB overlap: Ultralytics
pulls the non-headless wheel). No TensorRT. Python 3.12, so it cannot share
site-packages layers with the 3.13 images.

### Segmenter (7.4 GB)

| Layer | Size |
|---|---:|
| base `pytorch/pytorch` (`/opt/conda`) | 6.24 GB (of which `pkgs/` conda cache 0.58 GB, base-image layer) |
| apt `gcc g++ libgl1 libglib2.0-0 wget` + `upgrade` | 465 MB |
| `pip install /opt/sam3` + service deps | 502 MB |
| `COPY /src/sam3 /opt/sam3` (upstream source tree, incl. 55 MB `assets/`) | 73 MB |

`gcc`/`g++` are required at runtime (`SEGMENTER_COMPILE=1` builds inductor
kernels). The installed `sam3` package already carries its own
`assets/bpe_simple_vocab_16e6.txt.gz`, so `/opt/sam3` is only needed at
build time.

## 3. What the API image actually needs at runtime

Inference runs on Triton, but the API process is not a thin client:

| Dependency | Runtime importer | Verdict |
|---|---|---|
| CUDA `torch` / `torchvision` | `src/routers/health.py` (module-level, `torch.cuda.*` in `/health`), `src/services/detection/ensemble_nms.py` (module-level, imported by curation ingest), `src/clients/pe_encoder.py` (PyTorch PE text fallback), `src/services/curation/probe_models.py` | **stays**. Swapping to CPU torch would save ~5.5 GB but changes `/health` output and moves PE fallback / probe inference off the GPU. Owner decision, not a size-only change |
| `ultralytics` 8.4 | `src/clients/triton_client.py` (module-level `LetterBox`), probe models, `export_yolo26.py` | stays |
| `perception_models` (`core`) | PE text/image torch fallback | stays |
| `onnxruntime-gpu` | PE text ONNX backend, probe models | stays |
| `faiss-gpu-cu12`, `scikit-learn`, `umap-learn`, `hdbscan` | clustering | stays (cuML is not installed; GPU clustering is probed at runtime) |
| `cupy` | optional GPU clustering memory probe | stays; must be declared explicitly if `nvidia-modelopt` leaves the runtime |
| `paddleocr` | `src/services/detection/region_lean.py` (lazy) | stays |
| `tensorrt-cu13` (4.3 GB) | **only** `export/*` and `src/services/model_export.py`, which spawns `export/export_models.py` / `export_yolo26.py` as a subprocess for `POST /models/{name}/export` | export-only |
| `nvidia-modelopt[onnx]`, `onnx-graphsurgeon`, `onnxsim`, `onnxslim`, `onnxscript`, `polygraphy` | export scripts | export-only |
| `/opt/venv-y11` (7.1 GB) | re-exec target of `export/export_models.py` for YOLO11 EfficientNMS | export-only |

No other `src/` path imports `tensorrt` or `modelopt`. Training promote writes
ONNX/config into the model repo; it does not build a TensorRT engine in the
API.

The curation workers (`curation-*-worker`, `curation-cluster-refresh`) run
the same image. They must keep sharing whatever image the API uses: giving
them their own image would add a second multi-GB pull on the same host.

## 4. Model weights and engines baked into images

None. Searched every image for `*.pt *.pth *.plan *.engine *.onnx
*.safetensors *.ckpt` and `*.bin` over 1 MB, plus the HF/torch caches:

| Image | Hits | Size |
|---|---|---:|
| API | only `onnx` / `onnxruntime` package test fixtures (`onnx/backend/test/data/**`), once per environment | < 1 MB each, ~40 MB of `tests/` dirs total |
| evaluator, trainer | same package test fixtures | < 1 MB each |
| segmenter | none; `/opt/conda/pkgs` conda cache 0.58 GB (base-image layer) | — |
| all | `~/.cache` empty (4–12 KB) | — |

`.dockerignore` excludes `*.pt *.onnx *.safetensors *.pth models/*/1/*.plan`,
and the model-repo seed (`/opt/openprocessor/model_repo_seed`) copies only the
tracked config tree. All weights are downloaded at install time into
bind-mounted `models/`, `pytorch_models/` and the HF cache.

## 5. Estimated after-sizes

| Image | Now | Phase A (behaviour-identical) | Phase B (export split) |
|---|---:|---:|---:|
| `openprocessor` (API/workers) | 21.4 GB | **~15.4 GB** | **~9.9 GB** |
| `openprocessor-export` (new target) | — | — | ~15.3 GB standalone, **~5.4 GB incremental** over the API |
| `openprocessor-evaluator` | 15.0 GB | ~15.4 GB standalone, **~0.5 GB incremental** | ~10.4 GB standalone, ~0.5 GB incremental |
| `openprocessor-trainer` | 7.7 GB | ~7.5 GB | ~7.5 GB |
| `openprocessor-segmenter` | 7.4 GB | ~7.3 GB | ~7.3 GB |
| **Pull: API + evaluator** | 36.4 GB | ~15.9 GB | ~15.8 GB (incl. export) |

Phase B does not cut the first-install download much below Phase A (the
installer still needs the export toolchain once). Its value is a 9.9 GB
long-running image, a smaller attack surface, and the option to
`docker image rm` the export image after setup. Phase A carries almost all
of the download win and changes no behaviour.

## 6. Implementation plan

### Phase A: behaviour-identical (do first)

1. **Dedupe the YOLO11 venv** (`Dockerfile`, builder stage, after the venv
   install). Add `docker/dedupe_site_packages.sh <dup_site> <canon_build_site>
   <canon_runtime_site>`: for every top-level entry of
   `/opt/venv-y11/lib/python3.13/site-packages` (skip `__pycache__` and
   existing symlinks) whose same-named entry exists in
   `/root/.local/lib/python3.13/site-packages` and is identical
   (`diff -rq --no-dereference -x __pycache__` for dirs, `cmp -s` for
   files), `rm -rf` it and replace it with an absolute symlink to
   `/home/appuser/.local/lib/python3.13/site-packages/<name>` (the runtime
   path; dangling in the builder, valid after `COPY`). `*.dist-info` dirs
   differ (RECORD/INSTALLER), so they stay real and `importlib.metadata`
   still resolves inside the venv. Saves ~5.96 GB.
   - Do **not** use `--system-site-packages`: in a venv the user site is
     inserted before the venv site, so `ultralytics` 8.4 would shadow 8.3.
   - Must verify that `$ORIGIN`-relative RPATHs behave the same: in the y11
     venv, `import tensorrt, onnxruntime, cupy, modelopt.onnx` in the old and
     new image, dump `/proc/self/maps` `.so` paths (normalised with
     `realpath`), and diff.
2. **Evaluator shares the API layers.** Move it into the root `Dockerfile`
   as a target so BuildKit reuses the same stages:
   - Stages: `builder` (unchanged) → `runtime-base` (apt, user, `COPY
     .local`, `COPY venv-y11`, common `ENV`) → `evaluator` (FROM
     `runtime-base`; `USER appuser`; `pip install --user --no-cache-dir -r
     bakeoff-requirements.txt && pip install --user --no-cache-dir mlflow`
     in the same order as today; `COPY src scripts examples`; evaluator
     `LABEL`s, `PYTHONPATH=/app`, `OP_BAKEOFF_JOBS_DIR`, `HEALTHCHECK`,
     `ENTRYPOINT`/`CMD` copied verbatim) → `api` (last stage, so an untargeted
     `docker build .` still yields the API).
   - No compilers in the evaluator. If any bake-off/mlflow dependency lacks a
     cp313 wheel, build it in a throwaway `evaluator-wheels` stage and install
     the wheel.
   - Update `docker-compose.dev.yml` (`curation-evaluator.build`: `dockerfile:
     Dockerfile`, `target: evaluator`), `scripts/release/build_and_publish.sh`
     (`IMAGE_SPECS` gains a 4th `target` field, passed as `--target`; build
     `api` before `evaluator`), `tests/test_release_build_and_publish.py`
     fixture, `tests/test_compose_contract.py`
     (`_BUILD_SHA_DOCKERFILES` → check the `evaluator` and final stages of
     `Dockerfile`), and delete `docker/evaluator/Dockerfile`. Fix the comment
     references in `docker-compose.yml`, `docker/trainer/Dockerfile` and
     `docs/design/curation_design_rationale.md`.
3. **Move `ARG/ENV OP_BUILD_SHA` + the revision `LABEL` to the end** of each
   final stage (API, evaluator, trainer). Today they sit before the apt
   layer, so every new commit invalidates the multi-GB layers' cache keys and
   an upgrade re-pulls everything. After the move, an upgrade with unchanged
   requirements pulls only the source layers (MBs). The existing
   final-stage test still holds.
4. **Segmenter**: install `sam3` in a builder stage (`pip wheel /opt/sam3`)
   and install the wheel in the runtime, so `/opt/sam3` (73 MB) is not
   shipped. Keep `gcc`/`g++`.
5. **Trainer**: optional ~0.1–0.2 GB by constraining Ultralytics to
   `opencv-python-headless` (pip `--constraint` that pins `opencv-python`
   out is fragile; only do it if a smoke train passes).

Verification for Phase A (every changed image, local-only tags such as
`openprocessor:slim-test`, plain `docker run --rm`, no compose):
- `python -c` imports of `src.main`, `src.clients.pe_encoder`,
  `src.services.model_export`, `scripts.curation.bakeoff.bakeoff_runner`,
  `docker/trainer` `trainer`, segmenter `main`;
- `/opt/venv-y11/bin/python -c 'import ultralytics, torch, tensorrt,
  onnxruntime, modelopt.onnx; print(ultralytics.__version__)'` → `8.3.253`;
- `python -m export.preflight`; `--help` of `export/export_models.py`,
  `bakeoff_runner`, `trainer`;
- `pip check` in both environments; `pip freeze` diff old vs new must be
  empty for the API env and the y11 venv, and only additive for the
  evaluator (plus `perception_models`);
- GPU stack (coordinator, on the full deployment): YOLO11 `trt_end2end`
  export via the venv, `export_yolo26.py`, `POST /models/{name}/export`,
  a bake-off job, `/health`.

### Phase B: `openprocessor-export` target (needs an owner decision)

This changes behaviour, because `POST /models/{name}/export` (and `/models/
upload` with export) currently runs the export subprocess inside the running
API container.

1. **Requirements split**: `requirements.txt` → runtime set;
   new `requirements-export.txt` = `tensorrt-cu13==11.1.0.106`,
   `nvidia-modelopt[onnx]`, `onnx-graphsurgeon`, `onnxsim`, `onnxslim`,
   `onnxscript`. Declare `cupy-cuda12x` explicitly in the runtime set
   (today it arrives only via modelopt). Keep the tensorrt pin tests pointing
   at the new file.
2. **Dockerfile targets** (root `Dockerfile`): `builder` (runtime deps) →
   `builder-export` (FROM builder; `pip install --user -r
   requirements-export.txt`; build `/opt/venv-y11`; dedupe it against the
   export user site) → `runtime-base` → `export` (FROM `runtime-base`;
   `COPY --from=builder-export` only the added packages: install the export
   set with `--target /opt/op-export/site` + a `.pth`, so the layer is the
   delta, not a second copy of `.local`; `COPY venv-y11`) → `evaluator`
   (FROM `runtime-base`) → `api` (FROM `runtime-base`, last).
3. **Which service uses which image**: `yolo-api` and all curation workers
   → `openprocessor`; new compose service `model-export` (profile
   `setup`, image `${OP_EXPORT_IMAGE:-.../openprocessor-export:<ver>}`, same
   volumes/GPU reservation as `yolo-api`, no ports) → `openprocessor-export`;
   `curation-evaluator` → `openprocessor-evaluator`.
4. **Installer model setup** (`scripts/lib/model_setup.sh`, installer plan
   §4.1): replace `dc run --rm --no-deps -T yolo-api python /app/export/...`
   with `dc run --rm --no-deps -T model-export python /app/export/...`
   for groups 0–6 (preflight, weights, YOLO11, MobileCLIP, faces, OCR, PE);
   the seed `cp -rn` and the profile-config step can stay on `yolo-api`.
   `export/preflight.py`'s `import tensorrt` check moves to the export image;
   the API preflight checks `import core` and `torch.cuda` only. Add
   `OP_EXPORT_IMAGE` to `images.lock`, the `IMAGE_SPECS` table and the
   compose image-override contract test. `openprocessor repair --images` and
   the Makefile export targets use the same service. Offer `docker image rm`
   of the export image at the end of install (keep by default so re-exports
   after a Triton upgrade don't re-pull).
5. **`POST /models/{name}/export`**: either (a) hand the job to the export
   container via the existing job-file pattern (the API writes
   `/jobs/model_export/<id>.json`; a `model-export --watch` process runs
   `export_models.py`/`export_yolo26.py` and writes status back, like the
   trainer), or (b) keep the endpoint but return a clear 503 "export image
   not running" when `tensorrt` is not importable. (a) preserves behaviour.

### Not recommended now

- CPU-only torch in the runtime (~5.5 GB): changes `/health` GPU fields and
  where PE fallback / probe inference run.
- Stripping package `tests/` dirs (~40 MB) or `__pycache__`: small win, some
  packages import from their own test helpers.
- Rebasing the trainer on Python 3.13 / the API torch layer to share
  ~4 GB of CUDA wheels: only dedupes if both are built in the same BuildKit
  graph with the same torch wheel. Revisit once Phase B lands.
- Segmenter on `python:*-slim` + pip torch: would change the pinned
  `torch 2.7.0+cu126` / conda environment the model is validated on.

## Owner decision (2026-09-26)

**Plan B** (includes Plan A's dedupe and shared layers): a slim runtime API/worker image plus a separate export image holding the model-export environment. Better layering throughout, so that pulls dedupe across images.

To evaluate when the task resumes: use an upstream image as the export base instead of building our own, e.g. the official `ultralytics/ultralytics` GPU image, or NVIDIA's TensorRT container (`nvcr.io/nvidia/tensorrt`). Compare them on:
- size and layer reuse with our images;
- the TensorRT version matching the Triton server's (engines must be built with the same TRT major/minor);
- the license (ultralytics is AGPL-3.0, compatible with this project) and the NGC terms for redistribution;
- digest pinning.

Also: make torch a lazy import on /health and ingest where possible, so the runtime image can drop more.
