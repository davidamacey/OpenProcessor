# 13. Exact pins for every Python dependency (inventory and bump plan)

Status: plan only. This file changes no manifest. It is the work order for the commits in section 7.

Date of the survey: 2026-11-04. Versions are the newest non-yanked, non-pre-release files on PyPI that day.
"Installs" was checked by resolving each candidate set with a throwaway resolver run (uv and pip, outside
the repository, for the target Python of each image) and, for the CPU test set, a full `pip install
--dry-run --report`. Nothing was run on a GPU and no image was built, so every "works" below still needs
the checks in section 6.

## 1. Summary

The owner's rule (2026-11-04): every requirements file and `pyproject.toml` carries exact `==` pins at the
newest version that installs together and works, identical across the manifests that share a package.
There is no lock-file tool; Dependabot bumps the exact pins afterwards.

Today most runtime packages are unpinned or floors (`torch`, `numpy`, `opencv-python`, `tritonclient`,
`httpx`, `opensearch-py>=2.3.0`, and so on), so two builds on different days differ. This plan turns
about 106 packages into `==` pins (95 change their spec; 9 are new for the control plane, section 4). The
few that deliberately stay below the newest release are listed with the reason and the source of the rule:

| Held back | Pin | Newest | Why (hard constraint) |
|---|---|---|---|
| Ultralytics | `8.4.197` | 8.4.202 | The program pins it; it moves in B3 (`04_b3_ultralytics_one_version.md`). Moves in B3 |
| Ultralytics (YOLO11 export venv) | `8.3.253` | 8.4.202 (8.3.253 is the last 8.3.x) | The vendored end2end patch targets the 8.3 exporter; removal planned in `model_family_versions_plan.md` |
| TensorRT (`tensorrt-cu13`) | `11.1.0.106` | 11.4.0.106 | Must equal the Triton image TensorRT; `Dockerfile.triton` asserts `TRT_VERSION`. Moves only with the Triton base (ENV-12) |
| `onnx` | `1.21.0` | 1.23.2 | `nvidia-modelopt` 0.48.0 requires `onnx~=1.21.0` on Linux |
| `onnxruntime` / `onnxruntime-gpu` | `1.30.0` | 1.31.1 | `nvidia-modelopt` 0.48.0 requires `onnxruntime-gpu<1.31,>=1.29` on Linux x86_64; runtime and export images must resolve one version |
| kombu | `5.6.2` | 5.7.0 | Owner decision 2026-11-04 (review R-10): Celery 5.6.3 (the newest Celery) declares `kombu>=5.6.0` with no cap, and kombu 5.7.0 (2026-11-03) has no Celery release against it. Moves when a Celery release declares kombu 5.7 support |
| redis (redis-py) | `6.4.0` | 8.1.0 | kombu 5.6.2's `redis` extra accepts `redis>=4.5.2,<6.5` (excluding 4.5.5 and 5.0.2); 6.4.0 is the newest inside it. Moves with kombu |
| `numpy` in the segmenter image only | `1.26.4` | 2.5.4 | Upstream SAM 3 declares `numpy>=1.26,<2` |
| `opencv-python-headless` in the segmenter image only | `4.11.0.86` | 5.0.0.93 | OpenCV 4.12 and later need `numpy>=2` |
| `torch` / `torchvision` in the segmenter image | base image: `2.14.1` | 2.15.0 / 0.29.2 | Not a pip pin: the `pytorch/pytorch:2.14.1-cuda12.6` base ships it. Moves only with a base image bump |

Findings worth knowing before reading the tables:

1. Several stale statements in the repo become wrong with this plan: the `.github/dependabot.yml` ignore
   rules for `onnxruntime-gpu` (says `~=1.24.2`; modelopt now allows 1.29 to 1.30) and for `huggingface-hub`
   major (says tokenizers needs hub below 2; `tokenizers` 0.23.3 and `transformers` 5.22.1 now accept hub 2.x).
   `docs/design/v070_rebuild/reference/environment_and_setup.md` section 7.1 still shows `fastapi==0.143.3`; the
   manifests already carry `0.162.0` and `pydantic==2.14.2`.
2. The trainer image installs torch with `--extra-index-url .../cu128`. That index now stops at torch
   2.11, while PyPI has 2.15.0, so pip silently takes the PyPI wheel (CUDA 13.2 libraries). The index is
   decorative today. The plan moves it to `cu132`, which has 2.15.0 (section 3.5).
3. Dependabot cannot see `RUN pip install` pins. The Triton, segmenter and hardened-Triton images and the
   evaluator's bare `mlflow` hold pins inline, so no tool or test sees them. The plan moves them to small
   requirements files (section 3.6).
4. `nvidia-modelopt[onnx]` pulls `cupy-cuda13x`, while the runtime set pins `cupy-cuda12x`. Both install
   into the export image and both provide the top-level `cupy` module (pre-existing; a smoke test is in
   section 6).
5. About 15 pins are less than a week old (for example `ultralytics` 8.4.202, `tritonclient` 2.74.0,
   `kombu` 5.7.0, `SQLAlchemy` 2.1.8, `psycopg` 3.3.8, `filelock` 4.2.8, `virtualenv` 21.15.5). The owner
   asked for newest; the register notes them so an implementer can fall back one patch if a fresh release
   is pulled or yanked before the bump lands.

Counts: eight pins cross a major boundary versus the manifest's current floor or inline pin (OpenCV 4 to 5,
`huggingface-hub` 1 to 2, `pandas` 2 to 3, `opensearch-py` 2 to 3, `cupy-cuda13x` 13 to 14, `pytest` 8 to
9, `pytest-asyncio` 0.x to 1.x, `pytest-cov` 5 to 7). The repo's own CPU test venv already runs several of
them (OpenCV 5.0.0.93, `opensearch-py` 3.2.0, `pytest` 9.1, `pytest-asyncio` 1.4, `transformers` 5.19), so
the offline suite is not new to those.

## 2. Constraints and their sources

| Constraint | Source |
|---|---|
| API image Python is 3.13 (`python:3.13-slim-trixie`); trainer image is 3.12; segmenter base is Ubuntu 24.04 system Python 3.12; Triton base is Ubuntu 24.04 (3.12) | `Dockerfile`, `docker/trainer/Dockerfile`, `docker/segmenter/Dockerfile`, `Dockerfile.triton` |
| `pyproject.toml` `requires-python >=3.12`; the CPU test venv and CI use 3.12 | `pyproject.toml`, CONTRIBUTING.md |
| `faiss-gpu-cu12` 1.14.1.post1 needs Python `<3.14` and `numpy>=2,<3`; there is no CUDA 13 build on PyPI | PyPI metadata; the API image on 3.13 satisfies it, a move to 3.14 is blocked by it |
| Triton base is CUDA 13.3 / TensorRT 11.1; the `cu132` torch index is inside that CUDA major | `Dockerfile.triton` header and `TRT_VERSION` |
| `tensorrt-cu13` must equal the Triton TensorRT, asserted at image build; also pinned in `requirements-export.txt`, `requirements-export-y11.txt` and the `export` extra | `Dockerfile.triton`, `tests/test_requirements_contract.py::test_tensorrt_pin_matches_the_triton_image_everywhere` |
| `nvidia-modelopt` 0.48.0 `[onnx]`: `onnx~=1.21.0`, `onnxruntime-gpu>=1.29,<1.31`, `onnxslim>=0.1.78`, `onnx-graphsurgeon>=0.6.1`, Python `<3.15` | PyPI metadata |
| torch and torchvision move as a pair: torch 2.15.0 with torchvision 0.29.2 (torchvision requires `torch>=2.14`); the same pair exists on PyPI, the CPU index and the `cu132` index | PyPI, download.pytorch.org |
| Ultralytics 8.4.197 needs `numpy>=1.23`, `opencv-python!=4.13.0.90,>=4.7`, `torch>=1.8`, `polars`, `ultralytics-platform` | PyPI metadata |
| NumPy 2.5.4 with OpenCV 5.0.0.93 (needs `numpy>=2`), faiss 1.14.1.post1 (`numpy<3`), SciPy 1.18.1, pandas 3.1.0 | resolver run |
| `fastapi` pin is tied to the committed OpenAPI; bump only with `make contracts`; `pydantic` must be the version fastapi was tested with | `requirements-test.txt` header, `CLAUDE.md` |
| `mlflow` client must share a major with the MLflow server image (`ghcr.io/mlflow/mlflow:v3.17.0` in `docker-compose.yml`) | `docker/trainer/requirements.txt` comment |
| Segmenter: SAM 3 declares `timm>=1.0.17`, `numpy>=1.26,<2`, `ftfy==6.1.1`, `iopath>=0.1.10` | upstream `pyproject.toml` of the pinned SAM 3 source |
| `perception_models` stays `--no-deps` at a pinned commit | `tests/test_requirements_contract.py` |
| `Dockerfile` evaluator/export stages install on top of the runtime freeze as pip constraints, so a runtime pin cannot move in them | `Dockerfile` |
| Never install `requirements.txt` into the host venv (its `faiss-gpu-cu12` breaks `faiss-cpu`) | `CLAUDE.md` |

## 3. Inventory by manifest

Legend. "Major" means the pin crosses a major version against the manifest's current floor or inline pin.
"Same" means identical to the `requirements.txt` row. Links in the last column are for major bumps only.

### 3.1 `requirements.txt` (runtime; API image on Python 3.13)

| Package | Current spec | Latest | Proposed pin | Constraint / notes | Breaking-change link |
|---|---|---|---|---|---|
| fastapi | `==0.162.0` | 0.162.0 | `==0.162.0` | already latest; bump only with `make contracts` | |
| uvicorn[standard] | `>=0.41.0` | 0.54.0 | `==0.54.0` | the images already resolve 0.54.0 | |
| python-multipart | none | 0.0.33 | `==0.0.33` | needs Python 3.11 or newer; released 4 days ago | |
| orjson | none | 3.13.0 | `==3.13.0` | license MPL-2.0 AND (Apache-2.0 OR MIT) | |
| pydantic | via fastapi | 2.14.2 | `==2.14.2` (new explicit line) | must equal the tested fastapi pairing; test file already pins it | |
| pydantic-settings | none | 2.16.2 | `==2.16.2` | | |
| ultralytics | `==8.4.197` | 8.4.202 | `==8.4.197` | held: moves in B3 | |
| tritonclient[all] | none | 2.74.0 | `==2.74.0` | client versus Triton 26.06 server skew is tolerated by the KServe v2 protocol; released 2 days ago | |
| onnx | `>=1.12.0,<1.22` | 1.23.2 | `==1.21.0` | held by modelopt `~=1.21.0` | |
| onnxruntime-gpu | `>=1.24.2,<1.25` | 1.31.1 | `==1.30.0` | held by modelopt `<1.31`. The 1.30 wheel is a CUDA 13 build (has `cuda` and `cudnn` extras); GPU provider loading needs the smoke test | |
| torch | none | 2.15.0 | `==2.15.0` | PyPI default wheel uses CUDA 13.2 libraries | |
| torchvision | none | 0.29.2 | `==0.29.2` | pair with torch 2.15.0 | |
| opencv-python | none | 5.0.0.93 | `==5.0.0.93` | Major (4 to 5). Fallback `4.14.0.94` | https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration |
| simplejpeg | none | 1.9.0 | `==1.9.0` | | |
| numpy | none | 2.5.4 | `==2.5.4` | needs Python 3.12 or newer; OpenCV 5 needs numpy 2 | |
| scipy | none | 1.18.1 | `==1.18.1` | | |
| requests | none | 2.34.2 | `==2.34.2` | | |
| aiohttp | none | 3.14.5 | `==3.14.5` | | |
| httpx | none | 0.28.1 | `==0.28.1` | newest release is from 2024-12; `huggingface-hub` 2 uses its own `httpx2`, so this stays | |
| httpcore | none | 1.0.9 | `==1.0.9` | imported by `vlm_pinning.py` | |
| boto3 | none | 1.43.123 | `==1.43.123` | must equal botocore | |
| botocore | none | 1.43.123 | `==1.43.123` | must equal boto3 | |
| matplotlib | none | 3.11.2 | `==3.11.2` | | |
| seaborn | none | 0.13.2 | `==0.13.2` | last release 2024-01; no license metadata (BSD-3-Clause upstream) | |
| pandas | none | 3.1.0 | `==3.1.0` | Major (the trainer floor is 2.2). In `src/` only `ultralytics_patches/end2end_export.py` imports it | https://pandas.pydata.org/docs/whatsnew/v3.0.0.html |
| psutil | none | 7.2.2 | `==7.2.2` | | |
| tqdm | none | 4.70.1 | `==4.70.1` | MPL-2.0 AND MIT | |
| tabulate | none | 0.10.0 | `==0.10.0` | | |
| pyyaml | none | 6.0.3 | `==6.0.3` | | |
| cachetools | `>=7.2` | 7.2.8 | `==7.2.8` | | |
| structlog | `>=24.1.0` | 26.1.0 | `==26.1.0` | calendar-versioned; `.venv` already runs it | |
| tenacity | `>=8.2` | 9.2.1 | `==9.2.1` | `.venv` already runs it | |
| prometheus-client | `>=0.20` | 0.26.0 | `==0.26.0` | | |
| opensearch-py | `>=2.3.0` | 3.2.0 | `==3.2.0` | Major (2 to 3), but `.venv` already runs 3.2.0 with the offline suite. Test file uses `opensearch-py[async]` (extra differs, version identical) | https://github.com/opensearch-project/opensearch-py/blob/main/CHANGELOG.md |
| transformers | `>=4.30.0` | 5.22.1 | `==5.22.1` | needs Python 3.11 or newer; accepts `huggingface-hub` below 3 and `tokenizers` 0.23 | https://github.com/huggingface/transformers/releases/tag/v5.0.0 (v5 already in `.venv`) |
| timm | none | 1.0.31 | `==1.0.31` | open-clip needs `timm>=1.0.17` | |
| huggingface-hub | none | 2.4.0 | `==2.4.0` | Major (1 to 2). The HTTP stack moves to `httpx2`/`httpcore2`; APIs deprecated in 1.x are removed. Used by `export/download_pe_weights.py` and the segmenter; Dependabot ignore rule must go | https://huggingface.co/docs/huggingface_hub/main/en/concepts/migration_v2 |
| ftfy | none | 6.3.1 | `==6.3.1` | | |
| open-clip-torch | `>=3.2.0` | 3.3.0 | `==3.3.0` | 3.2.0 or newer needed for native MobileCLIP2-S2 (F-13b) | |
| cupy-cuda12x | `>=14.2,<15` | 14.2.0 | `==14.2.0` | | |
| faiss-gpu-cu12 | `>=1.10.0` | 1.14.1.post1 | `==1.14.1.post1` | Python `<3.14`, `numpy>=2,<3`; CUDA 12 only | |
| imohash | none | 1.1.0 | `==1.1.0` | last release 2024-09 | |
| einops | none | 0.8.2 | `==0.8.2` | | |
| regex | none | 2026.9.29 | `==2026.9.29` | | |
| scikit-learn | none | 1.9.1 | `==1.9.1` | | |
| umap-learn | none | 0.5.12 | `==0.5.12` | | |
| hdbscan | none | 0.8.44 | `==0.8.44` | | |
| joblib | none | 1.6.0 | `==1.6.0` | | |
| docker | `>=7,<8` | 7.2.0 | `==7.2.0` | | |
| pillow | none (transitive) | 13.0.0 | `==13.0.0` (new explicit line) | ENV-11: imported directly, declared nowhere. License now reported as MIT-CMU (the old HPND text, a rename only) | https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html |

### 3.2 `requirements-test.txt` (CPU test set; Python 3.12, CPU torch index)

Identical pins to section 3.1 for: fastapi, pydantic, uvicorn[standard], python-multipart, orjson,
pydantic-settings, einops, torch (CPU wheel `2.15.0+cpu`), torchvision (`0.29.2+cpu`), ultralytics,
transformers, opencv-python, simplejpeg, numpy, scipy, pyyaml, requests, cachetools, structlog,
tritonclient, prometheus-client, tenacity, httpx, httpcore, boto3, botocore, scikit-learn, onnx,
faiss (see below), umap-learn, hdbscan, joblib, imohash, docker, pillow. The full set resolves with pip
(158 distributions).

| Package | Current spec | Latest | Proposed pin | Constraint / notes | Link |
|---|---|---|---|---|---|
| opensearch-py[async] | `>=2.3.0` | 3.2.0 | `==3.2.0` | extra differs from runtime on purpose (pulls aiohttp) | |
| tritonclient[grpc,http] | none | 2.74.0 | `==2.74.0` | extras differ from `[all]`, version identical | |
| pyclipper | `==1.4.0` | 1.4.0 | `==1.4.0` | already latest; also in `Dockerfile.triton` | |
| pytest | `>=8.0.0` | 9.2.0 | `==9.2.0` | Major. Also in `pyproject.toml` dev and the e2e file (already 9.2.0) | https://docs.pytest.org/en/stable/changelog.html |
| pytest-cov | `>=5.0.0` | 7.1.0 | `==7.1.0` | Major | https://pytest-cov.readthedocs.io/en/latest/changelog.html |
| pytest-asyncio | `>=0.24.0` | 1.4.1.post1 | `==1.4.1.post1` | Major (0.x to 1.x); `.venv` already runs 1.4.0 | https://pytest-asyncio.readthedocs.io/en/stable/reference/changelog.html |
| pytest-timeout | `>=2.4.0` | 2.4.0 | `==2.4.0` | already latest | |
| pytest-xdist | not listed (CONTRIBUTING installs it) | 3.8.0 | `==3.8.0` | add to the file so CI and CONTRIBUTING stop installing it ad hoc; same pin as the e2e file | |
| mypy | `==2.4.1` | 2.4.1 | `==2.4.1` | already latest | |
| ruff | `==0.17.3` | 0.17.3 | `==0.17.3` | already latest; must equal the `.pre-commit-config.yaml` rev | |
| vulture | `==2.16` | 2.16 | `==2.16` | already latest | |
| types-PyYAML | none | 6.0.12.20261103 | `==6.0.12.20261103` | | |
| types-requests | none | 2.33.0.20261030 | `==2.33.0.20261030` | | |
| types-tabulate | none | 0.10.0.20261006 | `==0.10.0.20261006` | | |
| types-tqdm | none | 4.70.0.20261102 | `==4.70.0.20261102` | | |
| onnxconverter-common | none | 1.16.0 | `==1.16.0` | modelopt wants `~=1.16.0` | |
| onnxruntime (CPU) | none | 1.31.1 | `==1.30.0` | held to the runtime pin so tests run the same ORT | |
| faiss-cpu | none | 1.15.1 | `==1.15.1` | intentionally differs from `faiss-gpu-cu12` (never both in one venv); a CPU build, so no identical-pin rule | |
| pycocotools | none | 2.0.11 | `==2.0.11` | | |

### 3.3 `requirements-export.txt` and the `export` extra (engine-build toolchain, on top of the runtime freeze)

| Package | Current spec | Latest | Proposed pin | Constraint / notes | Link |
|---|---|---|---|---|---|
| tensorrt-cu13 | `==11.1.0.106` | 11.4.0.106 | `==11.1.0.106` | held: Triton image TensorRT. NVIDIA proprietary license (not OSI) | |
| nvidia-modelopt[onnx] | none | 0.48.0 | `==0.48.0` | sets the onnx / onnxruntime-gpu holds above; Python `<3.15`; pulls `cupy-cuda13x` | |
| onnx-graphsurgeon | none | 0.6.1 | `==0.6.1` | modelopt wants `>=0.6.1` | |
| onnxsim | `>=0.4.33` | 0.8.1 | `==0.8.1` | Python 3.11 or newer | |
| onnxslim | `>=0.1.71` | 0.1.99 | `==0.1.99` | modelopt wants `>=0.1.78`; released today | |
| onnxscript | none | 0.7.2 | `==0.7.2` | | |

The combined runtime plus export set resolves on Python 3.13 (tensorrt-cu13-libs and -bindings 11.1.0.106,
polygraphy 0.53.6, lief 1.0.0, onnx-ir 1.1.0 come in transitively). `requirements-export.txt` and the
`export` extra must stay set-equal (existing test).

### 3.4 `requirements-export-y11.txt` (YOLO11 export venv `/opt/venv-y11`, Python 3.13)

| Package | Current spec | Latest | Proposed pin | Constraint / notes | Link |
|---|---|---|---|---|---|
| ultralytics | `==8.3.253` | 8.4.202 | `==8.3.253` | held (last 8.3.x) | |
| torch / torchvision | none (CPU index) | 2.15.0 / 0.29.2 | `==2.15.0` / `==0.29.2` | CPU wheels; same version as the other manifests | |
| onnx | `>=1.12.0,<1.22` | 1.23.2 | `==1.21.0` | modelopt | |
| onnxslim, onnxsim, onnx-graphsurgeon | floors | 0.1.99, 0.8.1, 0.6.1 | `==0.1.99`, `==0.8.1`, `==0.6.1` | same as 3.3 | |
| nvidia-modelopt[onnx] | none | 0.48.0 | `==0.48.0` | | |
| tensorrt-cu13 | `==11.1.0.106` | 11.4.0.106 | `==11.1.0.106` | held | |
| pyyaml | none | 6.0.3 | `==6.0.3` | | |
| opencv-python-headless | none | 5.0.0.93 | `==5.0.0.93` | Major; the venv also gets `opencv-python` through Ultralytics; both must be the same version | OpenCV migration guide above |

### 3.5 `docker/trainer/requirements.txt` (trainer image, Python 3.12)

| Package | Current spec | Latest | Proposed pin | Constraint / notes | Link |
|---|---|---|---|---|---|
| torch | `>=2.6,<3` | 2.15.0 | `==2.15.0` | and change `TORCH_CUDA_INDEX` default from `cu128` to `cu132` (see finding 2) | |
| torchvision | `>=0.21,<1` | 0.29.2 | `==0.29.2` | | |
| numpy | `>=1.26,<3` | 2.5.4 | `==2.5.4` | | |
| opencv-python-headless | `>=4.10,<6` | 5.0.0.93 | `==5.0.0.93` | Major | OpenCV migration guide above |
| Pillow | `>=10.4` | 13.0.0 | `==13.0.0` | | |
| PyYAML | `>=6.0` | 6.0.3 | `==6.0.3` | | |
| requests | `>=2.32` | 2.34.2 | `==2.34.2` | | |
| scipy | `>=1.13` | 1.18.1 | `==1.18.1` | | |
| pandas | `>=2.2` | 3.1.0 | `==3.1.0` | Major; Ultralytics and MLflow both use it | pandas 3.0 notes above |
| matplotlib | `>=3.9` | 3.11.2 | `==3.11.2` | | |
| psutil | `>=6.0` | 7.2.2 | `==7.2.2` | | |
| tqdm | `>=4.66` | 4.70.1 | `==4.70.1` | | |
| onnx | `>=1.16,<2` | 1.23.2 | `==1.21.0` | kept equal to the other manifests per the owner's rule (the trainer alone could take 1.23.2; revisit when modelopt allows) | |
| onnxslim | `>=0.1.71` | 0.1.99 | `==0.1.99` | | |
| onnxruntime-gpu | `>=1.19` | 1.31.1 | `==1.30.0` | same as above | |
| albumentations | `>=2.0,<3` | 2.0.8 | `==2.0.8` | already latest in its range | |
| mlflow | `>=3.17,<4` | 3.18.0 | `==3.18.0` | lockstep: bump `MLFLOW_IMAGE` in `docker-compose.yml` from `v3.17.0` to `v3.18.0` (pin by digest) in the same commit; fallback is `mlflow==3.17.2` with no server change. mlflow 3.18.0 resolves `SQLAlchemy` 2.1.8 and `alembic` 1.20.1 | |
| pynvml | `>=11.5` | 13.0.1 | `==13.0.1` | deprecated upstream in favour of `nvidia-ml-py` (13.615.71); keep unless the code is changed | |
| structlog | `>=24.1` | 26.1.0 | `==26.1.0` | | |
| ultralytics | git SHA `8f5d355c...` (build arg) | 8.4.202 | unchanged | not a pip pin; moves with the git SHA to the tag in B3 | |

The CUDA index default is a Dockerfile edit (`docker/trainer/Dockerfile`, `ARG TORCH_CUDA_INDEX`), not a
requirements edit. Confirm the host driver supports CUDA 13.2 wheels before changing it.

### 3.6 `scripts/curation/bakeoff/requirements.txt` and the evaluator image

| Package | Current spec | Latest | Proposed pin | Constraint / notes | Link |
|---|---|---|---|---|---|
| pycocotools | `>=2.0.7` | 2.0.11 | `==2.0.11` | | |
| onnxruntime-gpu | unpinned | 1.31.1 | `==1.30.0` | equals runtime (installed under the runtime freeze) | |
| onnx | `>=1.12.0,<2.0.0` | 1.23.2 | `==1.21.0` | | |
| onnxslim | `>=0.1.71` | 0.1.99 | `==0.1.99` | | |
| tritonclient[all] | `>=2.40` | 2.74.0 | `==2.74.0` | | |
| mlflow | bare `mlflow` inline in `Dockerfile` (`builder-eval`) | 3.18.0 | `==3.18.0`, moved into this file | the same version as the trainer and the server image. The resolver run of this file under the runtime freeze succeeds | |

### 3.7 `pyproject.toml`

`[project.dependencies]` carries the same names and the same pins as section 3.1 (one change per package,
same value), plus the two new lines `pydantic==2.14.2` and `Pillow==13.0.0`. Rows that differ in form:

| Item | Current spec | Latest | Proposed pin | Notes |
|---|---|---|---|---|
| `fastapi` | bare | 0.162.0 | `==0.162.0` | ENV-11: bare today while both requirements files pin it |
| `[export]` extra | see 3.3 | | same as 3.3 | set-equal test stays |
| `[pe]` extra | git commit pin | | unchanged | `--no-deps` install |
| `[dev]` pre-commit | `>=4.0.0` | 4.7.0 | `==4.7.0` | |
| `[dev]` ruff, mypy, vulture | exact | same | unchanged | already latest |
| `[dev]` bandit[toml] | `>=1.9.4` | 1.9.4 | `==1.9.4` | |
| `[dev]` pytest, pytest-cov, pytest-asyncio, pytest-timeout | floors | see 3.2 | the 3.2 pins | identical to the test file |
| `[build-system]` setuptools, wheel | `setuptools>=68.0`, `wheel` | 84.2.1, 0.48.0 | `setuptools==84.2.1`, `wheel==0.48.0` | build isolation downloads exactly these; risk is low, revert to a floor if a build backend regression appears |

### 3.8 Inline pins in the Dockerfiles

| File | Item | Current | Latest | Proposed | Notes |
|---|---|---|---|---|---|
| `Dockerfile.triton` | torch / torchvision (CUDA 13 index) | `2.12.1` / `0.27.1` from `cu130` | 2.15.0 / 0.29.2 (`cu132`); the `cu130` index stops at 2.14.1 / 0.29.1 | `==2.15.0` / `==0.29.2`, `--index-url .../cu132` | base is CUDA 13.3; fallback is `2.14.1` / `0.29.1` on `cu130` |
| `Dockerfile.triton` | opencv-python-headless | `==4.13.0.92` | 5.0.0.93 | `==5.0.0.93` | Major. Needs numpy 2 or newer: read the numpy version of the 26.06 base first, because pip will upgrade it and Triton Python models import it |
| `Dockerfile.triton` | pyclipper | `==1.4.0` | 1.4.0 | unchanged | |
| `Dockerfile.triton` | cupy-cuda13x | `==13.6.0` (what the base ships) | 14.2.0 | `==14.2.0` | Major (13 to 14: NumPy 2 semantics; `broadcast_arrays` and `meshgrid` return tuples). Matches the API image's `cupy-cuda12x` 14.2.0, so the GPU decode kernels run on one CuPy generation. Touches `gpu_image_prep`; needs the GPU decode tests |
| `Dockerfile.triton` | starlette, anyio, certifi, filelock, virtualenv | floors | 1.10.0, 4.16.0, 2026.10.23, 4.2.8, 21.15.5 | exact `==` | CVE floors become exact pins; the last two were released today |
| `docker/hardened/triton/Dockerfile` | starlette, fastapi | `>=1.3.1`, `>=0.139.0` | 1.10.0, 0.162.0 | `==1.10.0`, `==0.162.0` | fastapi 0.162.0 pairs with starlette 1.x |
| `docker/segmenter/Dockerfile` | fastapi | `==0.162.0` | same | unchanged | |
| | uvicorn[standard] | `==0.32.0` | 0.54.0 | `==0.54.0` | stale pin |
| | pydantic | `>=2.7,<3` | 2.14.2 | `==2.14.2` | |
| | numpy | `>=1.26,<2.0` | 2.5.4 | `==1.26.4` | held by SAM 3 |
| | pillow | `>=13.0.0` | 13.0.0 | `==13.0.0` | |
| | certifi, filelock, dnspython | floors | 2026.10.23, 4.2.8, 2.9.2 | exact | |
| | opencv-python-headless | `>=4.10` | 5.0.0.93 | `==4.11.0.86` | held: numpy 1.x |
| | scikit-image | `>=0.24` | 0.26.0 | `==0.26.0` | |
| | huggingface-hub | `>=0.24` | 2.4.0 | `==2.4.0` | resolves with SAM 3's `timm` 1.0.31; the segmenter does download weights, so run its start-up test |
| | einops, pycocotools | floors | 0.8.2, 2.0.11 | exact | |
| | torch | base image `2.14.1` | 2.15.0 | unchanged | base image tag, not a pip pin |

Recommendation for the inline pins: move each into a requirements file the Dockerfile copies
(`docker/segmenter/requirements.txt`, `docker/triton/requirements.txt`, mlflow into the bakeoff file), so
Dependabot reads them and the consistency test in section 5 can compare them. The Dockerfile edit is one
`COPY` plus `pip install -r`.

### 3.9 Other requirements files found

| File | Package | Current | Latest | Proposed | Notes |
|---|---|---|---|---|---|
| `frontend/e2e/requirements.txt` | pytest | `==9.2.0` | 9.2.0 | unchanged | already latest; equal to the 3.2 pin |
| | pytest-playwright | `==0.10.0` | 0.10.0 | unchanged | |
| | playwright | `==1.64.1` | 1.64.1 | unchanged | |
| | pytest-xdist | `==3.8.0` | 3.8.0 | unchanged | |
| `scripts/research/requirements.txt` | igraph, leidenalg | none | 1.0.0, 0.12.0 | `==1.0.0`, `==0.12.0` | research only. GPL-licensed (leidenalg GPL-3.0-or-later, igraph GPL): not shipped, install outside the product; do not add to any product manifest |
| `scripts/research/fine_grained_requirements.txt` | torch, torchvision, timm, open_clip_torch, transformers, safetensors, numpy, scipy, scikit-learn, pillow, pytest, ultralytics | none | as above; safetensors 0.8.0 | same pins as 3.1 / 3.2; ultralytics `==8.4.197` | plus python-igraph `==1.0.0` (GPL), leidenalg `==0.12.0` |
| the one example requirements file under `examples/bakeoff/` | open-image-models | `>=0.5` | 0.6.0 | `==0.6.0` | MIT |

**Research venvs (added by the research integration pass, approved, RD-7).** Each research topic owns
its exact pins in its own tracked file, never in a product manifest, with a committed `pip freeze`
lock beside it: `scripts/research/gpu_pipeline/requirements-gpu-research.txt` (its `tensorrt-cu13`
and `cupy-cuda13x` move with the Triton image), `scripts/research/clustering/requirements-cpu.txt`
and `requirements-gpu.txt` (RAPIDS 26.10 cu13), `scripts/research/embedding_search/requirements.txt`,
`scripts/research/near_dup/requirements.txt`, `scripts/research/spatial/requirements.txt`,
`scripts/research/identity_text/requirements.txt`, `scripts/research/auto_label/requirements.txt`.
The existing unpinned `scripts/research/requirements.txt` and `fine_grained_requirements.txt` take
the pins of this table. A test (`tests/test_requirements_contract.py`) asserts every research file is
`==`-pinned and that no GPL package (igraph, leidenalg) appears in a product manifest.

**Product packages the research seams may add** (each only when its lane lands, with its licence in
ATTRIBUTION): `nvidia-ml-py` in the GPU worker images (1B-8, NVML inventory; skip if already
present); `imohash` leaves the product manifests if RD-43 replaces it with SHA-256 (ND-8);
`pdqhash==0.2.8` only on a `nd:P2` pass (ND-E); `rapidfuzz` only if the plate aligned vote ships
(W5e). Exact versions are taken at the lane's pin time, the newest that installs.

## 4. New packages and images for the control plane

Another plan designs the control plane (Celery workers, a database for job state). This section only
supplies versions and licenses. All resolve together with the runtime set on Python 3.13.

| Package | Latest | Proposed | License | Notes |
|---|---|---|---|---|
| celery (extra `redis`) | 5.6.3 (2026-03-26) | `celery[redis]==5.6.3` | BSD-3-Clause | the `redis` extra pulls `kombu[redis]`; needs billiard 4.3.2, vine 5.1.0, click 8.5.0 |
| kombu | 5.7.0 | `==5.6.2` (held; owner decision 2026-11-04 after review R-10) | BSD-3-Clause | 5.6.2 (2025-12-29) accepts `redis>=4.5.2,<6.5` minus 4.5.5 and 5.0.2. 5.7.0 (2026-11-03) would accept `redis>=6.4,<9` but its `redis` extra pins `packaging==26.3`, and no Celery release has been tested against it |
| redis (redis-py) | 8.1.0 | `==6.4.0` (held with kombu) | MIT | client for the Valkey server (Valkey speaks the Redis protocol); the newest release inside kombu 5.6.2's range; the 8.x line waits for kombu 5.7. Async pub/sub (used by the event wake-up) is in 6.4 |
| SQLAlchemy | 2.1.8 | `==2.1.8` | MIT | needs Python 3.11 or newer. Migration notes: https://docs.sqlalchemy.org/en/21/changelog/migration_21.html (read before writing models; MLflow 3.18.0 already resolves it in the trainer) |
| Alembic | 1.20.1 | `==1.20.1` | MIT | resolves with SQLAlchemy 2.1.8 (checked together with MLflow) |
| psycopg | 3.3.8 | `psycopg[c,pool]==3.3.8` in images; `psycopg[binary,pool]==3.3.8` in the dev and test venvs | LGPL-3.0-only | Recommendation: the C build in images. The psycopg documentation advises building from source for production, and the multi-stage API builder already has compilers; the runtime stage then needs `libpq5` only. Review fix R-27: the trainer, export and evaluator images also import `src/db`, so they get the same builder step and `libpq5` (B2 lanes 5A and 5B). `binary` needs no system libpq and is fine for tests. The version is identical; the extra differs by environment, like `opensearch-py[async]`. Use SQLAlchemy URLs `postgresql+psycopg://` (async works natively) |
| psycopg-pool | 3.3.3 | `==3.3.3` (the `pool` extra) | LGPL-3.0-only | LGPL is OSI-approved copyleft and is dynamically linked, which is compatible with this project's AGPL; record it in ATTRIBUTION.md |
| testcontainers | 4.15.0 | `testcontainers[postgres,redis]==4.15.0` in the test set only | Apache-2.0 | has `postgres`, `redis` and `valkey` extras; the `redis` module starts any image, so a Valkey image works. Needs Docker at test time, so it belongs to the live or integration marker, not the offline suite |
| flower | 2.2.0 | `==2.2.0`, only if Flower runs | BSD | needs `tornado<7,>=6.5.7`; optional (ENV-14) |

Replaces the sketch in section 7.1 of `environment_and_setup.md`, which names `redis:8.2.x` as the
server; use Valkey (below). A Redis 8 server is available under RSALv2/SSPL/AGPLv3 terms; Valkey is BSD-3-Clause
and avoids the question.

Container images (digest-pinned; refresh the digests at implementation time):

| Image | Tag | Digest | Reason |
|---|---|---|---|
| `postgres` (Docker official image) | `18.6-trixie` | `sha256:3a4264520e121176619f54ff867d12774bc81e95e7b84db58d63bccdb6144b81` | 18 is the newest stable major (19 is at rc1 only). Debian trixie rather than alpine: PostgreSQL's text sort order depends on the C library, and the Debian glibc image is the best-supported one (the docs warn musl differs); the repo's Python images are already trixie, so one OS family to scan. Updated 2026-10-31. Mount the data volume at `/var/lib/postgresql` (PG 18 images changed `PGDATA` to a versioned subdirectory) |
| `valkey/valkey` | `9.1.3-alpine3.24` | `sha256:bf09d58e96d6b4524a15396b65b242bbe18beebbcc15a04d8ee75ddac62fae0b` | latest stable 9.x (9.2.0 is rc1; 9.0.7 is the older line). Alpine is fine here because Valkey has no collation concern, and the image is small. Updated 2026-10-27 |

## 5. Shared-package consistency rules

Identical pin (version string, extras may differ) required across:

| Rule | Manifests | Enforced today by | Gap and fix |
|---|---|---|---|
| Runtime packages in `requirements.txt` equal `pyproject.toml` `[project.dependencies]` and equal `requirements-test.txt` where the package appears | `requirements.txt`, `pyproject.toml`, `requirements-test.txt` | names only (`test_pyproject_and_requirements_txt_agree`); exact version only for `ultralytics` (`test_ultralytics_is_one_exact_version_everywhere`) and `fastapi` (`test_the_image_pins_fastapi_as_the_tests_do`) | Generalize per ENV-11: every `==` specifier equal across the three, with a small allowlist for `faiss-cpu` versus `faiss-gpu-cu12`, `onnxruntime` versus `onnxruntime-gpu`, and `opencv-python` versus `-headless` (which must still carry the same version). Add a check that no runtime line is a floor or unpinned (owner rule) |
| Export pins equal | `requirements-export.txt`, `[export]` extra | set-equality test | extend to versions; add `requirements-export-y11.txt` for the shared ones except `ultralytics` |
| `tensorrt-cu13` equals `TRT_VERSION` | export file, y11 file, extra, `Dockerfile.triton` | `test_tensorrt_pin_matches_the_triton_image_everywhere` | none |
| Dev tools equal | `pyproject.toml` `[dev]`, `requirements-test.txt` | none | add to the generalized test; ruff also equals the pre-commit rev |
| Image-side pins equal the manifests | trainer, bakeoff, Triton, hardened Triton, segmenter files versus the runtime for: torch, torchvision, onnx, onnxruntime-gpu, onnxslim, tritonclient, opencv (except segmenter), numpy (except segmenter), pillow, mlflow (trainer = bakeoff = server image tag) | none | after section 3.8 moves inline pins into files, the same test reads them. Declare the two documented exceptions (segmenter numpy and opencv) in the test with the reason |
| CPU install commands name the torch pair | `CONTRIBUTING.md`, `.github/workflows/ci.yml` (3 places) | none | they install unpinned `torch torchvision` from the CPU index first. After pinning `requirements-test.txt` this drifts: when the CPU index gets a newer torch, pip would replace it with the PyPI CUDA wheel. Change the commands to `torch==2.15.0 torchvision==0.29.2` and add the version to the generalized test |
| MLflow client equals server | `docker/trainer/requirements.txt`, bakeoff file, `MLFLOW_IMAGE` in `docker-compose.yml` | none | add to the same test |

Dependabot (`.github/dependabot.yml`): with exact pins its `increase-if-necessary` comment no longer
describes the behaviour (exact pins always get PRs). Update the comment, remove the `huggingface-hub` major
ignore, change the `onnxruntime-gpu` ignore to `>=1.31.0` with the modelopt reason, keep the `onnx >=1.22`,
`ultralytics` and `tensorrt-cu13` rules, and add the `/docker/segmenter` and `/docker/triton` directories
once their pins are in files. Group torch with torchvision so they arrive in one PR.

## 6. Verification plan for the implementer

Run from the repository root, using the repository's venv binaries directly (never `source`, never install
`requirements.txt` into the host venv).

CPU test venv (the supported path from CONTRIBUTING.md, with the pair pinned):

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2
.venv/bin/pip install -r requirements-test.txt
.venv/bin/pip check
.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live and not families'
.venv/bin/pre-commit run --all-files
```

Resolve the sets that cannot install on the host (no GPU wheels, Python 3.13), without installing, in a
throwaway environment outside the repository:

```bash
# runtime set, API image target
uv pip compile requirements.txt --python-version 3.13 --python-platform x86_64-manylinux_2_28 -o /dev/null
# export set on top of the runtime set
cat requirements.txt requirements-export.txt | uv pip compile - --python-version 3.13 \
  --python-platform x86_64-manylinux_2_28 -o /dev/null
# YOLO11 venv, trainer (3.12), segmenter extras (3.12), bakeoff under the runtime freeze
uv pip compile requirements-export-y11.txt --python-version 3.13 --python-platform x86_64-manylinux_2_28 \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match -o /dev/null
uv pip compile docker/trainer/requirements.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28 -o /dev/null
```

(`uv` is used only as a resolver here; no lock file is committed.) `pip install --dry-run --report out.json
-r <file>` in a Python 3.12 venv is the equivalent for the 3.12 sets.

Contracts and API:

```bash
make contracts          # only needed if fastapi or pydantic changed; commit contracts/ in the same commit
make contracts-check
.venv/bin/python scripts/docs/check_docs_vs_code.py
.venv/bin/python scripts/codegen/check_naming_leaks.py
```

Images (use a compose project name `-p <name>` for anything that starts containers; builds below do not):

```bash
docker build --target api       -t openprocessor-scan:api       -f Dockerfile .
docker build --target export    -t openprocessor-scan:export    -f Dockerfile .
docker build --target evaluator -t openprocessor-scan:evaluator -f Dockerfile .
docker build -t openprocessor-scan:trainer   -f docker/trainer/Dockerfile .
docker build -t openprocessor-scan:segmenter -f docker/segmenter/Dockerfile .
docker build -t openprocessor-scan:triton    -f Dockerfile.triton .
```

Then in each image: `pip check` (the runtime images have no pip, so run it in the builder stage or with
`python -m pip check` from the `builder` target), and the smoke imports: API image `import cv2, faiss,
cupy, onnxruntime, torch, transformers, open_clip, huggingface_hub` and
`onnxruntime.get_available_providers()` on a GPU host (CUDA provider present); export image the same plus
`import cupy` after both cupy wheels are installed, `import tensorrt`, and one ModelOpt AutoCast export;
Triton image `python3 -c "import torch, cv2, cupy, numpy; print(numpy.__version__)"` and the GPU decode
tests. Per the test budget, GPU and image work is batched into one window, not run per commit.

Security scan, the same commands CI runs (see section 6 of `environment_and_setup.md`):

```bash
trivy fs --severity CRITICAL,HIGH --ignore-unfixed --exit-code 1 --ignorefile scripts/release/trivy-allowlist.txt .
trivy image --severity CRITICAL,HIGH --ignore-unfixed --exit-code 1 \
  --ignorefile scripts/release/trivy-allowlist.txt --timeout 30m openprocessor-scan:api
```

(repeat per image). Record `pip freeze` of every built image in the release evidence (ENV-13); the freeze
is the lock substitute the owner chose.

## 7. Suggested bump commits, in bisect order

Each line is one commit (or a pair, test first). Run the targeted check named in section 6 after each; the
full sweep runs once at the end of the series.

1. `test`: generalize the pin test (ENV-11) and add pillow and pydantic declarations. Red first: it fails on
   today's floors and on bare `fastapi` in `pyproject.toml`.
2. Dev and test tooling only: pytest family, pytest-xdist, pre-commit, bandit, types-*, CONTRIBUTING and CI
   torch-pair commands. No runtime code touched.
3. Pure-Python leaf libraries with no ABI: requests, tenacity, structlog, cachetools, tabulate, tqdm, pyyaml,
   psutil, docker, prometheus-client, einops, regex, joblib, imohash, ftfy, httpx, httpcore.
4. Web layer: uvicorn, python-multipart, orjson, pydantic-settings, pydantic explicit, pillow. Then
   `make contracts-check` (and `make contracts` if the schema moved).
5. Cloud and search clients: boto3 and botocore (together), aiohttp, opensearch-py 3.2.0, tritonclient.
6. Numeric stack: numpy, scipy, scikit-learn, umap-learn, hdbscan, pandas 3.1 (alone), matplotlib, seaborn.
7. OpenCV 5 alone (api, test, y11, trainer, Triton), then rerun the preprocessing parity tests; the
   nearest-neighbor and warp changes can move pixel-exact baselines.
8. Hugging Face stack alone: huggingface-hub 2, transformers, timm, open-clip-torch; plus the Dependabot
   ignore removal.
9. torch 2.15.0 with torchvision 0.29.2 across api, test, y11, trainer (index to `cu132`).
10. GPU libraries: onnx 1.21.0, onnxruntime(-gpu) 1.30.0, cupy 14.2.0, faiss, nvidia-modelopt and the rest
    of the export toolchain; tensorrt unchanged.
11. Image-only pins: move the Triton, hardened-Triton and segmenter inline pins into files and bump them
    (Triton torch on `cu132`, cupy-cuda13x 14, security floors to exact); one commit per image.
12. MLflow: trainer, bakeoff and `MLFLOW_IMAGE` together.
13. `ci`/docs: Dependabot config, update section 7.1 of `environment_and_setup.md` (which shows stale
    fastapi and pydantic values), ATTRIBUTION for the new licenses.

The control-plane packages and images (section 4) are added by the B2 queue work, not in this series, so
that a Celery or database problem never blocks a dependency bump.

### 7.1 Where the series runs (owner direction 3 of 2026-11-04; [00_master_plan.md](00_master_plan.md))

The owner's rule: the bumps that can change model output ride with the B3 GPU gate, so the parity and
accuracy check runs once for both; the others go where this bisect order puts them. The series is
therefore cut in two lanes, keeping the order above inside each:

| Part | Wave and lane | Commits | Why there |
|---|---|---|---|
| A | **W2.0, its own PR to `main` before W2.1** (owner decision 2026-11-04 after review R-16), lane `deps/pins-part-a` | 1, 2, 3, 4 without `pillow`, 5 | nothing here decodes images or computes model inputs; one CPU offline sweep closes it, so a dependency regression never blocks or muddies the B2 PR |
| B | W3, lane `deps/pins-part-b` (B3 track `deps`, first, before `stagea`) | 6, 7 with `pillow` and `simplejpeg` added, 8, 9, 10, 11, then 12 and 13 | 6, 7, 9, 10 and 11 are the owner's list (numpy and pandas with the numeric stack, OpenCV 5, torch, the onnx set, the image pins with CuPy 14); Pillow and simplejpeg decode images, so by the same rule they join commit 7 (owner decision MQ-4, 2026-11-04: Pillow, simplejpeg, the Hugging Face stack and MLflow are members of part B); 8 (Hugging Face stack) sits inside that run and 12 (MLflow) and 13 (Dependabot, docs) after it in the bisect order; the GPU gate of [04](04_b3_ultralytics_one_version.md) section 6 (step G11) checks them |

Adjustments the split needs (each in the commit named):
- Commit 1 (part A) adds the "no floor, no unpinned runtime line" check with a **shrink-only allowlist**
  of the part B packages, so the test is green at every commit; part B removes each package from the
  allowlist in the commit that pins it, and the last part B commit asserts the allowlist is empty.
- The CPU torch-pair commands in `CONTRIBUTING.md` and `.github/workflows/ci.yml` (listed in commit 2)
  move to commit 9, where the torch pin moves; part A does not touch those two files (B2 lane 1D owns
  them in W2.1, after part A merged).
- `simplejpeg` (in section 3.1, missing from the list above) is pinned in commit 7.
- `requirements-export-y11.txt` is skipped by part B: B3-A2 deletes it in the same wave.
- PR #477 (CI shards) edits `requirements-test.txt` and `pyproject.toml`; it merges before part A starts.
- Part A also re-applies the open frontend Dependabot PR #474 (frontend dev dependencies) in its own
  commit, following the deps-lane rule of [09](reference/github_actions.md) section 8.

## 8. Risk register (largest blast radius first)

| # | Bump | What can break | Mitigation / fallback |
|---|---|---|---|
| 1 | torch 2.15.0 + torchvision 0.29.2 in every GPU image (CUDA 13.2 wheels; the Triton index change `cu130` to `cu132`; trainer index `cu128` to `cu132`) | Driver too old for 13.2 wheels; Python backend models in Triton behave differently; ONNX export graphs change, so engines need a rebuild check; mixed CUDA 12 (`faiss`, `cupy-cuda12x`) and CUDA 13 libraries in the API process (pre-existing) | One GPU window: load every engine, run the export smoke test, compare detection results on the public sample set. Fallback: `2.14.1` / `0.29.1` on the `cu130` index |
| 2 | OpenCV 5.0.0.93 (api, Triton, trainer, y11) | Nearest-neighbor resize now follows Pillow rounding; warp and remap interpolation revised; 1D array semantics; `cv2.ml` and Darknet/Caffe loaders gone. Letterbox and crop preprocessing could shift scores slightly; pixel-exact test baselines need regeneration. Triton image: needs numpy 2 or newer, which the base may not have | Parity tests on CPU first (`.venv` already runs 5.0.0.93 offline); GPU decode comparison. Fallback: `4.14.0.94` everywhere |
| 3 | onnx / onnxruntime-gpu / modelopt / tensorrt set (export and runtime) | ONNX 1.21 versus engines built by the older stack; ORT 1.30 CUDA 13 wheel needs CUDA 13 runtime libraries next to CUDA 12 ones; two cupy wheels in the export image | Keep onnx 1.21.0 / ORT 1.30.0 until modelopt allows more; export one model per family and diff engine outputs; import test of `cupy` in the export image |
| 4 | numpy 2.5.4 across images, with scipy 1.18.1, pandas 3.1.0, scikit-learn 1.9.1 | C-extension ABI in wheels without numpy 2 support (hdbscan, umap, imohash, simplejpeg) | `pip check` plus import of every lazily imported module; the clustering tests already run on numpy 2.5 in `.venv` |
| 5 | huggingface-hub 2.4.0 (and transformers 5.22.1) | New `httpx2` HTTP stack and removed deprecated APIs; proxies or custom CA settings behave differently; segmenter weight download; Dependabot rule said it did not resolve (now it does) | Try the PE weight download script and the segmenter start-up; fallback `huggingface-hub==1.33.0` (transformers 5.22.1 accepts it) |
| 6 | fastapi / pydantic / starlette | Schema or OpenAPI shape drift; `starlette` 1.10.0 pulled in by fastapi | fastapi and pydantic are already at the target pins; only starlette and uvicorn move; `make contracts-check` |
| 7 | mlflow 3.18.0 with server image 3.17.0 | Client and server on different minors; registry REST calls | Bump server image and clients in one commit, or hold both at 3.17.x |
| 8 | cupy-cuda13x 14.2.0 in the Triton image | CuPy 14 NumPy 2 semantics (tuple returns, dtype promotion) in `gpu_image_prep` kernels | Keep `13.6.0` if the GPU decode tests fail; the API image already runs CuPy 14 |
| 9 | pandas 3.1.0 (trainer, runtime) | Default string dtype and copy-on-write change dataframe behavior in Ultralytics' results handling and the bench scripts | Trainer smoke run of a short training; `src/` has one import |
| 10 | pytest 9.2.0, pytest-asyncio 1.4.1, pytest-cov 7.1.0 | Fixture and event-loop changes, coverage option changes | Already running in `.venv` at 9.1.1 / 1.4.0 / 7.1.0 |
| 11 | Freshly released pins (psycopg, SQLAlchemy, kombu, filelock, virtualenv, onnxslim, ultralytics) | A same-week release gets yanked or has a quick regression patch | Re-run the survey the day the series starts; step back one patch rather than hold the series |
| 12 | Licenses | No license change found among packages that differ between the repo test venv and the proposal beyond SPDX renames (Pillow: MIT-CMU; setuptools: MIT). New LGPL-3.0 (psycopg); GPL research-only (igraph, leidenalg) must not enter product manifests; NVIDIA proprietary (tensorrt-cu13, CUDA wheels) and AGPL-3.0 (ultralytics) already present | Update ATTRIBUTION.md with psycopg and psycopg-pool |
