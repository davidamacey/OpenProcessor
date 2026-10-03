# GPU-accelerated clustering (cuML UMAP)

Status: research snapshot (2026-05); measured on a single 49 GB workstation GPU shared with a segmentation model, plus a 12 GB consumer GPU as the "does not fit" reference; numbers will drift.

Implemented in the public repo: partly. Runtime GPU/CPU backend detection, the cuML UMAP path, per-backend UMAP cache slots and cuML kNN graph construction for AHC are implemented in `src/services/curation/clustering/backend.py`, `src/services/curation/clustering/embedding_reduce.py` and `src/services/curation/clustering/methods/ahc.py`. The dedicated compose overlay and `make` targets described in section 6 are **not** in the public repo; they are a proposal. Note that the default clustering method is IVF k-means (see [clustering_methods.md](clustering_methods.md)), and UMAP is no longer a clustering pre-processor by default, so this document matters for the UMAP-based paths and for contributors reusing the GPU backend probe.

## 1. Goal

Reduce the residual-pool recluster from 20-35 minutes (CPU) to single-digit minutes by moving the UMAP dimensionality-reduction step to the GPU, while keeping a silent CPU fallback that produces the same output shape. Targets on a 90k-row pool of 1024-d embeddings:

- Whole recluster in 5-15 minutes (CPU baseline 30-50 minutes); the clustering stage drops from 20-35 minutes to 5-10 minutes.
- A worker without GPU access falls back to sklearn / umap-learn with no code change.
- Removing GPU access mid-run lets the current operation finish; the next run falls back to CPU.
- Cancel takes effect within 30 seconds during the embedding-fetch phase.
- The dashboard never shows `processed=0/0` for more than 5 seconds.

Out of scope: GPU agglomerative clustering, HDBSCAN as a replacement, GPU acceleration of the small per-cluster refine, multi-GPU distribution.

## 2. Library findings

cuML 26.04 on CUDA 13.x installs from the NVIDIA package index (`cuml-cu13==26.4.*`, with a matching `cupy-cuda13x`); the wheels statically link the CUDA runtime, so only the host driver and the NVIDIA container runtime are required.

`cuml.manifold.UMAP` has near-parity with `umap-learn` for the hyperparameters used (`n_components=50`, `n_neighbors=15`, `min_dist=0.0`, `metric='cosine'`, `random_state=42`). Gotchas:

- `build_algo='nn_descent'` (default, fast) is non-deterministic even with `random_state` set. `build_algo='brute_force_knn'` is deterministic but slower and uses about 2x the VRAM.
- VRAM creeps across runs (cuml#4068); call `cupy.get_default_memory_pool().free_all_blocks()` after every fit.
- A pickled cuML UMAP retains its training data on the GPU (cuml#5818), so reloading a cached reducer consumes VRAM proportional to the cached training set.

`cuml.cluster.AgglomerativeClustering` is not usable here for three reasons: only `linkage='single'` exists (complete linkage is needed), there is no `distance_threshold`, and there is no precomputed connectivity. With cosine it also forces pairwise connectivity, a roughly 32 GB allocation at 90k items. Decision: sklearn AHC stays on CPU. cuML is still used to build the kNN connectivity graph that constrains sklearn AHC.

CuPy/cuML kernels release the GIL during stream synchronization, so wrapping GPU work in `asyncio.to_thread` keeps a heartbeat coroutine alive.

VRAM budget on 90k x 1024:

| Build algorithm | Peak VRAM |
|---|---|
| nn-descent UMAP | 2-6 GB |
| brute_force_knn UMAP | 8-15 GB |

## 3. Backend detection

`detect_cluster_backend()` returns a `BackendInfo` (`name` of `gpu` or `cpu`, a detail string, free VRAM in MB, and the chosen UMAP build algorithm). It runs on **every call**, not once per process, because the probe is microseconds and an operator may strip GPU access from a running worker. Logic:

1. `import cuml, cupy`; `ImportError` means CPU.
2. No visible CUDA device means CPU.
3. `memGetInfo()` free VRAM below 3 GB (`MIN_FREE_VRAM_GB_NN_DESCENT`) means CPU.
4. Otherwise GPU. At 12 GB or more free (`MIN_FREE_VRAM_GB_BRUTE_FORCE`) the deterministic `brute_force_knn` build is chosen, else `nn_descent`.

`free_gpu_blocks()` drains the memory pool after each fit; `gpu_used_vram_mb()` feeds peak-VRAM telemetry.

## 4. UMAP cache isolation

`umap.UMAP` and `cuml.manifold.UMAP` are incompatible pickle classes, so each backend has its own cache slot: a `umap_state.joblib` file and `current` state document for CPU, and `umap_state_cuml.joblib` with a `current_cuml` document for GPU. Switching backends forces one refit on the new side (about 30 seconds on GPU).

## 5. Progress and cancel instrumentation (design)

- Pass the job's progress object into the clustering function and add `raise_if_cancelled()` plus a counter update inside the embedding-fetch scroll loop (a scroll page is roughly 1000 documents and 50 ms, so cancel is sub-second).
- Stages that wrap one long synchronous database call (id normalize, finalize) get a sibling ticker coroutine reporting elapsed seconds instead of `0/0`.
- Auto-promote reports a per-cluster counter.
- The job result carries per-stage durations and peak VRAM (GPU runs only), and the dashboard shows a backend chip (`gpu` or `cpu` with library versions).
- The per-cluster refine endpoint runs sklearn AHC inside `asyncio.to_thread` so a large refine cannot block the API event loop.

## 6. Deployment (proposal, not in the public repo)

A compose overlay would give only the clustering worker GPU access (`runtime: nvidia`, one pinned device) and two make targets would recreate the worker with or without that overlay. Because detection is per call, "GPU when available, sklearn fallback" needs no restart when the overlay is removed. cuML adds roughly 1.5 GB to a shared image; splitting a worker-only image is an option if that becomes painful.

## 7. Validation plan

1. GPU path: total runtime at or under 15 minutes on a ~90k pool, candidate clusters written, per-stage progress advancing.
2. CPU fallback: identical output shape (candidate count within 5% of the GPU run), no import errors.
3. GPU access removed mid-run: worker survives, next run uses CPU.
4. Output sanity, CPU vs GPU on the same pool: cluster count within 10%, no single cluster above 30% of items, comparable intra-cluster cosine distance.
5. Cancel within 30 seconds during fetch.
6. Every stage longer than 5 seconds shows non-zero progress.

## 8. Open risks

1. nn-descent is non-deterministic, so cluster boundaries shift more across refits. Refits are rare and candidate clusters are treated as ephemeral.
2. Contention on a GPU shared with another model can OOM a fit; the 3 GB probe falls back to CPU, with no correctness impact.
3. cuml#5818 VRAM retention after reloading a cached reducer; drain blocks after every load and fit.
4. Image size growth from the cuML wheel.
