# Scripts

Utility scripts for setup, deployment, maintenance, image processing,
and the curation subsystem.

## Root-Level Scripts

### setup.sh

One-shot fresh-install script: pulls Docker images, detects your GPU,
selects a profile, exports TensorRT engines, starts services, and runs
smoke tests. See [README.md](../README.md#quick-start).

```bash
./scripts/setup.sh              # interactive
./scripts/setup.sh --yes        # non-interactive, defaults
./scripts/setup.sh --profile=standard --gpu=0 --yes
```

### openprocessor.sh

Day-2 management commands — status, logs, restart — for a running
deployment.

```bash
./scripts/openprocessor.sh status    # Check service health (API/Triton/OpenSearch)
./scripts/openprocessor.sh logs -f   # View live logs
./scripts/openprocessor.sh restart   # Restart all services
./scripts/openprocessor.sh help      # See all commands
```

There is no separate `check_services.sh` — `openprocessor.sh status`
covers that.

### resize_images.py

Batch image resizing utility with multiprocessing support.

```bash
# Resize images to 640px max dimension
python scripts/resize_images.py /path/to/images --size 640

# Custom output directory and worker count
python scripts/resize_images.py /path/to/images --size 1024 --output /path/to/output --workers 16
```

**Features:**
- Maintains aspect ratio
- 100% JPEG quality (no compression artifacts)
- Parallel processing via multiprocessing
- Progress bar with ETA

### docker-build-push.sh / security-scan.sh / export_paddleocr.sh / setup_face_test_data.sh

Build/publish and one-off setup helpers — see each script's own header
comment for usage; they're small and self-documenting.

## `scripts/curation/` — curation subsystem workers and tooling

**Experimental**, ships behind the `curation` Docker Compose profile —
see [`docs/CURATION.md`](../docs/CURATION.md) for the full guide.

| Path | What it is |
|---|---|
| `ingest_walker.py`, `_fast_walk.py` | Parallel bulk-directory ingest: `os.scandir` walker → reader threads → bounded queue → concurrent `POST /curation/ingest/batch`, with a resumable progress file. |
| `vlm_worker.py` | Long-lived VLM labeling/verification loop (`curation-vlm-worker` service). |
| `auto_label_worker.py` | Drives the `/curation/pipeline/auto_label` protocol as a long-lived process (`curation-auto-label-worker` service). |
| `cluster_refresh_daemon.py` | Periodic residual-clustering retrain/refresh (`curation-cluster-refresh` service). |
| `sam_worker_main.py` | Detection-cascade worker entrypoint (`curation-detection-worker` service). |
| `worker/` | Shared worker library: cascade runner, HTTP clients (segmenter, VLM), state/checkpoint handling. |
| `backfill_scores.py` | One-off CLI to backfill item-quality scores onto existing indexed items. |
| `run_probe.py` | Probe-inference backfill: runs a probe detector ONNX over every non-holdout item and writes the `probe_pred_*` fields behind the `/review` uncertainty + model-disagreement tabs and the `mistakenness` score. Dry-run by default; `--resume` skips items already scored by the same `--model-version`. |
| `reclassify_after_registry_growth.py` | Registry-growth loop: after adding classes/synonyms, promotes `<prefix>_unmatched` items whose raw VLM label now resolves to an active class (`<prefix>_reclassified`, never validated). Dry-run by default, idempotent, `search_after`-resumable. Pairs with `GET /curation/review/unmatched_terms`. |
| `seed_live_harness.py` | Seeds the throwaway `docker/test/compose.yml` live-verification stack with deterministic data — **never point this at a real deployment** (it refuses to run against an index without a `verify_` prefix). |
| `bakeoff/` | Detector bake-off evaluation harness (`curation-evaluator` compose service runs this on demand). |

## Makefile Operations

Many common operations are available via the Makefile:

```bash
make help          # List all available targets
make up            # Start all services
make down          # Stop all services
make logs          # View service logs
make test          # Run the pytest suite (.venv/bin/python -m pytest tests/ -q)
make curation-up   # Start the curation worker services (experimental, opt-in)
make curation-down # Stop the curation worker services
```

## Related Folders

| Folder | Purpose |
|--------|---------|
| [export/](../export/) | Model export scripts (ONNX, TensorRT) |
| [tests/](../tests/) | Test scripts and utilities |
| [benchmarks/](../benchmarks/) | Go-based benchmarking tool |
| [docker/test/](../docker/test/) | Live write-path verification harness (see `docker/test/README.md`) |

## Port Reference

| Service | Port |
|---------|------|
| FastAPI | 4603 |
| Triton HTTP | 4600 |
| Triton gRPC | 4601 |
| Triton metrics | 4602 |
| Grafana | 4605 |
| OpenSearch | 4607 |
