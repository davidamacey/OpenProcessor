# Scripts

Setup and management scripts, release tooling, dataset fetchers, code
generators and the curation workers and tools.

Run Python scripts with the project venv binary (`.venv/bin/python`), or inside
the `yolo-api` container where the script says so.

## Root-level scripts

### `setup-openprocessor.sh` (repo root)

The one-line installer. It installs a pinned release into its own directory
with no git clone and no host Python. Flags and behavior:
[INSTALLATION.md](../INSTALLATION.md#one-line-installer).

### `openprocessor` (repo root) and `openprocessor.sh`

The management CLI. `scripts/openprocessor.sh` is a three-line shim that
`exec`s the root `openprocessor` script, so `./scripts/openprocessor.sh status`
works from a checkout. Every subcommand is listed in
[INSTALLATION.md](../INSTALLATION.md#the-openprocessor-cli); the ones you use
most:

```bash
./openprocessor status                 # containers, health, GPU memory
./openprocessor logs yolo-api -f       # follow one service
./openprocessor restart yolo-api
./openprocessor models                 # Triton model states
./openprocessor models install --only pe   # export and load one model group
./openprocessor curation up            # start the curation workers
./openprocessor sample coco            # public COCO sample
./openprocessor vlm list               # local VLM catalog
./openprocessor vlm use <id>           # switch the local VLM model
./openprocessor vlm key set <slug>     # store a VLM API key under secrets/vlm/
./openprocessor help
```

### `setup.sh`

Source-checkout setup: detects the GPU, picks a profile, pulls (or builds) the
API and Triton images, downloads and exports models, writes `.env`, starts the
services and runs smoke tests. Not for installed directories.

```bash
./scripts/setup.sh                                  # interactive
./scripts/setup.sh --yes                            # defaults, no prompts
./scripts/setup.sh --profile=standard --gpu=0 --yes
```

| Flag | Meaning |
|---|---|
| `--yes`, `-y` | accept defaults |
| `--profile=NAME` | `minimal`, `standard` or `full` |
| `--gpu=ID` | GPU to use (default 0) |
| `--skip-export` | keep existing TensorRT engines |
| `--skip-download` | keep existing weights |
| `--skip-start` | configure only |
| `--force` | overwrite an existing `.env` (it is never touched otherwise) |
| `--curation` | print the curation next steps at the end |

### `resize_images.py`

Batch image resizing with multiprocessing.

```bash
.venv/bin/python scripts/resize_images.py /path/to/images --size 640
.venv/bin/python scripts/resize_images.py /path/to/images --size 1024 --output /path/to/output --workers 16
```

### `docker-build-push.sh`, `security-scan.sh`, `export_paddleocr.sh`, `setup_face_test_data.sh`, `clone_reference_repos.sh`

Build, scan and one-off setup helpers. Each has a usage header; targets such as
`docker-build-push.sh all|api|triton|local` and `security-scan.sh all|api|triton|install`
are described there. `make scan` and `make clone-refs-*` wrap two of them.

## `scripts/lib/`

Shell libraries sourced by the installer and the CLI: `colors.sh`, `gpu.sh`,
`ports.sh`, `config.sh`, `download.sh`, `export.sh`, `model_setup.sh` (model
groups), `opensearch_heap.sh` (heap sizing), `image_keys.sh` (the
`images.lock` key table), `vlm_catalog.sh` (reads `examples/vlm/catalog.tsv`)
and `vlm_switch.sh` (`openprocessor vlm use|apply|status|probe`).

## `scripts/release/`

| Script | Purpose |
|---|---|
| `build_and_publish.sh` | Builds every published image, gates on a Trivy scan, and with `--push` pushes digest-pinned tags and writes `images.lock`. `--dry-run` pushes nothing (`make release-dry-run`, `make release`) |
| `build_deploy_bundle.sh` | Builds the installer's release assets (`SHA256SUMS`, the file list from `release-manifest.txt`): `build_deploy_bundle.sh vX.Y.Z [SRC] [OUT]` |

## `scripts/codegen/`

| Script | Purpose |
|---|---|
| `generate_contracts.py` | Regenerates (or `--check`s) every file under `contracts/` (`make contracts`, `make contracts-check`) |
| `export_api_contracts.py`, `export_region_status_to_ts.py` | The individual generators |
| `check_naming_leaks.py` | Pre-commit scan for private names and domain vocabulary in the tracked tree; allowlist in `naming_leak_allowlist.txt` |
| `check_no_literal_region_fields.py` | Region-field literal ratchet (see [CONTRIBUTING.md](../CONTRIBUTING.md)) |
| `check_file_size.py` | Per-file line-count cap |

## `scripts/datasets/`

Public, license-filtered sample data. No dataset ships in the repo.

| Script | Purpose |
|---|---|
| `fetch_coco_subset.py` | Pinned, seeded COCO 2017 subsets (`make sample-coco-readme`, `sample-coco`, `sample-coco-cars`). Manifests in `manifests/` |
| `fetch_openimages_plates.py` | Pinned Open Images V7 region sample (`make sample-plates`) |
| `build_import_fixture.py` | Builds the four dataset-import layouts (`yolo`, `coco`, `yolo_region`, `yolo_region_only`) and `FIXTURE.json` from a COCO subset (`make sample-coco-import`) |

`make sample-clean` removes everything fetched.

## `scripts/examples/`

`wheel_example_live.py` runs the cars and wheels example against a live stack:
creates the project, activates the example region profile and prompt pack,
ingests the images, waits for the detection worker and exports the wheel boxes.
Flags: `--api`, `--project`, `--api-prefix`, `--container-dir`, `--host-dir`,
`--drain-timeout`. See the example in [README.md](../README.md#example-cars-and-wheels).

## `scripts/docs/`

| Script | Purpose |
|---|---|
| `check_docs_vs_code.py` | Checks docs against the code: routes exist, `OP_*` variables are read, links and anchors resolve. `--only FILE...` limits it |
| `capture_backend_screens.py`, `capture_hero_frames.py`, `create-workflow-gif.sh` | Build the docs-site screenshots and walkthrough GIF from public sample data |

## `scripts/curation/`: workers and tooling

The curation subsystem ships behind the `curation` compose profile. See
[`docs/CURATION.md`](../docs/CURATION.md). Scripts that act on a project take
`--project SLUG` (default `$OP_CURATION_PROJECT`, else `default`) and bind it
before touching OpenSearch.

### Workers (compose services)

| Path | Service | What it does |
|---|---|---|
| `region_worker_main.py`, `worker/` | `curation-detection-worker` | Region cascade over pending items, for every active project in turn |
| `vlm_worker.py` | `curation-vlm-worker` | Long-lived VLM labeling and verification loop |
| `auto_label_worker.py` | `curation-auto-label-worker` | Drives the `POST /curation/projects/{project}/pipeline/auto_label` job protocol |
| `cluster_refresh_daemon.py` | `curation-cluster-refresh` | Triggers residual-clustering refresh as the item count grows |
| `bakeoff/` | `curation-evaluator` | Model-comparison harness: scores models per class on an eval dataset's test split, aggregates comparisons; domain examples under `examples/bakeoff/` |

### Ingest and import

| Path | What it is |
|---|---|
| `ingest_walker.py`, `_fast_walk.py` | Parallel bulk-directory ingest. Walks a directory the API container can see and posts paths to `POST /curation/projects/{project}/ingest/batch`, with a resumable progress file. Run it inside `yolo-api`: `--root /data/source/...` |
| `ingest_upload.py` | Byte-upload ingest for storage the API cannot mount: reads files locally and posts them to `POST /curation/projects/{project}/ingest/upload`; resumes with `POST /curation/projects/{project}/ingest/path_lookup` and server-side content-hash dedup |
| `import_labeled_dataset.py` | Client of `POST /curation/projects/{project}/datasets/imports`. Previews a dataset (YOLO, COCO, OpenProcessor export), builds the by-name mapping from `--map CLASS=ID`, `--create CLASS[=NAME]`, `--skip`, `--region`, `--accept-suggestions` (`--images-only` skips every class), starts or resumes (`--resume IMPORT_ID`) and polls. `--dry-run` previews only; `--state-dir` writes the per-split cohort `eval_regions_vs_gt.py` reads |
| `yolo_dataset.py` | Shared YOLO dataset discovery for the two tools above |
| `eval_regions_vs_gt.py` | Scores the region cascade against a whole-frame YOLO ground truth of the region class: recall at IoU 0.5 and `--iou`, precision, F1, mean IoU, background false positives, per-detector and per-status breakdowns and a `misses.jsonl`. `--wait-pending` polls until the cascade drains |

### Backfills and repairs

Most are dry-run by default and write only with `--apply`, under OCC; check `--help` for each.

| Path | What it does |
|---|---|
| `backfill_scores.py` | Backfill item-quality scores |
| `backfill_region_embeddings.py` | Write `region_box_embeddings` for every embeddable box that has none (human-drawn, moved or older boxes) |
| `run_probe.py` | Probe-inference backfill: writes the `probe_pred_*` fields behind the uncertainty and model-disagreement review tabs; `--resume` skips items already scored by the same `--model-version` |
| `requeue_regions.py` | Re-run the region stage on items parked in a terminal failure status, through the same selection and lock rule as `POST /curation/projects/{project}/reprocess`. `--missing-status` backfills items with no region status. Never touches human- or import-owned boxes |
| `rederive_region_text.py` | Re-choose each box's `text` from its stored VLM and OCR readings under the profile's text rules; never changes human text |
| `reclassify_after_registry_growth.py` | After adding classes or synonyms, promote unmatched items whose raw VLM label now resolves to an active class (never validated) |
| `cluster_raw_labels.py` | Cluster the VLM's free-text class labels into candidate sub-classes and write the fields read by `GET /curation/projects/{project}/review/raw_label_clusters` (`--dry-run --report` previews) |
| `seed_class_registry.py` | Seed or extend `class_registry.json` from a detector ONNX's embedded names or a `data.yaml`, append-only; `--check` fails on class-order drift |
| `repair_empty_vlm_answers.py`, `repair_unmatched_class.py`, `repair_stale_label_source.py`, `repair_merged_active_classes.py`, `revert_class_cluster_promotions.py` | One-off data repairs for known legacy states; read each script's header before use |
| `prune_exports.py`, `prune_training_runs.py` | Keep-last retention for export directories, finished training runs and bake-off output |
| `seed_live_harness.py` | Seeds the throwaway `docker/test/compose.yml` live stack. It refuses to run against an index without a `verify_` prefix. Never point it at a real deployment |

## Makefile operations

```bash
make help           # every target
make up             # core stack (Triton, API, OpenSearch)
make up-monitoring  # core plus Prometheus, Grafana, Loki
make down
make logs-api
make test           # offline pytest suite
make curation-up    # curation workers
make contracts      # regenerate contracts/
make sample-coco-readme
```

## Related folders

| Folder | Purpose |
|---|---|
| [export/](../export/) | Model export scripts (ONNX, TensorRT) |
| [tests/](../tests/) | Offline test suite and utilities |
| [benchmarks/](../benchmarks/) | Go-based benchmarking tool |
| [docker/test/](../docker/test/) | Live write-path verification harness |

## Port reference

| Service | Port |
|---------|------|
| API | 4603 |
| Triton HTTP / gRPC / metrics | 4600 / 4601 / 4602 |
| Prometheus | 4604 |
| Grafana | 4605 |
| Loki | 4606 |
| OpenSearch | 4607 |
| OpenSearch Dashboards | 4608 |
| MLflow | 4609 |
| DCGM exporter | 4610 |
| Segmenter | 4611 |
| Local VLM | 4612 |
