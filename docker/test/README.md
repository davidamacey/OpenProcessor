# Live write-path verification harness

An isolated, disposable Compose stack whose only job is to let the
`tests/live/` suite exercise the curation **write** endpoints against a real
OpenSearch, a real file protocol, and a real OpenAI-compatible chat endpoint —
without a GPU, without Triton, and without being able to touch a production
deployment.

Until this existed, none of the curation write endpoints had ever been run
against a live stack: the offline suite covers them with mocked clients, which
cannot catch a mis-wired env var, a mapping that rejects a real document, a
query that silently matches nothing, or a job file nobody writes.

## Running it

```bash
# From the repo root.
mkdir -p docker/test/verify-data/jobs

docker compose -p op-live-verify -f docker/test/compose.yml up -d --wait
python -m pytest tests/live -q --no-cov -m live
docker compose -p op-live-verify -f docker/test/compose.yml down -v --remove-orphans
```

The suite wipes and reseeds the harness at the start of every session
(`scripts/curation/seed_live_harness.py --wipe`, run inside the API
container), so runs are repeatable and order-independent across sessions.

`tests/live` is marked `live` and deselected by the default `addopts`, so an
ordinary `pytest tests/` never touches any of this.

## What it starts

| Container | Host port | Role |
|---|---|---|
| `op-verify-opensearch` | 14702 | OpenSearch 3.6, single node, 512 MB heap |
| `op-verify-api` | 14701 | this repo's API image, `--workers=1` |
| `op-verify-fake-vlm` | 14704 | ~200-line stdlib OpenAI-compatible chat endpoint |
| `op-verify-trainer` | — | shell loop implementing the trainer half of the job-file protocol |

Design choices that are not arbitrary:

* **Host ports are in the 147xx band.** Outside 4600–4610 (a real deployment),
  outside the 5xxx band, and outside the ports an older ad-hoc harness used, so
  both can coexist on one machine.
* **Every index name is `verify_`-prefixed.** `tests/live/conftest.py`'s first
  fixture refuses to run if the OpenSearch endpoint is not on a 147xx port, or
  if the cluster holds any index that is neither `verify_`-prefixed nor a known
  OpenSearch-internal one. A mis-wired host cannot write into real data.
* **`/var/run/docker.sock` is never mounted.** The GPU arbiter is a no-op in
  this repo and must stay one; `tests/live/test_live_train.py` snapshots
  `docker ps` either side of `POST /train/start` to prove it.
* **`--workers=1`.** Several cancel endpoints and the SSE hub keep in-process
  state; with two workers a cancel POST can land on a different worker than the
  start POST.
* **Everything writable is one bind mount** (`docker/test/verify-data/`, which
  is git-ignored). The registry, exports, job files and state dir are all
  readable from the test process, so assertions check the bytes that were
  actually written rather than the response envelope.

## The fake VLM

`fake_vlm.py` answers `POST /v1/chat/completions` with a fixed reply chosen by
prompt kind (classification / region verify / region visibility). It exposes a
small control surface for tests: `POST /__control` to change a reply knob,
`GET /__stats` for per-kind call counts, `POST /__reset`.

**It proves wiring and persistence, not label quality.** Every answer is a
constant. What the tests verify is that the router builds a well-formed
request, that the labeler's parser understands the reply, and that the parsed
verdict lands in OpenSearch under the documented field names.

## The fake trainer

`fake_trainer.sh` implements the trainer side of the file protocol in
`src/services/training/jobs.py`: it watches the shared jobs directory for
`<job_id>.job.json`, walks the run through `queued → running → finished`
(three fake epochs, one per poll), honours a `<job_id>.cancel` sentinel, and
writes `<job_id>.run.log` plus `<job_id>.manifest.json`. It never touches a
GPU or loads a model.

## Explicitly NOT verifiable here (needs a GPU / Triton)

These write endpoints are out of reach of this harness by construction, and
are the only ones that are:

| Endpoint | Why |
|---|---|
| `POST {prefix}/train/promote/{job_id}` | needs a real exported ONNX checkpoint and a live Triton to load it into |
| `DELETE {prefix}/models/{model_name}` | unloads a model from a live Triton |
| `POST {prefix}/pipeline/auto_label` and `/auto_label/start` | run the detector cascade, i.e. real inference |
| the evaluation half of `POST {prefix}/bakeoff/run` | runs a model over an eval dataset |

Everything else on the curation write surface is exercised by `tests/live/`.

## Findings this harness has already produced

Both are asserted as `xfail(strict=True)` tests, so they will fail loudly the
day they are fixed:

1. **Region verdict key mismatch** — the built-in prompt pack asks the model to
   answer with `is_region`, but both region-verdict parsers in
   `src/services/labeling/vlm_labeler.py` read `is_plate`. A model that follows
   the shipped prompt is parsed as a negative verdict with
   `reason='parse_failure'`. See
   `tests/live/test_live_vlm.py::test_vlm_verdict_key_matches_the_shipped_prompt`.
2. **Two index names bypass `CurationConfig`** — `op_umap_viz_state`
   (`embedding_viz.py`) and `op_umap_state`
   (`clustering/embedding_reduce.py`) are hardcoded, so no `OP_*_INDEX`
   setting can move them. See
   `tests/live/test_live_clusters.py::test_viz_state_index_honours_the_configured_index_prefix`.

A third, non-bug prerequisite is documented by
`test_embedding_scorers_report_their_missing_prerequisite`: the `uniqueness`
and `near_dup` scorers require a trained IVF centroid store, which only the
item-clustering pipeline produces.
