# CC-10 evidence: in-flight upstream requests of the VLM batch routes (4A-0)

Status: written from code reading on the `freeze/w23a` head; no live run, no stub run in this lane.
Task 4A-0 of [05_b4_stateless_api.md](../05_b4_stateless_api.md) asked for this record; B2 lane 4D-8
(D-29a, D-84) already moved the two routes to jobs and bounded the fan-out, so this file records the
before and after numbers and where the executable evidence lives.

## What the route accepted

| Route | Largest request | Chunking |
|---|---|---|
| `POST /curation/projects/{project}/vlm/label_batch` | 64 `crop_ids` (400 above) | chunks of `max_images_per_call` (endpoint default 8, `OP_VLM_MAX_IMAGES_PER_CALL`, range 1-64) |
| `POST /curation/projects/{project}/vlm/verify_region_batch` | 64 `items` (400 above) | the same chunking (`verify_region_batch`) |

## Peak in-flight upstream requests of one batch

| `max_images_per_call` | Before 4D-8 (`asyncio.gather` over all chunks, inside the API worker) | Now (`run_chunks`, in the job's worker) |
|---|---|---|
| 8 (default) | 8 | 8 |
| 4 | 16 | 8 |
| 1 | 64 | 8 |

Before: `ceil(64 / max_images_per_call)` chunk calls started at once, per request, per API worker; with
N concurrent batch requests the product was N times that, times up to 32 workers. After:
`src/services/labeling/vlm_chunks.py` `run_chunks` keeps at most `VLM_CHUNK_CONCURRENCY = 8` in flight
inside an `asyncio.TaskGroup` (Python docs, "Task Groups": a failing chunk cancels its siblings), and
the call runs in a Celery job (`vlm_label_batch`, `vlm_verify_region_batch`), so the API worker holds
none of it. The per-endpoint token bucket (`requests_per_second`, default 500) still sets the rate.

## The bound

8 per batch. Why 8: it equals the default chunk count of a full 64-crop batch at the default cap (so the
default path loses no parallelism) and keeps one batch from monopolising a small local vLLM. A job-wide
cap across concurrent jobs is not needed: the two kinds have no exclusive group, but each job holds at
most 8 and the worker's concurrency bounds the jobs.

## Executable evidence

`tests/labeling/test_vlm_chunk_fanout.py` counts calls in flight on a stub upstream: 64 crops at one
image per call make 64 calls with a peak of 8 (the base before 4D-8 gave 64). The wall time of a stub
call is a test parameter (5 ms), so it is not recorded as a measurement; the real wall time depends on
the VLM and belongs to the live harness (`tests/live/test_live_vlm.py`), not to this record.
