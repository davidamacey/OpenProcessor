# W5e: registration-plate text (the plate-text wave, between W5d and W6)

| Field | Value |
|---|---|
| Status | plan, not started; written 2026-11-04 by the research integration pass |
| Design | [09h_research_plate_text.md](09h_research_plate_text.md) sections 5 and 7 (the contract; this file only orders the work and fixes file ownership) |
| Owner direction | plate text is ordinary searchable label data, stored as readable text, searchable (exact, fuzzy, fragment), detectable and labelable, "a development tool for labeling data" (2026-11-05, D-79: no privacy modes, no ceiling, no salt, no plate purge, no sink test); plate text is a must (2026-11-04); its own wave W5e (RD-3, implied by the owner's RD-4 answer) and its tables in their own W5e revision `0004_region_text` (RD-4, approved 2026-11-04; [00d](reference/owner_questions.md)); the plate-text index normalizer stays (Q-A); PP-OCRv6 is one short reader comparison (Q-B); the plate data are freely downloadable sets plus set P, without UFPR-ALPR and LPLC (RD-10) |
| Depends on | W5c merged (track `ocr` adds the `rec_char_min` output, [06c](06c_b5_crops_ocr_faces.md) research amendment); W4 merged (the capture lane writes `capture_taken_utc` on items, which the `capture_window` consensus basis reads; there is no plate privacy lane in W4 any more (owner decision of 2026-11-05, D-79)); W2.3b merged (the near-duplicate runs that the `near_dup` basis reads); the plate prototypes P0-P5 read in PW-2 ([00f](08_research_prototype_track.md); `pt:P6` is cut before the tag, so plate boxes come from the segmenter with the text hint, today's path) |
| Integration branch | `int/b5e-plate-text`, cut from `main` after the W5d PR merged; one PR |
| Lanes | store first; then fields and jobs in parallel; the reader lane only if P1 adopts a dedicated reader |
| Merge order | `w5e/region-text-store`, `w5e/region-text-fields`, `w5e/region-text-jobs`, `w5e/region-text-reader` (conditional), `w5e/frontend-text` (conditional) |
| GPU | one short gate at the end of GPU window 2 (about 2 h, GPU 0, `-p op-b5e-gpu`): the chosen reader on Triton and a public-sample read |
| Issues | #401 (closes with the W5e PR on the owner's go; milestone v0.7.0 per 00b section 4.5) |

Real plate readings never appear in tracked text: fixtures use synthetic strings built by the
test that fail every plate format, never a real reading (review 3.7); search fixtures that must
survive the alphanumeric normalizer are 12 characters long, longer than any plate format.

---

## Lane `w5e/region-text-store` (first)

**Owns:** new `src/db/models/region_text.py`, new revision `src/db/migrations/versions/0004_region_text.py`
(`down_revision = '0003_near_duplicates'`, [02d](02d_research_seam_schema.md) section 4), new
`src/db/repos/region_text.py`, `tests/db/test_region_text_schema.py`.

| ID | Task | Red test | Control |
|---|---|---|---|
| PT-S1 | the four tables of the plate plan 5.2 with the X-1 change of 02d section 4; `region_text_consensus_marker` in the `project_state` key `CHECK`; RLS | `tests/db/test_region_text_schema.py`: every table, index and policy present; the tables hold no text-valued column besides the listed codes; a box in two current groups is a unique violation | `alembic` one head; downgrade to 0003 and up |
| PT-S2 | repositories: runs (one open `region_pass` run per reader configuration by the partial unique index), consensus groups replaced in one project-scoped transaction | `tests/db/test_region_text_repo.py`: a second open run for one configuration is a unique violation | read after write |

## Lane `w5e/region-text-fields` (after the store lane merged)

**Owns:** `src/domain/region_boxes.py` (the new `text_*` keys; `text_engine_version` deleted), new
`src/domain/region_text.py` (normalize, fold, tiers, consensus methods; pure), `region_text.py` and
`region_text_stage.py` (the region-stage reader path), the region-box mapping lines for the text
fields in the items body module, the `region_profile` body model fields of plate plan 5.5 (moved
here from 2C: their first reader is this wave), `src/services/curation/wire.py` (`region_boxes_to_wire`),
the review and crops query builders (exact, fuzzy, fragment; `same_text_as`), `contracts/`, tests
under `tests/region_text/`.

| ID | Task | Red test | Control |
|---|---|---|---|
| PT-F1 | per-box fields of plate plan 5.1 written by the reader and the consensus; `text` stays the resolved reading, `text_frame` the frame's own (RD-66, answered as recommended) | `tests/region_text/test_box_fields.py`: a reading writes `text_norm` and `text_fold` from `text`; `text_engine_version` is gone | existing region-stage tests |
| PT-F2 | the domain fold agrees with the index normalizer `op_text_compact` (2F-6) on a fixture list | `tests/region_text/test_fold_matches_normalizer.py` (fake `_analyze` or live) | - |
| PT-F3 | search: `term`, `fuzzy` (fuzziness 1, transpositions, `max_expansions` 50) and `wildcard` on `text_fold.sub`, metacharacters escaped; the regions-tab text filter moves to the same builder | `tests/region_text/test_plate_search.py` | a non-identity profile's text search unchanged |
| PT-F4 | wire: the new keys added, `text_engine_version` dropped; the text values are on the wire as readable text; `text_norm` and `text_fold` stay storage-only; `make contracts` | `tests/region_text/test_wire_key_set.py` pins the key set | - |

## Lane `w5e/region-text-jobs` (after the store lane merged)

**Owns:** new `src/config/job_kinds/region_text.py`, `src/config/task_routes/region_text.py`, new
`src/tasks/region_text.py`, the two handlers, `src/config/queue.py` (declares `text_read`), the beat
entry `op.periodic.region_text_consensus_sweep` (60 s) in `src/tasks/beat_schedule.py` and `src/tasks/periodic.py`, the `text-read-worker` service in
`docker-compose.yml`, `docker-compose.dev.yml` and `docker/test/compose.yml`, the reread and consensus routes, tests under `tests/region_text/test_jobs_*.py`.

| ID | Task | Red test | Control |
|---|---|---|---|
| PT-J1 | `region_text_reread` (queue `text_read`) and `region_text_consensus` (queue `jobs`), one exclusive group `region_text_write` ([02b](02b_jobs_locking_events.md) section 2.1) | `tests/region_text/test_jobs_exclusive.py`: a reread while a consensus runs is 409 `job_busy`; `tests/test_queue_consumers_from_compose.py` lists `text_read` | the per-family recipe tests |
| PT-J2 | consensus: `best_frame`, `position_vote`, `aligned_vote` (the method and basis from P3 and P4); the `near_dup` basis reads the image-level `dedup` groups of the current near-duplicate run; the `capture_window` basis reads `capture_taken_utc` (W4), falling back to `capture_taken_local` only within one camera make and model (RD-68, answered as recommended) | `tests/region_text/test_consensus.py`: five frames of one synthetic plate with two misreads give the majority reading, `text_frames = 5`, `text_frames_agree = 3` | a single frame keeps its own reading |
| PT-J3 | the sweep starts a consensus for a project whose marker is older than 30 s; project delete removes the text (no separate retention task, D-79) | `tests/region_text/test_sweep_marker.py` | no new readings, no job |

## Lane `w5e/region-text-reader` (only if P1 adopts a dedicated reader; pending RD-11)

The Triton model directory of the reader, its export step, the installer model group (opt-in, off by
default), `ATTRIBUTION.md` with the provenance gap stated; red: the P5 numbers reproduced on Triton
within the 2x-spread rule. Without an adoption this lane does not exist and PP-OCRv5 stays the reader
(or PP-OCRv6, if `pt:P1` and [06c](06c_b5_crops_ocr_faces.md) adopted it in W5c).

## Lane `w5e/frontend-text` (only if `rg -n "text_engine_version" frontend/src` finds a reader)

Cropwright follows the wire change (the reader and version fields, the tier badge); lane checks as
every frontend lane.

## Sweep and exit

One full sweep on `int/b5e-plate-text`: T1 offline (pre-commit `--all-files`, the sharded suite,
`-m db`, `make contracts-check`, the docs check), the frontend set if `frontend/` changed, then the
short GPU gate in window 2 (a public sample read through the chosen reader, consensus on a public
burst set, exact, fuzzy and fragment search). Exit: #401's
acceptance in the plate plan section 8; the R-final row of [07](07_release_acceptance.md) section 4.7.
