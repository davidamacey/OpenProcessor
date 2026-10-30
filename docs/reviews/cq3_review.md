# CQ3: label operations with server-side undo

Branch `feat/v070-cluster-outputs`, on top of CQ1 and CQ2 (`7a87133d`). Design:
[`../design/cq3_label_operations.md`](../design/cq3_label_operations.md) (merged into this branch first,
docs only). Plan: section 6 (CQ3) of `docs/design/ship_track_plan.md` on branch `docs/ship-track-plan`.
Issues: none closed.

## What changed

- `a30a18df` feat(curation): tag human label writes with their operation and restore the audit verdict on undo.
- `8c3ced96` feat(curation): record bulk label and move operations and list them.
- `262f9638` feat(curation): undo a label operation by id, bounded and resumable.
- `test(curation): pin the detector filter fields in the clause golden`: two failures the full run found that
  predate this wave (both fail at `7a87133d`): the CQ2 filter fields had no golden case, and the CQ2 review
  named a review route that does not exist.
- This commit: the probe `scripts/bench/label_op_probe.py` (with its helper tests) and this package.

Summary:

- `PUT /crops/batch_label` and `POST /crops/move` (both forms) are recorded as operations: a record file
  under `<project_state_dir>/label_ops/<op_id>/` written before the first item, `members.json` after the
  bulk, a `label_op_id` keyword tag on each written item and an `op_id` on its class history entry. The
  response gains `op_id` (null for an empty selection; a dry run has no `op_id`, its shape is unchanged).
- `{P}/label_ops` (GET): newest first, `limit` 1 to 50, `undoable` filter, `current` counted with one
  `filters` aggregation per 50 operations. The newest 200 records per project are kept.
- `{P}/label_ops/{op_id}/undo` (POST, `dry_run` default true): per item `restored`, `already_undone`,
  `kept_changed`, `kept_holdout`, `kept_stale_class`; the decision is the pure `undo_outcome`
  (`src/domain/label_ops.py`), re-made inside the OCC merge. Reads the stored operation, so no id cap;
  `occ_bulk_rounds` pages of 500, one request in flight, `undo.lock` per project, `write.lock` per
  operation. Refusals: `404 label_op_not_found`, `409 already_undone | label_op_busy | label_op_in_progress`.
- One shared restore, `class_undo_update` (`src/services/curation/class_undo.py`), used by the per-item
  undo and the operation undo (a domain test pins byte equality). It now also restores the audit verdict
  and the previous tag, so an undone label returns the item to the audit queue (fixes the same stale verdict
  on the per-item undo).
- `POST /crops/label/undo_batch` is typed (`CropUndoBatchResponse`) and moved onto `occ_bulk_rounds`
  (5,000-id cap kept); `POST /crops/discard_batch` onto `occ_update_bulk`; the region `undo_batch` is bounded
  at 32 in flight. `undo_batch` left `LEGACY_DICT`.
- `make contracts` regenerated `contracts/openapi/curation.json` in the commits that changed routes or models.
  Docs: "Label operations" in `docs/design/curation_api_contract.md`, the `label_op_id` row in
  `docs/opensearch_schema_design.md`, `CHANGELOG.md` `[Unreleased]`.

## Acceptance

Setup: two throw-away CPU-only API containers (the `0.6.0` image, one uvicorn worker, no GPU), one with this
branch's `src` (code identical to `262f9638`; later commits change tests, the probe and docs only) and one with
`main`'s (`0e2d3fe7`), mounted read-only, on the `optest1` network against its OpenSearch. One new synthetic
project `cq3p1` (100,000 items from `cluster_list_probe.py seed`, plus a class registry holding the COCO names
in order and the two classes the probe writes). The project and both containers were deleted afterwards. A
project `cq3probe` created first was deleted after a sanity run. `/health` answered 200 in 28 to 53 ms before
each row. **The host was busy for the whole session** (1-minute load 8 to 100 over the three hours; every
row is marked in the raw JSON, rows above load 16 are `noisy: true`). Numbers are therefore upper bounds
except where stated.

| Bar | Measured | Met | Raw data |
|---|---|---|---|
| Undo of a 10,000-item op under 60 s, 0 lost and 0 extra writes (per item on `CLASS_STATE_FIELDS` and audit verdict, all 100,000 items), the 50 edited items keep their edit, histogram equals pre-op plus the edits | 7.1, 11.1, 11.6 s (all three noisy, load 40 to 85). Audit: 0 lost or extra, 0 wrong edits, histogram matches, 0 items still tagged, counts 9,950 restored / 50 kept_changed / 0 conflicts, partition holds | yes | `undo10k/branch_{1,2,3}.json` |
| Undo of a 20,000-item op under 90 s, same audit; dry run under 30 s | undo 20.9 (noisy), 26.2 (noisy), 25.0 s (quiet, load 12 to 15); dry run 4.4, 6.2, 6.5 s. Audit clean, counts 19,950 / 50 / 0 | yes | `undo20k/branch_{1,2,3}.json` |
| API worker peak RSS during the 20,000 undo at most 200 MB above the pre-undo level | peak minus baseline: 1.3, 1.3, 0.4 MB (baseline 928 to 933 MB, 42 to 53 samples at 500 ms; the 10,000 runs 0.2 MB) | yes | same |
| `{P}/label_ops` p95 at most 100 ms with 200 retained operations at 100k items (30 calls, 3 repetitions) | median of the three p95 values: 21.3 ms (`limit=20`), 26.3 ms (`limit=50`), 23.3 ms (`limit=1&undoable=true`); worst single call 31 ms. Load 24 to 26 (noisy) | yes | `list200/probe_*.json` |
| Batch label of 20,000 within 1.1x of the same call on `main` in the same session | 9 interleaved pairs (order alternated): branch / main = 1.09, 1.02, 0.99, 1.19, 0.94, 1.02, 0.96, 1.05, 0.91; **median 1.02**, one pair above 1.1 (1.19, both runs quiet). Absolute 23.6 to 41.2 s, drifting up through the session for both builds as the probe project's item histories grew. Two earlier unpaired runs read 28.7 against 25.1 s (1.14) under load 45 to 70 and are superseded by the paired runs | yes (median); not met in every single pair | `label20k_pairs/`, `undo20k/` (`label_s`), `label20k_main/` |
| Counts: `restored = N - K`, `kept_changed = K`, `conflicts = 0`, partition holds | all six runs | yes | `undo10k/`, `undo20k/` |

Caveats, stated plainly:

- All measurements ran through one API worker in a container on a host with load far above the protocol's 16.
  Only `undo20k/branch_3` and most of the paired label runs were below it.
- The synthetic seed stores some validated items outside their class cluster (`cluster_id` a candidate id).
  The existing restore rule puts a validated class back in its class cluster, so those items come back with
  `cluster_id == class_id`. The probe reports them as `normalised_cluster_only` (190, 190, 172 in the 10,000
  rows, 0 in the 20,000 rows) and counts them as restored; every other field matched. Real items satisfy the
  invariant, so this does not occur on real data.
- The first run of the probe on the project, before the registry held the COCO names, reported 271 of 500
  items as `kept_stale_class`: the seed's class ids named other classes than the registry. That is the rule
  working (a restore target the registry no longer names that way is refused), not a defect.
- `main`'s own `undo_batch` (used by the probe only to revert its baseline labels) twice answered
  `409 project_read_only` mid-revert: the project registry refresh failed under load and the binding
  fails closed. The label timings are recorded before the revert and were not affected; pairs 4 (main side)
  was lost, so 9 of the 10 pairs have both sides. The probe now retries that answer.
- The branch container's start migrated the shared stack's `default` project items index (additive keyword
  field `label_op_id`) as any branch API start does.

## How it was tested

Red first (each seen failing before the code existed, as `ImportError` on the missing modules or a failing
assertion):

- Commit 1 at `7a87133d`: `tests/curation/test_label_ops_domain.py` (id format, decision table with the rule
  order, the history stack, byte-equal restore, tag and audit verdict in snapshots, `current_query` equals the
  `restore` set), the mapping and migration test in `test_items_mapping_explicit.py`, the combine copy
  assertion in `tests/projects/combine/`.
- Commit 2 at `a30a18df`: `tests/curation/test_label_op_record.py` (op id and tag on written items and
  history, nothing recorded for a dry run or an empty selection, no write without a record, the list route:
  order, `current`, limits, `undoable`, interrupted and writing status, retention that never prunes a live
  operation), `tests/curation/cross_project_leak/test_label_ops_isolation.py` (list half).
- Commit 3 at `8c3ced96` (15 of 16 failed; the 404 test passes vacuously on a missing route and is proved by
  the mutation below): `tests/curation/test_label_op_undo.py` (exact restore of a 12,000-item selection, later
  edit kept, ops undone in any order, one-by-one undo counted `already_undone`, holdout, exclusion before and
  after, merged and renamed class, second undo 409 on real and dry run, OCC conflict reported and resumed,
  dry run predicts the real run and writes nothing, one request in flight and pages of 500, busy and
  in-progress locks, an interrupted writer undone by the tag scan, malformed or other-project ids, move ops,
  the partition invariant on every response), `tests/curation/test_undo_batch_bulk.py`, the region 32 bound
  in `test_region_undo.py`, the undo half of the leak test and the every-route sweep rows
  (`op_id` in `route_params`, the undo body in `route_bodies`, a `NO_WRITE` reason).
- `tests/test_contract_typed_responses.py`: the `undo_batch` allowlist entry removed (the ratchet fails on a
  stale entry).
- `tests/test_label_op_probe.py`: the probe's pure helpers (audit, expected histogram, RSS reader).

Mutations, each made alone on the committed code, the named tests watched failing, then restored (all 16 killed;
one test needed strengthening after the first pass, see the notes):

| Mutation | Failing tests (design row) |
|---|---|
| `history[-1]` instead of `find_undo_snapshot` in `undo_outcome` | `test_ops_undo_in_any_order`, `test_items_undone_one_by_one_count_as_already_undone` (10, 11) |
| `label_op_id` removed from `CLASS_STATE_FIELDS` | `test_snapshot_records_and_restore_returns_the_tag_and_audit_verdict`, `test_ops_undo_in_any_order` and 3 more (5, 10) |
| class-name fingerprint check removed | `test_merged_or_renamed_class_is_kept`, decision table `merged_class` and 2 more (14) |
| exclusion-after-op check removed | `test_excluded_after_the_op_is_kept_and_before_the_op_is_restored_parked`, `excluded_after`, `excluded_no_stamp` and 1 more (13) |
| holdout check removed | `test_holdout_items_are_kept`, decision table `holdout` and 2 more (12) |
| registry stale check removed | `test_merged_or_renamed_class_is_kept`, decision table `renamed_restore_target` (14) |
| per-item undo stops recording the cancelled `op_id` | `test_items_undone_one_by_one_count_as_already_undone` and 3 more (11) |
| conflicts counted as restored | `test_occ_conflict_is_reported_not_swallowed` (16, 25) |
| dry run calls `bulk` | `test_dry_run_writes_nothing_and_predicts_the_real_run`, `test_holdout_items_are_kept` and 2 more (17) |
| bulk loop replaced by `asyncio.gather` of `occ_update_one` | `test_undo_keeps_one_request_in_flight` and 9 more (18) |
| `undo.lock` not taken | `test_a_second_undo_in_the_project_is_409_busy` (19) |
| record written after the bulk instead of before | 16 tests, among them `test_undo_while_writing_is_409_and_an_interrupted_op_is_undoable` and `test_a_write_without_a_record_never_happens` (20) |
| op id joined to the path with no regex check | `test_unknown_malformed_or_pruned_op_id_is_404` (21) |
| store dir built from a fixed slug instead of the bound project | `test_an_operation_is_listed_and_stored_only_in_its_own_project`, `test_another_projects_operation_id_is_404_and_nothing_changes` (27) |
| `already_undone` check removed | `test_second_undo_is_409_already_undone_and_writes_nothing` (15) |
| audit verdict not restored | `test_exact_restore_of_a_12000_item_selection_op`, `test_snapshot_records_and_restore_returns_the_tag_and_audit_verdict` (5, 8) |

Notes on the mutation pass: the first run of `history[-1]` failed only test 10 and the first run of the
path-join mutation survived. Test 11 now also has an item whose history ends in a machine entry on top of the
operation's (the stack walker skips it, `history[-1]` does not), and the 404 test plants a real record where an
unchecked join would land, so both mutants are killed.

Suites run: the touched files above plus `tests/test_class_label_single_writer.py`,
`tests/curation/test_crops_undo_exclude.py`, `test_new_class_resolve.py`, `test_crop_discard.py`,
`test_bulk_selection_routes.py`, `test_labels_export_roundtrip.py`, `test_router_wireup.py`,
`test_ensure_indexes.py`, `test_index_schema_marker.py`, the whole `tests/curation/cross_project_leak/` sweep
(31 tests), `tests/projects/combine/`, `tests/test_contract_typed_responses.py` and
`tests/curation/test_script_api_prefix.py`; `-k "history or undo or upsert or label or occ or dataset_import or audit"`
over all of `tests/` after the first commit (995 passed). Pre-commit (all hooks) passes on each commit. The full offline suite (`-m 'not live' -n 4`, 15.5 minutes) ran once on the finished wave: 9,602 passed,
24 skipped, 3 failed. Two were the CQ2 leftovers above (fixed); the third was the probe's own dict literal
tripping `test_no_second_class_validated_writer` (fixed). The three files were re-run green; the rest of the
suite was not repeated. The frontend was not touched (see Risks).

## Risks and what was not done

- **Frontend contract pin.** `frontend/src/lib/contract/selectionContract.test.ts` pins the exact keys of
  `BatchRelabelResponse`; the added `op_id` makes it fail until UI3 adds `op_id: string | null` to
  `BulkLabelResult` (`frontend/` was not touched, per the lane split). UI3 also deletes the client-held undo
  for label and move and decides whether `updated_ids` still has a reader.
- Error code for a read-only binding: the design said `read_only`; the code uses the existing
  `project_read_only` (409), which is what the rest of the API answers for a read-only project.
- An OpenSearch outage during the member scan or the undo surfaces as a 500, not the 503 the design names:
  the app has no mapping for transport errors and none was added. Nothing is written and the record is left
  `undoing` or unchanged, so a re-send resumes.
- Undo is synchronous: 26 s for 20,000 items here under load, inside the 120 s general proxy limit.
  UI3 adds the `label_ops/` location with a 600 s read timeout to `frontend/nginx.conf` (D-CQ3-8). A 504 is
  not a failed undo: re-read `{P}/label_ops`.
- `current` overcounts only items whose restore target class was renamed (the registry rule is not
  expressible as a query); the undo reports them `kept_stale_class`.
- The loop rollback (`as_left`, on the assist-loop branch) checks `history[-1]`; an item labelled then
  restored by an operation undo is kept by it (fail safe, noted for the L lane).
- Members files: up to about 0.9 MB per operation, 200 per project.
- Not done: folding discard and exclude into operations (D-CQ3-6), a background job (only if the 90 s bar
  is missed; it was not), the per-item undo's stale-class check (D-CQ3-3 applies to the operation undo only).

## Decisions needed from the owner (gate ids)

All taken as the design recommends, as instructed: G7 and D-CQ3-1 to D-CQ3-8. D-CQ3-3 is applied to the
operation undo only; whether the per-item undo should refuse a renamed class too stays open.

## Five-minute check for the owner

On a stack running this branch's `src` (project with a few hundred items and two classes):

```bash
P=http://127.0.0.1:5603/curation/projects/<project>
curl -s -X PUT "$P/crops/batch_label" -H 'content-type: application/json' \
 -d '{"selection":{"filter":{},"limit":200,"sample":"random","seed":1},"class_id":<id>}' | python3 -c \
 "import sys,json; d=json.load(sys.stdin); print(d['updated'], d['op_id'])"
curl -s "$P/label_ops?limit=1" | python3 -m json.tool
curl -s -X POST "$P/label_ops/<op_id>/undo" -H 'content-type: application/json' -d '{}'
curl -s -X POST "$P/label_ops/<op_id>/undo" -H 'content-type: application/json' -d '{"dry_run": false}'
curl -s -X POST "$P/label_ops/<op_id>/undo" -H 'content-type: application/json' -d '{"dry_run": false}'
```

Expected: the label returns `updated` 200 and an `op_id`; the list shows it with `current` 200 and
`undoable` true; the first undo (default dry run) reports `restored` 200 and changes nothing; the second
restores 200 items (`status` `undone`); the third answers `409 already_undone`. Relabel one item by hand
between the label and the undo and the undo reports it `kept_changed` and leaves it alone.
Probe: `.venv/bin/python scripts/bench/label_op_probe.py run --api <url> --opensearch <url> --project <slug> --items 10000 --container <api container>`
after `cluster_list_probe.py seed` and `label_op_probe.py classes`.
