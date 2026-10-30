# CQ3: label operations with server-side undo (design review)

Status: design, not implemented (2026-10-29). Branch `feat/v070-cluster-outputs`, on top of CQ1 and
CQ2 (`7a87133d`). Plan: section 6 (CQ3) and section 8 (UI3, UI4) of
`docs/design/ship_track_plan.md` on branch `docs/ship-track-plan` (not on this branch). Gate **G7**
(label operation semantics) must be answered before any code; section 12 lists the other owner
decisions.

Read first: [`CLAUDE.md`](../../CLAUDE.md) (lock rule, OCC helpers, class identity by name, wire
contract, fail closed), [`../VISION_AND_GOALS.md`](../VISION_AND_GOALS.md), the CQ1 and CQ2 review
packages ([`../reviews/cq1_review.md`](../reviews/cq1_review.md),
[`../reviews/cq2_review.md`](../reviews/cq2_review.md)) for the conventions this wave follows.

Planned routes are written `{P}/...` (with `{P}` = `/curation/projects/{project}`), method in
brackets after the path, as the docs checker requires.

---

## 1. Problem, from the code

| Finding | Evidence |
|---|---|
| Undo of a bulk label is capped at 5,000 ids while the label covers up to 20,000 | `CropUndoBatchRequest.crop_ids` `max_length=5000` (`src/routers/curation/_common_models.py`); `MAX_BULK_SELECTION = 20_000` (`src/routers/curation/_selection.py`) |
| Undo has no concurrency limit | `undo_crop_labels` in `src/routers/curation/label_undo.py` runs `asyncio.gather` over every id, each an `occ_update_one` with `refresh='wait_for'`; `discard_crops` (same file) and `undo_crop_regions` (`src/routers/curation/edit_undo.py`) have the same unbounded gather |
| Undo does not survive a reload | Cropwright keeps the written ids client side (`undoStore.recordWrites(body.updated_ids)` in `frontend/src/lib/itemFilter/selectionActionController.svelte.ts`) and replays them through `POST /crops/label/undo_batch` |
| `POST /crops/label/undo_batch` is untyped | `LEGACY_DICT` in `tests/test_contract_typed_responses.py` |
| An undo leaves a stale accuracy-audit verdict | `audit_outcome_fields` (`src/domain/audit_math.py`) stamps `audit_outcome` / `audit_human_name` on a human label; `restore_class_state` (`src/domain/history.py`) never restores them |

What already works and is reused:

- Every human class write snapshots the full pre-write class state into `class_id_history`
  (`record_class_snapshot`, `CLASS_STATE_FIELDS`), and `find_undo_snapshot` walks it as a stack
  (an unlabel entry cancels the next older human label entry). `restore_class_state` puts the item
  back exactly.
- `occ_bulk_rounds` (`src/clients/occ_bulk.py`) is the shared paged read-merge-conditional-bulk loop:
  one `_mget` and one conditional `_bulk` per page of `OCC_BULK_PAGE_SIZE` (500), 409s retried with
  re-read and re-merge, `refresh` only on the last bulk of a round.
- The loop rollback on `feat/v070-assist-loop` (`src/services/curation/assist_loop/rollback.py`,
  `as_left`) restores an item only while it is still exactly as the loop left it, re-checks inside
  the OCC merge, and counts the rest `kept_changed`. CQ3 applies the same rule to human operations.

## 2. Semantics (gate G7)

A **label operation** ("op") is one bulk human class write: `PUT /crops/batch_label` or
`POST /crops/move` (both forms: explicit `crop_ids` or a `selection`). Undoing an op is the per-item
undo (`POST /crops/{crop_id}/label/undo`) applied to exactly the items whose **next undo step is this
op**, and to no other item.

Per item, an undo pass decides one outcome (pure function, section 5.2):

| Outcome | When | Written |
|---|---|---|
| `restore` | The item still carries this op's tag, the op's entry is the top of its undo stack, nothing changed its class since (fingerprint, section 5.2), it is not in the test holdout, and the state to restore names a class that still exists under that name | the pre-op class state, exactly as recorded |
| `already_undone` | The op's write on this item was already undone (a previous pass, or a per-item undo) | nothing |
| `kept_changed` | Anything else: a person or an import relabelled it, a later op wrote it, it was excluded after the op, its class was merged or renamed, the item was deleted | nothing |
| `kept_holdout` | The item is frozen in the test holdout now | nothing |
| `kept_stale_class` | The state to restore names a class id whose registry name is no longer the recorded name (rename, merge, delete) | nothing |

Consequences, all deliberate:

1. **Later edits always win.** A human edit after the op is never reverted by undoing the op. The
   decision is re-made inside the OCC merge, so a person's write racing the undo wins.
2. **Undo is per item, not all or nothing.** An op with some changed items is still undoable for the
   rest. The plan's "`undoable` false once any item's newest entry is no longer this op" is replaced
   by "`undoable` while at least one item still carries this op's label" (owner decision D-CQ3-1).
3. **Ops undo in any order and stay consistent.** Op A labels items 1 to 100; op B relabels 50 to 100.
   Undo A restores 1 to 49 and keeps 50 to 100 (`kept_changed`, they carry B). Undo B puts 50 to 100
   back to A's label and A's tag (the tag is part of the snapshot), so A is undoable again for
   exactly those 51 items. This is why a second undo of A is refused only when nothing is left to
   restore (section 6.3), not merely because A was undone once.
4. **The holdout is never touched by a bulk undo**, even when the op itself labelled a frozen item
   through explicit ids or `include_test`. Per-item undo remains available for those (D-CQ3-2).
5. **Class identity by name.** The op records the class name it wrote; the fingerprint compares
   names (normalized with `normalize_class_name`), and the restore target is checked against the
   current registry by name. A restore never resurrects a renamed or deleted class name (D-CQ3-3).
6. Undo is a human write (`writer = HUMAN_UNLABEL_WRITER`), so it does not consult the automated-writer
   lock (`is_locked_class`): the item's current human label is the op's own label, which undo exists
   to remove. Every other human- or import-validated state is protected by the fingerprint and the
   stack check, not by the lock predicate.

## 3. What is stored where

| Data | Where | Bound | Retention |
|---|---|---|---|
| Op record (`state.json`) | `<project_state_dir>/label_ops/<op_id>/state.json`, written through `FileJob` (`src/services/curation/file_job.py`: atomic temp and rename) | under 4 KB | newest 200 ops per project (`LABEL_OPS_KEEP`), pruned on create; an op that is writing or undoing is never pruned |
| Members (the ids the op wrote) | `<op dir>/members.json`, written atomically after the bulk write returns | at most `MAX_BULK_SELECTION` = 20,000 ids (about 0.9 MB) for a selection, 5,000 for explicit ids | with the record |
| Locks | `<op dir>/write.lock` (held while the op writes), `label_ops/undo.lock` (held while any undo of the project runs); `fcntl.flock` through `exclusive_start_lock` (`src/services/curation/job_lock.py`) | empty files | with the dir |
| Item tag | new top-level items field `label_op_id` (keyword, indexed) | one keyword | until the next human class write or an undo |
| Per-item history | `class_id_history` entries gain `op_id` (the op that made the write, or whose write an undo entry cancels); `label_op_id` joins `CLASS_STATE_FIELDS`, so every snapshot carries the pre-write tag and every restore puts it back | about 40 bytes per entry | existing cap (32, none once validated) |

Why files and not an OpenSearch index (D-CQ3-4): `project_state_dir` is already per project
(`src/config/projects.py`), on the shared state volume every API worker mounts, deleted with the
project (`_project_scoped_dirs` in `src/services/projects/delete_paths.py`), and marked with the
owning slug (`ensure_marked_dir` in `src/config/project_context.py`). A new index role would touch the
guard, capacity, lifecycle, combine and delete paths; a new jobs dir would need a new environment
variable. Nothing queries op records except the two new routes.

Why an indexed item tag in addition to members: the list route must say how many items still carry
each op's label without reading 20,000 documents per op (one `filters` aggregation answers 50 ops),
and a crash between the bulk write and `members.json` must still be recoverable (a scan on the tag
finds what was written).

### 3.1 Op record (storage model `LabelOpRecord`, schema 1)

| Field | Type | Meaning |
|---|---|---|
| `schema` | int | 1 |
| `op_id` | str | `lop_<YYYYMMDDTHHMMSSffffff>_<6 hex>`, UTC; lexicographic order is creation order; matched by `LABEL_OP_ID_RE` before any path is built |
| `kind` | `label` or `move` | which route |
| `writer` | str | `human:batch_label_crops` or `human:move_crops` (the history writer of the op's entries) |
| `at`, `finished_at` | ISO-8601 | write start, write end |
| `class_name` | str or null | the class the op wrote (null for a candidate move) |
| `class_id` | int or null | registry id at write time, internal only (never served; identity is the name) |
| `target_cluster_id`, `target_kind` | int or null, `class` / `candidate` / null | move target |
| `label_source` | str or null | as sent |
| `target` | `crop_ids` or `selection` | request form (the filter itself is not stored) |
| `selected` | int | ids resolved |
| `count`, `conflicts` | int or null | written and conflicted, set when the write finishes |
| `fingerprint` | object | what the op wrote on every item, section 5.2 |
| `status` | `writing`, `written`, `undoing`, `undone` | stored state |
| `undo_passes` | list, newest 10 kept | `{at, restored, already_undone, kept_changed, kept_holdout, kept_stale_class, conflicts}` |

Served status adds two derived values: `interrupted` (stored `writing` and `write.lock` free: the
writer died) and `undo_interrupted` (stored `undoing` and `undo.lock` free).

## 4. Write path of the op (batch label and move)

`crops_write.py` stays thin; a new service function does the recording:

```python
# src/services/curation/label_ops/record.py
async def recorded_relabel(
    opensearch: Any,
    ids: list[str],
    *,
    op: NewLabelOp,                                   # kind, writer, class_name, class_id, target, fingerprint, ...
    merger_for: Callable[[str], Callable[[dict[str, Any]], dict[str, Any]]],  # op_id -> merger
) -> dict[str, Any]:                                  # BatchRelabelResponse dict plus 'op_id'
```

1. `create_op` writes `state.json` with `status='writing'` **before** any item write. A refused
   create (read-only binding: 409 `read_only`; disk error: 503) writes nothing. No item write
   without a record.
2. Inside `write_lock(op_id)`: `occ_bulk_human_relabel(opensearch, ids, merger_for(op_id),
   writer_id=op.writer)` (unchanged helper, `src/services/curation/crop_bulk.py`).
3. `write_members(op_id, result['updated_ids'])`, then `save_op(status='written', count, conflicts,
   finished_at)`.
4. Return the existing response plus `op_id`. `updated_ids` stays for now (section 10).

A crash or exception in step 2 leaves `status='writing'` with the lock released: served
`interrupted`, undoable through the tag scan. No exception handling is added to make that happen.

Merger changes (the single class-label writer stays the single writer, see
`tests/test_class_label_single_writer.py`):

- `ItemLabel` (`src/services/curation/class_label.py`) gains `op_id: str | None = None`;
  `ItemLabel.human(...)` takes it.
- `class_label_fields`: for `source='human'` always sets `label_op_id: label.op_id` (so a single
  label clears a stale tag); for imports the key is not written.
- `class_label_update`: `record_class_snapshot(current, writer=label.writer, restorable=True,
  op_id=label.op_id)`.
- `human_move_class_update(..., op_id)` and `candidate_move_update(..., op_id)` record `op_id` on the
  snapshot and set `label_op_id`; in `candidate_move_update` the tag is set after the
  `dict.fromkeys(CLASS_STATE_FIELDS)` clear.
- `src/domain/history.py`: `record_class_snapshot(..., op_id: str | None = None)` stores
  `entry['op_id']`; `CLASS_STATE_FIELDS` gains `'label_op_id'`; new
  `AUDIT_VERDICT_FIELDS = (AUDIT_OUTCOME, AUDIT_HUMAN)` (from `src/domain/audit_math.py`) are recorded
  when `current_source.get(AUDIT_SAMPLE) is True` and restored by `restore_class_state` only from an
  entry that carries them (the `REVIEW_DISMISS_FIELDS` pattern). New pure helper
  `op_write_undone(history, op_id) -> bool`: an unlabel entry with that `op_id` exists after the op's
  restorable entry.
- Per-item undo records which op it cancelled: the shared restore (section 5.3) sets the unlabel
  entry's `op_id` to the cancelled snapshot's `op_id`. That is what lets an op undo count an item
  undone one by one as `already_undone` instead of `kept_changed`.
- Mapping: `'label_op_id': {'type': 'keyword'}` in the items body
  (`src/clients/curation_opensearch/bodies_core.py`, beside the class fields), and an additive
  migration `ensure_items_label_op_field` in `ensure_overlay_fields.py` (pattern of
  `ensure_items_request_id_field`) registered in `_migrations()` of
  `src/services/curation/index_bootstrap.py`.
- Combine: `'label_op_id'` joins `_SOURCE_NUMBERED_FIELDS` in
  `src/services/projects/combine/copy_docs.py` (a source project's op means nothing in the target;
  `class_id_history` is already dropped there).
- `label_op_id` in `CLASS_STATE_FIELDS` also puts it under the human guard of `occ_upsert_bulk`
  (`_HUMAN_GUARD_OWNED` in `src/clients/occ_upsert.py`): re-ingest never clears the tag of a
  human-owned row. Automated class writers cannot reach a tagged label or class-move item (it is
  human-locked); a candidate-move item is not locked, and section 11 covers it.

Fingerprint per kind (`OpFingerprint`, stored in the record):

| Kind | `class_name` | `class_source` | `class_validated` | `cluster_id` |
|---|---|---|---|---|
| `label` | registry name written | `human` (`HUMAN_CLASS_SOURCE`) | true | not compared |
| `move` to a class cluster | target class name | `human_move` | true | not compared |
| `move` to a candidate cluster | not compared | not compared | false | target id |

## 5. Undo write path

### 5.1 Flow (`src/services/curation/label_ops/undo.py`)

```python
async def undo_label_op(
    opensearch: Any,
    record: LabelOpRecord,
    *,
    dry_run: bool,
    registry_name: Callable[[int], str | None],   # class id -> current registry name, from RegistryDep
) -> LabelOpUndoResult
```

1. Members: `read_members(op_id)`; if absent (interrupted op or a lost `members.json`),
   `scan_items(opensearch, {'term': {'label_op_id': op_id}}, index=items_index(),
   includes=['crop_id'], max_docs=MAX_BULK_SELECTION + 1)` (`src/services/curation/reprocess_targets.py`)
   and `members_known=False`. More than `MAX_BULK_SELECTION` hits is an invariant breach: refuse
   (500, logged), nothing written.
2. Dry run: pages of `mget_crops(..., source_includes=UNDO_READ_FIELDS)`, `undo_outcome` per doc,
   counts. Nothing written, no lock, no status change.
3. Real run: `occ_bulk_rounds(opensearch, index=items_index(), ids=members, prepare=...,
   on_written=..., on_failed=..., max_retries=5, refresh='wait_for',
   source_includes=UNDO_READ_FIELDS)`. `prepare(doc_id, doc)` computes `undo_outcome`; for `restore`
   it returns `class_undo_update(source)` (5.3), otherwise it records the outcome in a per-id dict
   (overwritten on a retry round, never incremented) and returns `None`. `on_written` marks
   `restored`; `on_failed` marks `conflict`.
4. Counts come from the per-id dict, so `matched = restored + already_undone + kept_changed +
   kept_holdout + kept_stale_class + conflicts` holds exactly (a test asserts it).

`UNDO_READ_FIELDS = ['crop_id', 'label_op_id', 'class_id_history', 'class_excluded', 'excluded_at',
'test_holdout', AUDIT_SAMPLE, *CLASS_STATE_FIELDS, *REVIEW_DISMISS_FIELDS, *AUDIT_VERDICT_FIELDS]`:
never embeddings or boxes, so a 500-doc page stays small.

### 5.2 The decision (`src/domain/label_ops.py`, pure, imports only `src/domain` and `src/utils`)

```python
def undo_outcome(source: dict[str, Any] | None, fp: OpFingerprint, *,
                 registry_name: Callable[[int], str | None]) -> UndoOutcome
```

In order, first match wins:

1. `source is None` (deleted): `KEPT_CHANGED`.
2. `source['label_op_id'] != fp.op_id`: `ALREADY_UNDONE` if `op_write_undone(history, fp.op_id)`,
   else `KEPT_CHANGED`.
3. `source['test_holdout']`: `KEPT_HOLDOUT`.
4. `snap = find_undo_snapshot(history)`; `snap is None`, `snap['op_id'] != fp.op_id` or
   `snap['writer'] != fp.writer`: `KEPT_CHANGED` (a later human write sits on top; automated entries
   are skipped by the stack walker, so step 5 catches them).
5. Excluded after the op (`class_excluded` and not (`excluded_at` present and `excluded_at < fp.at`)):
   `KEPT_CHANGED`. An item excluded before the op is restored with the excluded parking of 5.3.
6. Fingerprint mismatch (table in section 4; `class_name` compared normalized; this is what catches
   `class_merge` and `registry_reclassify`, whose `record_class_history` can dedupe away its own
   entry): `KEPT_CHANGED`.
7. `snap['class_id'] is not None` and
   `normalize_class_name(registry_name(snap['class_id']) or '') != normalize_class_name(snap['class_name'] or '')`:
   `KEPT_STALE_CLASS`.
8. `RESTORE`.

Also in this module: `LABEL_OP_ID_RE`, `new_label_op_id(now)`, `LabelOpKind`, `OpFingerprint`,
`UndoOutcome` (a `(str, Enum)` with the same `noqa: UP042` reason the repo uses), and
`current_query(fp) -> dict` (the query-expressible part of the decision, for the list counts: term
`label_op_id`, the fingerprint terms, `must_not` holdout, and `must_not` an exclusion that rule 5
keeps: `class_excluded` with `excluded_at` missing or at or after `fp.at`). Rules 4 and 7 are not
expressible; in practice a tagged item always has the op on top of its stack (every human class
writer resets or restores the tag), so `current` overcounts only items whose restore target class was
renamed or deleted.

### 5.3 The restore (`src/services/curation/class_undo.py`, new, shared)

```python
def class_undo_update(current: dict[str, Any]) -> dict[str, Any]:
    snap = find_undo_snapshot(current.get('class_id_history'))
    restored = restore_class_state(snap)
    if current.get('class_excluded'):
        restored = park_restored_state_while_excluded(restored)
    history = record_class_snapshot(current, writer=HUMAN_UNLABEL_WRITER, restorable=False,
                                    op_id=(snap or {}).get('op_id'))
    return {**restored, 'class_id_history': history, 'updated_at': now_iso()}
```

`_undo_merger` in `label_undo.py` calls it (its `NothingToUndoError` check stays in the router), so
per-item undo, `undo_batch` and op undo write byte-identical updates for the same document. A domain
test pins that equality.

### 5.4 Routes and their checks (`src/routers/curation/label_ops.py`, new)

Undo route flow:

1. `read_op(op_id)`: `LABEL_OP_ID_RE` first (a `../` or any other shape is 404 before a path is
   joined), then the bound project's `label_ops` dir; missing or pruned: 404 `label_op_not_found`.
2. Dry run skips locks. A real run takes `undo_lock()` non-blocking (busy: 409 `label_op_busy`)
   and `write_lock(op_id)` non-blocking (busy: 409 `label_op_in_progress`).
3. Stored `undone` and the `current_query` count is 0: 409 `already_undone`, nothing written (dry
   run answers the same 409, so the UI learns it before confirming).
4. `save_op(status='undoing')` only after the members are known (an OpenSearch failure before that
   leaves the record untouched), run `undo_label_op`, then `save_op(status='undone',
   undo_passes=[...newest 10])`, log `label_op_undone` with the counts (no ids).

## 6. Idempotency, resumability, concurrency

### 6.1 Idempotent by state, not by bookkeeping

Each decision reads the item itself. A restored item no longer carries the tag and its history shows
the op's write cancelled, so it classifies `already_undone` on any later pass. Re-running an undo
after any failure is always safe and restores exactly the remainder.

### 6.2 Resumable at 20k and beyond

- Crash during the op write: served `interrupted`; the tag scan recovers the members.
- Crash during an undo: served `undo_interrupted`; a new POST resumes (the flock died with the
  process).
- OCC conflicts that outlast the retries stay tagged; `current` stays above 0 and the op stays
  undoable; the response lists them (`conflict_ids`, at most 100, sorted) and counts all of them.
- Size: undo reads from the stored op, so no request carries ids and no 5,000 cap applies. The bound
  is the op's own size (`MAX_BULK_SELECTION`).

### 6.3 Second undo

`already_undone` (409) exactly when an undo pass completed before and no item is left that a pass
could restore (the `current_query` count). Undoing a later op can bring items back (section 2, point
3); then a further pass is allowed and restores only those.

### 6.4 Concurrency limits

| Path | Bound |
|---|---|
| Op undo | at most one undo per project at a time across all API workers (`undo.lock`); inside it, one `_mget` or one `_bulk` in flight, pages of `OCC_BULK_PAGE_SIZE` (500), `mget_crops` chunks of 256 |
| Op undo against its own write | refused while the op is writing (`write.lock`) |
| `POST /crops/label/undo_batch` | moves onto `occ_bulk_rounds` (same paging; no per-item `refresh='wait_for'`), 5,000-id cap kept |
| `POST /crops/discard_batch` | moves onto `occ_update_bulk` (its merger is already pure) |
| `POST /crops/region/undo_batch` | `asyncio.Semaphore(32)` around `undo_edit` (edit history has no bulk merger yet) |
| Dry run | read-only, same paging, no lock |
| Op write | unchanged (`occ_bulk_human_relabel`, already paged); ops and undos may run at the same time, OCC plus the fingerprint decide per item |

The plan's "bounded concurrency (32)" becomes "one paged bulk in flight": the per-item path was the
cost (a `wait_for` refresh per id), and the bulk loop is the bounded form the codebase already uses
for human bulk writes.

## 7. API and wire models

New wire models in `src/schemas/curation/label_ops.py` (pydantic `BaseModel`; payloads are small, so
`response_model=` is used rather than CQ1's TypedDict-through-`responses=`):

```python
class LabelOpUndoPass(BaseModel):
    at: str
    restored: int; already_undone: int; kept_changed: int
    kept_holdout: int; kept_stale_class: int; conflicts: int

class LabelOp(BaseModel):
    op_id: str
    kind: Literal['label', 'move']
    at: str
    finished_at: str | None
    class_name: str | None                  # the name written; class ids are never served
    target_cluster_id: int | None
    target_kind: Literal['class', 'candidate'] | None
    label_source: str | None
    target: Literal['crop_ids', 'selection']
    selected: int
    count: int | None                       # written; null while writing or interrupted
    conflicts: int | None
    current: int                            # items still carrying this op's label (current_query)
    status: Literal['writing', 'interrupted', 'written', 'undoing', 'undo_interrupted', 'undone']
    undone: bool                            # at least one undo pass completed
    undoable: bool                          # status not writing/undoing and current > 0
    last_undo: LabelOpUndoPass | None

class LabelOpList(BaseModel):
    items: list[LabelOp]                    # newest first
    limit: int
    retention: int                          # LABEL_OPS_KEEP (200)

class LabelOpUndoRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dry_run: bool = True                    # like the loop rollback: writing needs an explicit false

class LabelOpUndoResponse(BaseModel):
    op_id: str
    dry_run: bool
    status: str                             # served status after the call
    members_known: bool                     # false: members came from the tag scan
    matched: int                            # = restored + already_undone + kept_* + conflicts
    restored: int                           # restored, or that a dry run would restore
    already_undone: int
    kept_changed: int
    kept_holdout: int
    kept_stale_class: int
    conflicts: int
    conflict_ids: list[str]                 # at most 100, sorted

class CropUndoBatchResponse(BaseModel):     # POST /crops/label/undo_batch, shape unchanged
    items: list[dict[str, Any]]             # wire items (the shared serializer)
    undone: int
    nothing_to_undo: list[str]
    conflicts: list[str]
    not_found: list[str]
```

Routes:

| Route | Change |
|---|---|
| `{P}/label_ops` (GET) | New. Query `limit` 1 to 50 (default 20; 51 is a 422), `undoable` (bool, optional: only undoable ops, so the UI takes `items[0]` without filtering). Reads at most 200 small files and runs one `filters` aggregation (one filter per op on the page) for `current`. Unreadable record: skipped and logged |
| `{P}/label_ops/{op_id}/undo` (POST) | New. Body `LabelOpUndoRequest`. 200 `LabelOpUndoResponse`; 404 `label_op_not_found`; 409 `already_undone`, `label_op_busy`, `label_op_in_progress` |
| `PUT /crops/batch_label`, `POST /crops/move` | `BatchRelabelResponse` (`src/routers/curation/_selection_write_models.py`) gains `op_id: str \| None` (null for a dry run or an empty selection, where no op is recorded) |
| `POST /crops/label/undo_batch` | `response_model=CropUndoBatchResponse`; leaves `LEGACY_DICT` in `tests/test_contract_typed_responses.py` in the same commit |
| `GET /crops/{crop_id}/history` | entries gain `op_id` and `label_op_id` (`_HISTORY_KEYS` in `label_undo.py`); still on `LEGACY_DICT` |

The four error codes join `ErrorCode` in `src/schemas/curation/config_common.py` and are raised with
`rejection(...)` (`src/services/rejections.py`). Every route or model change runs `make contracts`
and commits `contracts/` in the same commit; `docs/design/curation_api_contract.md` gets a "Label
operations" section and `docs/opensearch_schema_design.md` the `label_op_id` row.

## 8. Audit and history interplay

- **Class history.** The op write entry and the undo entry both carry `op_id`; the history route
  shows who did what and which op it belonged to. Per-item undo and op undo share one restore
  function and one stack, so neither can re-apply a step the other already undid.
- **Accuracy audit.** A bulk label on audit-sampled items stamps a verdict; undo now restores the
  pre-write verdict (usually null), so the item returns to the audit queue instead of keeping a
  verdict for a label that no longer exists. This also fixes the same stale verdict on per-item undo.
- **Class write guard.** The op write and the undo both change `class_state_token`
  (`src/services/curation/class_write_guard.py`: history length and last entry), so an automated
  writer that read the item before either skips it. No change needed.
- **Edit history** (`edit_history`, region and VLM-dismiss undo) is untouched: ops are class writes.
- **Loop rollback** (`as_left` checks `history[-1]`, not the stack): an item the loop propagated, then
  labelled by an op, then restored by undoing the op, is back in the loop's state but its newest
  entry is the unlabel entry, so a later loop rollback keeps it as `kept_changed`. Fail-safe, noted
  for the L lane (section 11).
- **Logs.** `label_op_created` (op id, kind, selected), `label_op_written` (count, conflicts),
  `label_op_undone` (all counts). No ids, no class names beyond the op's own.

## 9. Failure modes

| Failure | Effect | Handling |
|---|---|---|
| Record create fails (disk, read-only binding) | nothing written | 503, or 409 `read_only`, before any item write |
| Process dies during the op write | partial write, stored `writing` | served `interrupted`, undo by tag scan, `members_known=false` |
| `members.json` write fails after the bulk | record lacks members | tag scan fallback, error logged |
| Process dies during an undo | partial restore | served `undo_interrupted`; re-POST resumes (6.1) |
| OCC conflict outlasts 5 retries | item kept | `conflicts` and `conflict_ids`; still tagged, op still undoable |
| `_bulk` transport error | page retried (`retry_transport=True`), then conflict | as above |
| OpenSearch down before or during the scan | nothing written | 503, record unchanged |
| Two undos in one project, any workers | second refused | 409 `label_op_busy` |
| Undo while the op is still writing | refused | 409 `label_op_in_progress` |
| Malformed, unknown, pruned or other-project op id | nothing read outside the bound project's dir | 404 `label_op_not_found` |
| Class renamed, merged or deleted since | no resurrected name | `kept_stale_class` or `kept_changed` |
| Item frozen in the holdout, excluded after the op, relabelled, deleted | kept | `kept_holdout` / `kept_changed` |
| Tag scan finds more than 20,000 | invariant breach | 500, logged, nothing written |

## 10. What UI3 and UI4 consume

- **UI3 bulk label with undo.** Unchanged dry run (`SelectionDryRunResponse`) and write; the toast's
  Undo uses the served `op_id`. "Operations" popover: `{P}/label_ops` (rows rendered verbatim:
  `class_name`, `count`, `current`, `status`, `last_undo`). Undo: POST with the default dry run,
  `Modal.svelte` shows the served `restored`, `kept_changed`, `kept_holdout`, `kept_stale_class`,
  `already_undone`, then POST `{"dry_run": false}`; refusals (`already_undone`, `label_op_busy`,
  `label_op_in_progress`, `label_op_not_found`) shown with the served message. After an undo the UI
  reloads the cluster list and detail (no restored items are returned; 20,000 would be about 30 MB).
  A 504 from the proxy is not a failure of the undo: the UI re-reads `{P}/label_ops` and shows the
  served status. UI3 adds the `label_ops/` location with a 600 s read timeout to
  `frontend/nginx.conf`.
  UI3 deletes the client-held undo for label and move (`undoStore.recordWrites` on those paths);
  whether `updated_ids` then has a reader decides if UI3 removes it from `BatchRelabelResponse` in
  the same merge (no dead fields).
- **UI4 keyboard.** `cluster.undo_op`: GET `{P}/label_ops?undoable=true&limit=1`, then the same
  dry-run-confirm-write sequence on `items[0].op_id`; an empty list shows the served empty state.
- Contract pins: `$contracts` for `LabelOp`, `LabelOpList`, `LabelOpUndoResponse`, and the
  `op_id` key on `BatchRelabelResponse`.

## 11. Risks

- Candidate-move ops leave items unvalidated and unlocked. A later VLM or cluster write that moves
  the item fails the fingerprint (kept). One that changes the class but not the cluster is reverted
  by the undo, as per-item undo already does today: a human undo outranks a machine label.
- The loop rollback interplay of section 8 (kept, not lost). If the L lane wants such items back, its
  `as_left` should use the stack walker; out of CQ3's files.
- `members.json` up to 0.9 MB per op, 200 ops: worst case about 180 MB per project on the state
  volume. Typical ops are far smaller; D-CQ3-5 can lower `LABEL_OPS_KEEP`.
- The undo is synchronous. Cropwright's general API location in `frontend/nginx.conf` has a 120 s
  `proxy_read_timeout` (the same limit a 20,000-item batch label already lives under). The 20,000 bar
  is therefore 90 s (section 14), and UI3 adds a regex location for `label_ops/` with a 600 s read
  timeout, as the ingest location has. If the browser still sees a 504, the undo keeps running in the
  API (`apiFetch` sends a write once and never retries it), the list shows `undoing`, and a later POST
  after it finishes is either a 409 or a resume. If the bar is missed, the follow-up is a background
  job around the same service function, not a different algorithm.
- The registry check (5.2 step 7) is new behaviour for the op undo only; per-item undo still restores
  a stale name until D-CQ3-3 says otherwise.

## 12. Owner decisions

| Id | Question | Recommendation |
|---|---|---|
| G7 | Undo restores per item and keeps later edits (section 2) | yes |
| D-CQ3-1 | `undoable` while any item still carries the op's label, and a further pass allowed when undoing a later op brings items back (replaces the plan's "false once any item changed" and its unconditional second-undo 409) | yes: the plan's rule makes a 20,000-item op unundoable after one edit |
| D-CQ3-2 | A bulk undo never touches an item frozen in the test holdout, even one the op itself labelled; per-item undo stays available | yes |
| D-CQ3-3 | A restore target whose class was renamed, merged or deleted is `kept_stale_class`. Apply the same check to per-item undo in this wave (409 with a new code) | yes for op undo; per-item in this wave only with a yes |
| D-CQ3-4 | Op records as files under `project_state_dir`, not an OpenSearch index | yes |
| D-CQ3-5 | Retention: newest 200 ops per project, no time expiry; a pruned op answers 404 (items keep their labels, per-item undo works) | yes |
| D-CQ3-6 | Scope: ops for batch label and move (both forms). Discard and exclude keep their own undo (`undo_batch`, `batch_unexclude`) | yes; folding discard in is a later wave if wanted |
| D-CQ3-7 | Undo request defaults to `dry_run: true` | yes, like the loop rollback |
| D-CQ3-8 | Synchronous undo (up to 20,000 items, bar 90 s) with a 600 s proxy location for `label_ops/`, rather than a background job now | yes; a job only if the bar is missed |

## 13. Tests that must fail first

All offline, query-capable fake (`tests/curation/query_fakes.py`), `project_state_dir` pointed at
`tmp_path` with `monkeypatch`. Write them, watch them fail (import errors on missing modules count),
then implement.

`tests/curation/test_label_ops_domain.py` (pure):

1. `test_op_id_format_and_regex` (`lop_...` round trip; `../x`, empty, uppercase hex refused).
2. `test_undo_outcome_decision_table` (parametrized, one row per rule of 5.2, plus the rule order).
3. `test_op_write_undone_follows_the_stack`.
4. `test_class_undo_update_matches_per_item_undo` (same doc, byte-identical update from
   `class_undo_update` and the per-item merger).
5. `test_snapshot_records_and_restore_returns_the_tag_and_audit_verdict`.
6. `test_current_query_matches_undo_outcome_restore` (query-capable fake: the query selects exactly
   the docs `undo_outcome` calls `restore`, on a fixture with excluded-before, excluded-after,
holdout, relabelled and merged items and no renamed class).

`tests/curation/test_label_op_undo.py` (routes):

7. `test_batch_label_returns_an_op_id_and_tags_every_written_item` (and records `op_id` on the
   newest history entry; dry run and empty selection return `op_id` null and create no record).
8. `test_exact_restore_of_a_12000_item_selection_op` (12,000 items by filter: every item's
   `CLASS_STATE_FIELDS` plus audit verdict equal the pre-op state; also proves the 5,000 cap is gone).
9. `test_a_human_edit_after_the_op_is_kept_changed`.
10. `test_ops_undo_in_any_order` (A then B, undo A, undo B, undo A again: final state equals the
    pre-A state, and each response's counts match section 2 point 3).
11. `test_items_undone_one_by_one_count_as_already_undone`.
12. `test_holdout_items_are_kept` (frozen after the op, and labelled while frozen).
13. `test_excluded_after_the_op_is_kept_and_before_the_op_is_restored_parked`.
14. `test_merged_or_renamed_class_is_kept` (merge: `kept_changed`; rename of the restore target:
    `kept_stale_class`; nothing written for either).
15. `test_second_undo_is_409_already_undone_and_writes_nothing` (real and dry run).
16. `test_occ_conflict_is_reported_not_swallowed` (fake bulk answers 409 for one id beyond the
    retries: `conflicts=1`, id listed, still tagged, a further POST restores it).
17. `test_dry_run_writes_nothing_and_predicts_the_real_run`.
18. `test_undo_keeps_one_request_in_flight` (instrumented fake: max concurrent `mget`/`bulk` is 1,
    no single-doc `update`, every bulk at most `OCC_BULK_PAGE_SIZE` actions).
19. `test_a_second_undo_in_the_project_is_409_busy` (hold `undo.lock`).
20. `test_undo_while_writing_is_409_and_an_interrupted_op_is_undoable` (hold `write.lock`; then a
    write that raises after half the pages: record `writing`, served `interrupted`, undo restores the
    written half, `members_known=false`).
21. `test_unknown_malformed_or_pruned_op_id_is_404` (includes a `..%2F` id; asserts no file outside
    the project's `label_ops` dir was opened).
22. `test_label_ops_list_newest_first_counts_and_limits` (`current`, `undoable`, `undoable=true`
    filter, `limit=51` is 422).
23. `test_retention_keeps_the_newest_200_and_never_prunes_a_live_op`.
24. `test_move_ops_undo` (class move and candidate move, each fingerprint).
25. `test_counts_partition_matched` (the invariant of 5.1 step 4 on every response above).
26. `test_undo_batch_is_typed_bulk_and_bounded` and `test_discard_batch_is_bulk`; in
    `tests/curation/test_region_undo.py`, `test_region_undo_batch_in_flight_at_most_32`.

Elsewhere:

27. `tests/curation/cross_project_leak/test_label_ops_isolation.py`: alpha's op is not listed by beta;
    beta POST undo with alpha's `op_id` is 404 and alpha's items are unchanged; only beta's indexes
    are reached; op files only under alpha's `project_state_dir`. Add `'op_id'` to `route_params` and
    the undo body to `route_bodies` (`tests/curation/cross_project_leak/_route_bodies.py`) so the
    every-route sweep covers both routes.
28. `tests/curation/test_items_mapping_explicit.py`: `label_op_id` is `keyword`; the migration is
    registered in `_migrations()`.
29. The combine copy test (`rg -l _SOURCE_NUMBERED_FIELDS tests/projects/combine`): `label_op_id` is
    dropped.
30. `tests/test_contract_typed_responses.py`: `POST /crops/label/undo_batch` removed from
    `LEGACY_DICT` (the ratchet fails on a stale entry).

### 13.1 Mutation list (each made alone, a named test must fail, then restored; record in the review)

| Mutation | Must fail |
|---|---|
| `history[-1]` instead of `find_undo_snapshot` in `undo_outcome` | 10, 11 |
| `label_op_id` removed from `CLASS_STATE_FIELDS` | 5, 10 |
| class-name fingerprint check removed | 14 (merge row) |
| exclusion-after-op check removed | 13 |
| holdout check removed | 12 |
| registry stale check removed | 14 (rename row) |
| per-item undo stops recording the cancelled `op_id` | 11 |
| conflicts counted as restored | 16, 25 |
| dry run calls `bulk` | 17 |
| bulk loop replaced by `asyncio.gather` of `occ_update_one` | 18 |
| `undo.lock` not taken | 19 |
| record written after the bulk instead of before | 20 |
| op id joined to the path before the regex check | 21 |
| store dir built from the path slug instead of the bound project | 27 |
| `already_undone` check removed | 15 |
| audit verdict not restored | 5, 8 |

## 14. Measurable acceptance

Isolated stack as in CQ1 (throw-away API container with this branch's `src` mounted read-only, one
worker, the synthetic 100k project of `scripts/bench/cluster_list_probe.py`), never the default
compose project. New probe `scripts/bench/label_op_probe.py`: selects N items by a seeded random
filter selection, snapshots their class state and the project's `class_name` histogram straight from
OpenSearch, runs batch label (timed), relabels K = 50 of them one by one, runs the undo dry run and
the undo (timed), samples the API worker's RSS every 500 ms, then audits. Three repetitions per row,
load average recorded (rows above 16 marked noisy), raw JSON under
`$MAIN/artifacts_local/bench/v070/CQ3/<row>/`.

| Bar | Target |
|---|---|
| Undo of a 10,000-item op | under 60 s, 0 lost writes (an item that should be restored and is not) and 0 extra writes (any other item changed), checked per item on `CLASS_STATE_FIELDS` and audit verdict against the pre-op snapshot; the K edited items keep their edit; the project histogram equals the pre-op one plus the K edits |
| Undo of a 20,000-item op | under 90 s (headroom under the 120 s proxy limit), same audit; dry run under 30 s |
| API worker peak RSS during the 20,000 undo | at most 200 MB above the pre-undo level |
| Batch label of 20,000 | within 1.1x of the same call on `main` in the same session (the record and the tag add two small file writes and one field) |
| `{P}/label_ops` with 200 retained ops at 100k items | p95 at most 100 ms (30 calls, 3 repetitions) |
| Counts | `restored = N - K`, `kept_changed = K`, `conflicts = 0`, invariant of 5.1 holds |

## 15. Implementation brief (for a fresh Sonnet agent)

Worktree `$MAIN/../OpenProcessor-wt-cq`, branch `feat/v070-cluster-outputs`; read this file, then
`CLAUDE.md`. Do not start before the owner answered G7 (and D-CQ3-1 to D-CQ3-8; use the
recommendation for any left unanswered and say so in the review). Commands from the worktree root:
`$VPY -m pytest <touched tests> -q --no-cov`, `$VPY -c 'import src.main'`, `$VPC run --all-files`
before each commit; never `--no-verify`; no push. No wave ids in code comments, docstrings or test
file names (the process-id guard). Every file under 640 lines.

**Step 1: tag and history (no route change).** Tests 1 to 6, 28, 29 first, seen failing. Then:
`src/domain/history.py` (`op_id` on `record_class_snapshot`, `label_op_id` in `CLASS_STATE_FIELDS`,
`AUDIT_VERDICT_FIELDS`, `op_write_undone`), new `src/domain/label_ops.py` (5.2),
new `src/services/curation/class_undo.py` (5.3) and `label_undo.py`'s `_undo_merger` calling it,
`class_label.py` (`ItemLabel.op_id`, tag in `class_label_fields`, `op_id` on both move updates),
mapping and migration (section 4), combine field. Run `tests/curation` touched files plus
`tests/test_class_label_single_writer.py`, `tests/curation/test_crops_undo_exclude.py`,
`tests/curation/test_new_class_resolve.py`. Commit `feat(curation): tag human label writes with their operation and restore the audit verdict on undo`.

**Step 2: op store, recording and the list route.** Tests 7, 22, 23 and the list half of 27 first.
New package `src/services/curation/label_ops/` (`store.py`: `LabelOpRecord`, `label_ops_root`,
`create_op`, `read_op`, `save_op`, `list_ops`, `write_members`, `read_members`, `write_lock`,
`undo_lock`, `served_status`, `LABEL_OPS_KEEP`; `record.py`: `recorded_relabel`; `views.py`:
`current_counts`, `label_op_wire`), `src/schemas/curation/label_ops.py`, the four error codes,
`crops_write.py` (`batch_label_crops`, `move_crops` through `recorded_relabel`),
`BatchRelabelResponse.op_id`, new router `src/routers/curation/label_ops.py` with the list route,
imported in `src/routers/curation/__init__.py`. `make contracts`; from `frontend/` run
`npx vitest run src/lib/contract` and, if a pinned key map fails on the added `op_id`, update only that
type or map (as CQ2 did). Commit `feat(curation): record bulk label and move operations and list them`.

**Step 3: undo.** Tests 8 to 21, 24 to 27, 30 first. `src/services/curation/label_ops/undo.py`
(5.1), the undo route (5.4), `undo_crop_labels` onto `occ_bulk_rounds` with `CropUndoBatchResponse`,
`discard_crops` onto `occ_update_bulk`, the semaphore in `undo_crop_regions`, `_HISTORY_KEYS`,
`LEGACY_DICT` entry removed, `make contracts`. Docs: "Label operations" section in
`docs/design/curation_api_contract.md`, `label_op_id` in `docs/opensearch_schema_design.md`,
`CHANGELOG.md` `[Unreleased]` (Added: label operations; Changed: `undo_batch` typed and bulk; Fixed:
unbounded undo and discard concurrency, stale audit verdict after undo),
`$VPY scripts/docs/check_docs_vs_code.py`. Run the mutation list of 13.1 and record each result.
Commit `feat(curation): undo a label operation by id, bounded and resumable`.

**Step 4: measure and review.** Write `scripts/bench/label_op_probe.py` (a small test like CQ1's probe
tests if it has pure helpers), run section 14 on an isolated stack (`docker compose -p <name>`,
preflight first; no GPU needed), write `docs/reviews/cq3_review.md` in the template of the plan's
section 12 (bars, raw-data paths, red-first tests with the commit they failed at, the mutation table,
decisions taken). Commit `docs(reviews): add the CQ3 owner review package`.
