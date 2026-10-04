# Label confirmation: detector vs VLM vs human ground truth

Status: plan only (no code changed). Base: `main` at 5df30815. Audience: a fresh
implementation agent. All line numbers are from that commit.

## 0. Trigger and short answer

On a project such as `sample-coco-2k-v3`, 11,103 of 14,023 labelled crops show
`class_source='vlm'`, 0 are human, 2,920 are unlabelled. The owner asks whether
(a) the detector needs human ground truth and (b) the VLM should run on every crop.

Short answer, from the code:

1. The VLM dominates because of two defaults that combine: the default ingest
   policy leaves every detection an **unlabeled proposal** (no class), and the
   always-on `curation-vlm-worker --continuous` VLM-labels every unlabeled,
   embedded, unvalidated crop. Nothing limits it by class, confidence band,
   cluster representative, sample size or budget.
2. VLM labels are **never trainable or holdout-eligible without a human**: export
   and holdout read `class_validated=true` (holdout also `class_source='human'`),
   and no VLM path sets `class_validated`. The one non-human validator is
   `auto_promote`, and it only validates *classifier*-sourced labels.
3. The detector's own class is **not stored as a class** in the default mode (it is
   only `proposal_name`), so there is no stored detector-vs-VLM comparison and no
   review tab for "VLM disagrees with detector".
4. Nothing measures detector or VLM accuracy against human ground truth. That is
   the real gap.
5. Pre-0.4.0: no code change is required for correctness. One docs paragraph and a
   one-line UI wording note are the smallest safe change. The `vlm_scope` knob,
   audit sampling and detector-disagreement tab are 0.5.0.

## 1. What the VLM runs on today (file:line)

Three VLM class writers exist. Only the first two are in the default stack.

| Writer | Trigger | Scope |
|---|---|---|
| `curation-vlm-worker --continuous` (`docker-compose.yml:338-362`, profile `curation`; `scripts/curation/vlm_worker.py:99-176`) | Always running when the `curation` profile is up; polls forever for every active, unpaused project | Every embedded, non-validated crop except: classifier-labelled at conf >= 0.80 (`DEFAULT_CLASSIFIER_CONF_SKIP`, :78-84), already `vlm`/`classifier_vlm_agreement`/`cluster_majority_agreement`/`vlm_unmatched`/`vlm_new_class_pending` (:149-162), recent empty answer (:128). No class, confidence, representative, sample or budget limit. Only per-project pause (`pipeline_paused.flag`, `_project_pause.py`) and GPU-arbiter sentinel stop it |
| `POST /pipeline/auto_label/start?run_vlm=true` (`pipeline_start.py:60`, `pipeline.py:96,353-385`, `autolabel/selection.py:35-76`) | **Opt-in**: `run_vlm` defaults to `false` (`pipeline.py:96`; stage reports "skipped", `selection.py:95`) | Same global-sweep predicate as above plus 24 h skip of items the region worker already verified. Knobs: `class_id`, `cluster_id`, `max_vlm_crops` (cap, 0 = all, `pipeline.py:80`), `classifier_confidence_skip_vlm` (default 0.80, `pipeline_start.py:58`), `vlm_batch_size`, item filter |
| `POST /vlm/label_cluster/{id}` (`pipeline_control.py:67-98`) | Operator action | Every unvalidated, non-holdout, non-excluded member of one cluster (`selection.py:52-58`) |

So the auto-label pipeline run does NOT use the VLM by default; the continuous
worker does. The docs say as much without warning of the cost
(`docs-site/docs/operations/curation-workflow.mdx` section 5: "or let
`curation-vlm-worker` suggest classes continuously").

Why almost all crops qualify: `DetectFilter.class_resolution` defaults to
`'proposal'` (`ingest_policy.py:48-53`), and the primary profile defaults to
`assigns_class=false` (`ingest_profiles.py:6-11`). Result: ingest writes
`class_source='<primary>_proposal'`, no `class_id`. The classifier-confidence skip
needs a `*_model` source (`ingest_class_sources.py:60-71`), so it never fires. The
VLM is the only classifier. With `class_resolution='by_name'` a detector hit with
conf >= 0.80 gets a `*_model` class and the VLM skips it.

Segmenter gating tiers (`scripts/curation/worker/crop_gate.py`, tier 1
`parent_classes`, tier 2 VLM visibility, tier 3 learned hit rate) gate the
**region/segmenter** stage, not item class labelling. They are not a lever for
this question (tier 2 does spend VLM calls per item, but only with a region
profile active).

Ingest `embedding` policy (`all|selected|lazy`, `ingest_policy.py:70-86`) is an
indirect lever: the VLM only sees embedded crops (`embedded_clause()` in both
selectors). `lazy`/`selected` therefore cap VLM scope today, but by embedding
state, which is the wrong abstraction for a labelling budget.

## 2. Detector class vs VLM class: storage, comparison, preservation

- Default (`proposal`): detector label kept only as `proposal_name`
  (`item_doc.py:197-202`, "diagnostic only", survives later overrides) and
  `confidence` (detector score; the VLM writes `vlm_confidence` = high/medium/low,
  a separate field, `vlm_class_attempt.py:147-154`). No `detector_class_id`.
- `by_name` / secondary classifier: detector class is in `class_id`/`class_name`
  with `class_source='<name>_model'`. A later VLM write overwrites `class_id`,
  `class_name`, `class_source='vlm'` (`vlm_class_attempt.py:170-176`) and appends
  the previous class to `class_id_history` (`label_batch_write.py:55-67`,
  `with_class_snapshot`, `history.py:record_class_history`; capped at 32 entries
  for unvalidated items, dedupe on id+source). So the detector's class is
  recoverable from history but not queryable as a field, and the overwrite
  happens with no human in the loop.
- `detector_class_id/_name/_confidence` exist only in the legacy
  `labels_confirmed` mapping (`curation_opensearch.py:585-598`); nothing writes
  them.
- Review surfacing: `mismatches` tab = `class_source == 'vlm_unmatched'` only,
  i.e. VLM answer not in registry (`review_queries.py:359-364`). The comment there
  mentions classifier-vs-VLM disagreement but the query does not implement it.
  `vlm_low_conf` = VLM's own confidence medium/low (:365-373).
  `model_disagreements` = validated crops where a *trained probe model* disagrees
  with the human label (:474+), not the detector. `classifier_blind_spots` /
  `primary_low_conf` cover missed detections. **No tab shows "detector class !=
  VLM class".**
- Human override: a human label is locked against every automated writer
  (`class_write_guard.py:65-68`, `ClassWriteGuard`, OCC skip in
  `vlm.py:285-300`), and undo restores the snapshot.

## 3. What counts as ground truth today

| Use | Rule | Evidence |
|---|---|---|
| YOLO export / training | `class_validated=true`, non-excluded, non-dismissed, has box and registry class id | `export_readiness.py:59-80` |
| Holdout freeze | `class_validated=true AND class_source='human'`, sha1 per class, 5/class floor, 422 on empty | `holdout.py:84-101`, `review.py:526-596`, `dataset_thresholds.py:18` |
| VLM label (`class_source='vlm'`) | Never validated: comment "WITHOUT auto-validation", `pipeline.py:468-474`; `class_validated` is set only by human/import writers (`class_label.py:121,201`) and `auto_promote` | grep `class_validated': True` |
| `auto_promote` | Validates classifier-sourced members of high-purity clusters (`auto_promote.py:264-266,331-333`), stamps `class_source='cluster_majority_agreement'`; trainable, NOT holdout-eligible, NOT human. Off by default in `auto_label` (`pipeline.py:312-321`) but `curation-cluster-refresh` re-runs it. Does not touch VLM labels | |
| Dashboard | `by_human` counts only human-prefixed sources, not the validated flag (`stats.py:175-190`) | |

Conclusion: VLM labels cannot reach training or holdout without a human validate
action; that is correct and needs no change. Gaps:

1. No warning anywhere that an unvalidated majority means the export is small
   (the export only reports "0 validated" at zero, `export_readiness.py:54`).
2. `auto_promote`-validated labels train without any human check, and detector
   accuracy is never measured. If the detector's labels are wrong in a cluster,
   purity agrees with itself.
3. Nothing estimates how accurate the detector or the VLM is, so "VLM vs detector"
   is a guess. Human validation of VLM labels in review is both labelling and an
   implicit audit, but its agreement rate is never computed.

## 4. Options

| Option | What | Pros | Cons |
|---|---|---|---|
| A. Status quo (VLM on all) | Today | Zero setup, every crop gets a suggestion | Cost scales with crops; detector signal ignored; a wrong VLM label is indistinguishable from a right one; no accuracy estimate |
| B. Detector default, VLM on uncertain only | `class_resolution='by_name'`, VLM only for conf < threshold, `*_low_conf`, unmatched, or detector/VLM disagreement | Cheapest; keeps detector class queryable; uses existing skip | Needs a detector whose label space maps to the registry; COCO-style open registries often do not |
| C. VLM on cluster representatives, propagate with human confirm | VLM labels top-K representatives per cluster; cluster members get a *suggested* class (not a write); one human confirm applies to the cluster | Order-of-magnitude fewer VLM calls; matches how the dashboard already works (cluster cards, representatives) | Impure clusters propagate errors unless purity-gated; needs a new suggestion field |
| D. Sampling-based human audit | Stratified random sample of machine-labelled crops goes to human review; compute per-class agreement + confusion matrix for detector and VLM; min sample size per class | The only way to get real accuracy numbers; also feeds holdout | Human time; small classes need a floor |
| E. Active-learning ordering | Order review by VLM-vs-detector disagreement, low `vlm_confidence`, low purity, cluster distance, probe entropy | Spends human time where it matters | Needs the stored detector class to compute disagreement |

### Recommendation

Adopt a layered design, shipped in this order:

1. **Per-project `vlm_scope`** (A -> B/C): `all | uncertain | representatives | off`,
   default `all` for 0.5.0 compatibility (keeps current behaviour) but the docs and
   Cropwright recommend `uncertain` or `representatives` once a project exceeds a
   size threshold. Semantics:
   - `off`: no worker/pipeline VLM class writes (explicit cluster/crop requests
     still work).
   - `uncertain`: unlabelled crops plus detector conf below `vlm_conf_max`
     (default 0.80), `*_low_conf`, or detector class != cluster class.
   - `representatives`: only the top `vlm_per_cluster` (default 5) members per
     cluster by `core_first` distance (representatives already exist,
     `/clusters`), plus `unassigned`.
   - Common: `vlm_max_crops_per_day` budget (0 = unlimited) enforced in the
     selector, and `vlm_sample_frac` (0..1) for a random cap.
2. **Preserve the detector class** (`detector_class_name`,
   `detector_class_id`, `detector_confidence`) at ingest whenever the detector
   supplied a label, independent of `class_resolution`, so disagreement is
   queryable. Reuse the legacy mapping names (already declared in
   `curation_opensearch.py:593-595`, so no new names). Never overwrite them.
3. **`detector_disagreements` review tab**: `class_source in vlm` and
   `detector_class_name != class_name`, sorted by `vlm_confidence` asc then
   `confidence` desc (hard cases first).
4. **Accuracy audit** (`POST /audit/start`, `GET /audit/report`): draws a
   stratified sample (per detector class, `min_per_class` default 30, global
   default 300) of machine-labelled, unvalidated, non-holdout crops into a review
   queue (`review_status`-style marker `audit_sample=true`). When a human
   validates, record agreement. Report: per-class precision of detector and VLM,
   confusion matrix, Wilson 95% interval, and a flag `insufficient_sample` below
   the floor. Audited crops that a human labelled are valid holdout candidates
   (they are `class_source='human'`).
5. **Gate `auto_promote`** behind an audit result: refuse (409) for a class whose
   audited detector precision is below `promote_min_precision` (default 0.95), or
   whose audit sample is below the floor, unless `force=true`.

Cost estimate. There is no sustained-VLM measurement in `docs/PERFORMANCE.md`
(its VLM line items are only the stage names, lines 352-427; the doc reports the
VLM workers idle during the ingest baseline, line 446). The only datapoint is the
owner's: about 23 s for 20 crops including startup, i.e. at most ~1 crop/s cold
and well above that when warm. At a deliberately conservative 1 crop/s, 11,103
crops is ~3.1 h; at 5 crops/s it is ~37 min. `representatives` at 5 per cluster on
a 100-cluster project is 500 calls (~8 min at 1/s). The implementation must
measure real warm throughput first (verification step V1) and put the measured
number in `PERFORMANCE.md`; do not publish the estimate as a benchmark. For a
hosted (external) VLM the cost is per call and the `vlm_max_crops_per_day` budget
is the main safeguard.

## 5. Release split

### Before v0.4.0 (smallest safe change)

No code change is needed for correctness: VLM labels are never trainable or
holdout-eligible unvalidated, and a human label is never overwritten. The
following docs-only edits are the recommended minimum; skip them only if the
release is frozen for docs:

1. `docs-site/docs/operations/curation-workflow.mdx` section 5: state that
   `curation-vlm-worker` labels **every** unlabelled embedded crop of every active
   project by default, that VLM labels are suggestions (`class_source='vlm'`,
   unvalidated) and are excluded from export and holdout until a human validates,
   and how to stop it today (`POST /pause`, `docker compose stop
   curation-vlm-worker`, or `auto_label` with `run_vlm=false`, which is already
   the default).
2. Same page: note that `ingest_policy.detect.class_resolution='by_name'` makes a
   confident (>= 0.80) detector class bypass the VLM.
3. `CHANGELOG.md` "Known limitations": no VLM scope or budget knob and no accuracy
   audit yet; both planned for 0.5.0 (link the issue).
4. Cropwright wording (frontend repo, no contract change): where it shows
   `class_source='vlm'` use "VLM suggestion (unvalidated)", never "labelled".

### 0.5.0

Items 1-5 of the recommendation, in that order, one PR each.

## 6. Implementation spec

### PR 1: `vlm_scope` policy

Files:
- `src/services/curation/ingest_policy.py`: no change (ingest policy is for
  ingest). Add a sibling model.
- New `src/services/curation/vlm_policy.py`: `VlmPolicyBody` (`scope: Literal['all','uncertain','representatives','off']='all'`, `conf_max=0.80`, `per_cluster=5`, `max_crops_per_day=0`, `sample_frac=1.0`), stored under the settings doc key `vlm_policy` using the exact optimistic-revision pattern of `ingest_policy_store.py` (`revision`, `PolicyConflictError`). Do not add it to `ingest_policy`.
- New `src/routers/curation/vlm_policy.py`: `GET/PUT /vlm/policy` (409 on stale revision), registered in `src/routers/curation/__init__.py`.
- `src/services/projects/clone_settings.py`: clone `vlm_policy` with the project.
- `src/services/curation/autolabel/selection.py`: `vlm_selection_query` takes the policy; add scope clauses (`uncertain`: must `bool.should` of no class / `confidence < conf_max` / `*_low_conf`; `representatives`: terms on a representatives id set from the existing representatives query; `off`: return match-none). `sample_frac` via `random_score` or id hash.
- `scripts/curation/vlm_worker.py`: `_build_pending_query` reads the same policy through the API (the worker is an HTTP client; add `GET /vlm/policy` call per project per poll, cached 30 s) and applies the same clauses. Daily budget: count `vlm_class_attempted_at >= today` per project via `_count`, stop fetching when reached. Keep the two selectors in one shared function to stop drift (they are duplicated today, `vlm_worker.py:99` vs `selection.py:35`; extract `vlm_pending_clauses(policy, classifier_skip_conf)` into `selection.py` and import from both).
- `contracts/`: regenerate the OpenAPI snapshot; add `VlmPolicy` schema.
- Docs: `docs-site/docs/guides/vlm-selection.mdx` (new "Scope and budget" section), `docs-site/docs/api-reference/curation.mdx`.

API delta (additive): `GET/PUT /curation/projects/{p}/vlm/policy`; `auto_label/start` honours the policy unless `vlm_scope` query override is given. Default `all` = unchanged behaviour.

Tests (`tests/` curation unit tests; watch each fail first):
- selector: each scope returns the expected clauses; `off` selects nothing; `representatives` limits to K per cluster; budget stops at N.
- worker and pipeline selectors are identical for the same policy (one parametrised test on the shared function).
- PUT with stale revision -> 409; clone carries the policy; human-validated crops never selected under any scope.

### PR 2: keep detector class

Files: `src/services/curation/item_doc.py` (write `detector_class_id/_name/_confidence` from the detection when present, never from the VLM), `ingest.py`/`ingest_detector.py` (populate), ensure `ensure_*` mapping adds the three fields to the items index (they are only in the `labels_confirmed` mapping now; additive, `curation_opensearch.py`), `wire.py` + `_item_models.py` (serve them). Backfill script: from `proposal_name` and `class_id_history[0]` where possible; optional.
Tests: ingest with `by_name` stores detector fields; VLM overwrite leaves them intact; human label leaves them intact; re-ingest does not clobber.

### PR 3: `detector_disagreements` tab

Files: `review_queries.py` (add to `KNOWN_TABS`, `TAB_LABELS`, `build_tab_query` branch with a script or `must` comparing `.keyword` fields; mind the text-fielddata note at :480-495), `_review_tab_models.py`, `review.py` catalog, docs `curation-workflow.mdx`. Contract: a new tab id in `GET /review/tabs` (additive).
Tests: seeded items with agree/disagree/validated; only unvalidated disagreements appear; tab counts match.

### PR 4: accuracy audit

Files: new `src/services/curation/audit.py` (stratified sampler: per detector class, `min_per_class`, sha1-deterministic like `holdout.py`), `src/routers/curation/audit.py` (`POST /audit/start`, `GET /audit/report`, `GET /audit/queue`), mapping `audit_sample`, `audit_batch_id`, `audit_outcome` (agree/detector_wrong/vlm_wrong/both_wrong), a hook in the human label writer (`class_label.py`, near :121/:201) that stamps `audit_outcome` when `audit_sample` is true. Report math in a pure function (Wilson interval, confusion matrix as `{detector_class: {human_class: n}}`), `insufficient_sample` flag.
Tests: sampler floors and determinism; report math against a hand-computed table; outcome stamping; audited human labels are holdout-eligible and unaudited VLM labels are not.

### PR 5: auto_promote gate

Files: `clustering/auto_promote.py`, `routers/curation/clusters.py:486` (409 `audit_required` / `audit_precision_low`, `force=true` bypass logged distinctly, mirroring the promote `force` precedent). Docs.
Tests: promote refused without audit; allowed above threshold; `force` bypasses.

## 7. Frontend (Cropwright) implications and contract changes

All additive; no removals, so old frontends keep working.

1. Project settings: VLM scope panel bound to `GET/PUT /vlm/policy` (scope radio, thresholds, per-day budget), with the estimated crop count from a new `GET /vlm/policy/preview` (count of crops the policy would select) and a cost line.
2. Item/Crop cards: render `detector_class_name` and `confidence` next to the label when `class_source='vlm'`; badge wording "VLM suggestion" vs "Human-confirmed" vs "Auto-validated" (`cluster_majority_agreement`). Dashboard: show unvalidated-by-source counts so "14k labelled" is not read as ground truth.
3. Review: new `detector_disagreements` tab (read label and description from `GET /review/tabs`, which already serves them).
4. Audit UI: audit queue view, report page with confusion matrix and `insufficient_sample` state.
5. Warnings: export/preflight page shows the validated-vs-total ratio and a warn row when validated < a floor.
6. Regenerate the frontend client types from the new OpenAPI snapshot; fix `cropwright.lock` pin.

## 8. Verification

- V1 (throughput): on the bound project, `POST /vlm/label_cluster/{id}` for a 200-crop cluster after warm-up; record crops/s and add a labelled row to `docs/PERFORMANCE.md`.
- V2: with policy `off`, up `curation-vlm-worker` for 2 minutes against a project with unlabelled crops; assert zero `class_source='vlm'` writes (`GET /stats`).
- V3: `representatives` with K=5: VLM call count <= 5 x clusters (worker log + `vlm_class_attempted_at` count).
- V4: validate one VLM label via `PUT /crops/{id}/label`; confirm `class_source='human'`, locked, and eligible for `POST /test_holdout/freeze`; confirm an unvalidated VLM crop is absent from `POST /export/yolo`.
- V5: run the audit on a seeded project, compare the report to a manual count.
- Always: `.venv/bin/pre-commit run --all-files`, `.venv/bin/python -m pytest tests/ -v`, contract diff check, docs-site build, and a real browser pass of any Cropwright change (screenshots desktop and narrow).
