# P2b: Curate (phase spec)

| Field | Value |
|---|---|
| Status | plan, 2026-11-13. Starts after the P2a-2 exit bar and the owner's answer to Q4 ([90](90_owner_questions.md)) |
| Master plan | [00_photo_library_plan.md](00_photo_library_plan.md); P2a specs [10_p2a1_organise.md](10_p2a1_organise.md) (query builder, projection drain, annotations) and [15_p2a2_people.md](15_p2a2_people.md) (the face-identity service, review band, forget) |
| Research input | [01_feature_research.md](01_feature_research.md) section 5 (excluding people), sections 3.3-3.5 (bursts, culling, face quality) |

Notation as in 10: `{L}` = `/curation/projects/{project}/library`. Wording rule: this file and every test,
fixture and UI string describe excluded subjects generically ("person A", "an excluded identity"); no real
name or description of any person appears anywhere.

---

## 1. Scope and non-goals

**In scope.** Exclusion rules by person, object class (by name), OCR text, date range, annual date (holiday
or anniversary), place and album, at three levels; high-recall identity capture (looser threshold,
context links, review queue, exemplar growth); inheritance by new photos; fail-closed handling of
unprocessed photos; per-rule coverage and audit; reset. Burst stacks with RAW+JPEG pairing, an explainable
top pick and "keep the top, trash the rest". Assisted culling with visible thresholds and a quality filter.

**Non-goals.** Deleting or moving any original; excluding from derived workspace projects that already
exist (Send to project leaves excluded photos out from now on; existing copies are the user's own
project); video (P3); a learned per-user aesthetic model beyond the top-pick weights.

**Needs from P1.** N1 and N7 (faces with quality, yaw, eyes open), N5 (OCR text), N6 (sharpness, exposure,
aesthetic). Needs from P2a-1 and P2a-2: query builder, projection drain, annotations and undo, the face-identity service and its review band.

**Exclusion and forget.** An exclusion keeps the person's embeddings so new photos are caught; "forget this
person" (P2a-2) deletes them and also deletes the person's rules. The rule editor says which to use.

---

## 2. Exclusion design

### 2.1 Levels and the surfaces they cover

| Surface | `show_less` | `block` | `hide` |
|---|---|---|---|
| Timeline, albums, smart albums, folders, places, map, search, facets | shown | shown | removed |
| People list | shown | shown | the person is hidden |
| A memory *about* the subject (P2c) | never | never | never |
| Memories, featured photos, slideshows, renders: photos containing the subject | group shots allowed; weight 0.2 for cover and key photos | removed, group shots too | removed |
| Send to project | included | left out (served count) | left out |
| Photo detail by direct id | served with `excluded_by` reasons | same | same; reachable only from Settings › Exclusions › Show excluded photos |

Levels follow Google's "show less / blocked" and Apple's "feature less / never feature" (01 section 5),
plus `hide`, which no vendor offers. Every level is reversible; deleting a rule restores everything on the
next request.

### 2.2 Rule kinds and their predicate

| `subject_kind` | `subject` | Matches photos where (image doc fields) |
|---|---|---|
| `person` | `{person_id}` | `faces.person_ids` or `faces.candidate_person_ids` contains it (the candidate band counts as a match: recall first) |
| `class` | `{class_name}` | `objects.class_names` contains it (name, never id) |
| `text` | `{text}` | `text.ocr` matches the phrase |
| `date_range` | `{from, to}` | `capture.taken_local` in range |
| `annual_date` | `{month, day, span_days}` or `{holiday: <holidays id>}` | `capture.taken_month_day` in the expanded set |
| `place` | `{country_code?, admin1?, locality?}` or `{bbox}` | `place.*` or `capture.location` |
| `album` | `{album_id}` | `lib.album_ids` contains it |

Photo-level human decisions override a rule both ways: `lib.excl_forced` (rule ids the user added by hand)
and `lib.excl_released` (rule ids the user released by hand). Excluded under rule R ⇔ (predicate(R) or R in
`excl_forced`) and R not in `excl_released`.

### 2.3 Where it is applied (query time)

`src/domain/library_visibility.py` (pure) compiles the enabled rules into two clause sets:
`view_clauses(rules)` for the browsing surfaces (hide-level only, plus trash/archive/hidden from P2a) and
`created_clauses(rules)` for created outputs (block and hide, and show-less as a weight). The one library
query builder (`src/services/library/query.py`, from P2a) calls `view_clauses` for every list; P2c's
eligibility calls `created_clauses`. Rules are read from Postgres once per request (one indexed query, no
in-process cache). A rule change applies on the next request; there is no re-apply job for views.

### 2.4 High recall for person rules

| Mechanism | Detail |
|---|---|
| Looser threshold | `t_exclude` (below `t_review`) is calibrated on the eval split so that face-visible recall meets the registered bar; stored with the face model id. Faces of excluded persons scoring at least `t_exclude` against the person's prototypes get the person in `candidate_person_ids` |
| Recall sweep job | `library_exclusion_sweep` on rule create and on demand: seeds = confirmed and assigned faces of the person (all if at most 200, else the 8 prototypes plus 192 sampled by quality); k-NN over the faces index (k 200 per seed, filtered `score ≥ t_exclude`), excluding cannot-linked faces; writes candidates with OCC; enqueues the photos in the projection drain |
| Inheritance at ingest | the P2a-2 assignment tick also scores each new face against persons with an enabled rule at `t_exclude`; a hit becomes a candidate in the same write as its assignment |
| Context link (Apple's within-moment idea, 01 section 1.2) | a face scoring in `[t_exclude − δ, t_exclude)` becomes a candidate when a confirmed face of the person is in a photo within ±10 min (same burst stack or same capture window); δ calibrated on the eval |
| No visible face (measured option, lane L-B5) | person-class detector boxes within ±10 min of a confirmed face of the person whose PE crop vector is close to that person's own body crops become photo-level candidates; ships only if it raises no-face recall on the eval without breaking the false-exclusion bar |
| Exemplar growth | confirming a candidate adds a must-link; prototypes refresh at the next grouping run, so later sweeps find more |
| Review queue | candidates for excluded persons first, highest score first; confirm (must-link, stays excluded) or reject (cannot-link, the photo returns unless another rule matches); photo-level candidates (no face) are decided with `excl_forced` / `excl_released` |

### 2.5 Fail closed

| State | Created outputs | Views |
|---|---|---|
| `faces.state` in (`pending`, `failed`) while any person rule is enabled | not eligible | eligible, unless a `hide`-level person rule is enabled and the owner's Q4 answer keeps the recommended "hide until processed" |
| Photo in `library_projection_pending` | not eligible | eligible (it is not yet in any list that depends on the pending fields) |
| `objects.projected_at` missing while a class rule is enabled | not eligible | as faces |
| A rule's last sweep failed | the rule's coverage shows `sweep_failed`; outputs still apply the predicate | same |

The served coverage (`{L}/exclusions/{rule_id}/coverage`) always says how many photos are waiting, so a
user is never told "done" while photos are still unprocessed.

### 2.6 Audit

`exclusion_audit` rows per sweep and per nightly check: counts excluded, candidates, reviewed, waiting for
faces, thresholds, face model. Counts only, never image ids or names, never logged with names.

---

## 3. Stacks (bursts and pairs)

| Concern | Design |
|---|---|
| Engine | the existing near-duplicate engine (`src/services/near_duplicates/`, `src/domain/near_dup.py`) gains a group kind: `near_dup_runs.group_kind` in (`duplicate`, `burst`); a burst run links image pairs whose capture times are within `burst_window_s` (default 10) and whose frame cosine is at least `burst_min_cosine` (default 0.90, calibrated on the triage set; bursts score 0.05-0.17 below the 0.98 duplicate cut per 01 section 3.3). One engine, two group kinds; decisions (`same`, `different`) apply to both |
| RAW + JPEG pairing | same folder, same file stem, capture within 2 s: a forced `same` decision with reason `raw_jpeg_pair`; the JPEG is the default top |
| Top pick | score per member = weighted sum of: face sharpness (Laplacian variance on face crops, else the subject or the frame), minimum `eyes_open` over faces with side at least 60 px, mean face `quality`, `signals.aesthetic`, exposure penalty, resolution, and a prior for favourite or rating. Default weights fitted by logistic regression (scikit-learn) on pairwise preferences of the public triage set; refit per library from the owner's own picks once at least 50 exist. Weights live in the `culling_profile` config document (revisioned, `src/services/config_store/`) |
| Explanation | each member's per-signal contribution is served; the UI shows "sharper, eyes open" style reasons from served labels |
| Projection | `stack.group_id`, `stack.is_top`, `stack.size` on images (the `near_dup_fields.py` projection pattern); human top picks in `stack_picks` win over automatic ones (lock rule) |
| Grid | `LibraryFilter.collapse_stacks` (default true): only tops and unstacked photos; the tile badge shows the size; opening a stack shows its members |
| Keep the top | trashes the other members in one library operation (undoable); never touches sources |

## 4. Assisted culling

| Concern | Design |
|---|---|
| Profile | config document kind `culling_profile`: thresholds `blur_max`, `eyes_closed_max`, `under_exposed`, `over_exposed`, `aesthetic_min`, `duplicate_of_top: bool`, top-pick weights; served schema like the near-dup profiles |
| Suggestions | per photo `pick`, `reject` or `none` with reasons and the measured values; never applied without the user |
| Apply | writes `lib.pick` through the P2a annotation path in one undoable operation; human flags are never overwritten |
| Quality filter | `LibraryFilter.quality: Literal['low','ok']`; a library setting `library.hide_low_quality` (default off: hiding by default risks losing photos) |

---

## 5. Tables and migrations

Revision `<next>_library_curate.py` (lane L-B1 writes all of it in its first commit, with the wire models of section 6 and the generated contracts):

| Table or change | Columns |
|---|---|
| `exclusion_rules` | `id uuid7`, `project_id`, `subject_kind` (`person/class/text/date_range/annual_date/place/album`), `subject jsonb` (`{person_id}`, `{class_name}`, `{text}`, `{from,to}`, `{month,day,span_days}` or `{holiday}`, `{country_code,admin1,locality}` or `{bbox}`, `{album_id}`), `level` (`show_less/block/hide`), `enabled bool`, `note null`, `revision`, `created_at`, `updated_at` |
| `exclusion_audit` | `id`, `project_id`, `rule_id`, `run_at`, `n_photos_excluded int`, `n_candidates int`, `n_reviewed int`, `n_pending_faces int`, `thresholds jsonb`, `face_model` (counts only, never image ids or names) |
| `exclusion_photo_decisions` | `project_id`, `rule_id`, `image_id text`, `decision` (`forced/released`), `created_at`; pk (`rule_id`,`image_id`) |
| `stack_picks` | `project_id`, `group_id uuid` (near-dup group), `image_id text`, `source` (`auto/human`), `explain jsonb`, `revision`; pk (`project_id`,`group_id`) |
| `near_dup_runs` (change) | `group_kind text not null default 'duplicate'` check in (`duplicate`,`burst`) |
| config store kinds | `culling_profile` added to the document kinds (`src/db/models/config.py` check list and `src/services/config_store/`) |

OpenSearch: `lib.excl_forced`, `lib.excl_released` (keyword[]) in `library_annotation_fields.py`; `stack.*`
in a new `library_stack_fields.py`.

---

## 6. Routes and wire models

| Route | Request | Response |
|---|---|---|
| `GET {L}/exclusions` | | `ExclusionRules {rules: list[ExclusionRule {rule_id, subject_kind, subject, subject_label, level, enabled, note, n_excluded, n_candidates, n_waiting, revision, updated_at}]}` (`subject_label` served: the person's name or "Unnamed person", the class name, the holiday name) |
| `POST {L}/exclusions` | `{subject_kind, subject, level, note?}` | `ExclusionRule`; a person rule starts a sweep job and returns `sweep_job_id` |
| `GET/PATCH/DELETE {L}/exclusions/{rule_id}` | patch `{level?, enabled?, note?, expected_revision}` | `ExclusionRule` |
| `POST {L}/exclusions/preview` | `{subject_kind, subject, level}` | `ExclusionPreview {by_surface: dict[str, int], n_candidates, n_waiting}` |
| `GET {L}/exclusions/{rule_id}/coverage` | | `ExclusionCoverage {n_excluded, n_candidates, n_reviewed, n_waiting_faces, last_sweep_at, last_sweep_state, thresholds, audit: list[AuditRow]}` |
| `POST {L}/exclusions/{rule_id}/sweep` | | `JobRef` |
| `GET {L}/exclusions/{rule_id}/candidates` | `cursor` | `ExclusionCandidates {photos: list[PhotoTile & {evidence: list[{kind: Literal['face','context','body'], score, face_id \| None}]}], next_cursor}` |
| `POST {L}/exclusions/{rule_id}/candidates/decide` | `{image_ids, decision: Literal['forced','released']}` | `LibraryOpResponse` |
| `POST {L}/exclusions/reset` | `{dry_run, confirm_count}` | `{n_rules_deleted}` |
| `GET {L}/review/faces` | `person_id?`, `cursor` | `Faces` (P2a model) with `rule_id` per face, excluded persons first |
| `GET {L}/stacks` | `LibraryFilter`, `cursor` | `Stacks {stacks: list[{group_id, size, top: PhotoTile, top_source}], next_cursor}` |
| `GET {L}/stacks/{group_id}` | | `Stack {group_id, members: list[PhotoTile & {rank, score, contributions: dict[str, float], reasons: list[str]}], top_image_id, top_source}` |
| `POST {L}/stacks/{group_id}/top` | `{image_id, expected_revision}` | `Stack` |
| `POST {L}/stacks/{group_id}/keep_top` | `{dry_run}` | `LibraryOpResponse` |
| `POST {L}/stacks` | `{image_ids}` (manual) | `Stack` (writes `same` decisions) |
| `DELETE {L}/stacks/{group_id}` | | unstack (`different` decisions) |
| `GET/PUT {L}/culling/profile` | put: profile body with `expected_revision` | `CullingProfile {thresholds, weights, revision, schema}` |
| `GET {L}/culling/suggestions` | `LibraryFilter`, `cursor` | `CullingSuggestions {photos: list[PhotoTile & {suggest, reasons: list[{label, value, threshold}]}], counts: {pick, reject, none}}` |
| `POST {L}/culling/apply` | `{selection, accept: Literal['pick','reject','both'], dry_run}` | `LibraryOpResponse` |

P2a routes change: every list adds `collapse_stacks` and `quality`; `PhotoDetail` adds
`excluded_by: list[{rule_id, level, reason_label}]`; `POST {L}/send_to_project` leaves out block and hide
photos and returns `n_left_out_by_rules`. `make contracts` in the same commit as each change.

---

## 7. Jobs and queues

| Job | Queue | Notes |
|---|---|---|
| `library_exclusion_sweep` | `library` | `exclusive_group='library_exclusion_sweep'`, `exclusive_scope='rule_id'` (one active sweep per rule); batched kNN; OCC writes; audit row |
| assignment tick (P2a-2, extended) | `periodic` | advisory `LIBRARY_PERSON_ASSIGN`; candidate scoring for persons with enabled rules |
| `library_exclusion_nightly` (daily tick, advisory `LIBRARY_EXCLUSION_NIGHTLY`; enqueues sweeps) | `periodic` | re-runs sweeps whose person changed since the last sweep; writes audit rows |
| near-dup run with `group_kind='burst'` | near-dup queue (existing) | incremental after ingest, like duplicate runs |
| `library_culling_refit` | `library` | refits top-pick weights from owner picks (at least 50) |

---

## 8. Cropwright components

| Piece | Path (new) | Reuses |
|---|---|---|
| Exclusions settings | `routes/p/[project]/library/exclusions/+page.svelte`, `lib/components/library/{ExclusionList,ExclusionEditor,ExclusionCoverage}.svelte` | `ui/DataTable.svelte`, `ui/Segmented.svelte` (level) |
| "Exclude…" entry points | person page menu, class chip menu, photo info panel, holiday picker | `ui/Menu.svelte` |
| Candidate review | `lib/components/library/ExclusionReview.svelte` (faces and photo-level evidence) | `FaceReview.svelte` (P2a), keyboard pattern of `components/review/**` |
| Excluded banner | `lib/components/library/ExcludedBanner.svelte` in the viewer | |
| Stacks | stack badge in `PhotoTile.svelte`, `lib/components/library/StackViewer.svelte` with per-signal reasons | `PhotoViewer.svelte` |
| Culling | `lib/components/library/CullingPanel.svelte` (thresholds from the served schema, suggestion counts, apply with dry run) | `ui/Drawer.svelte` |

UI copy states the limit plainly: "Photos where the person's face is not visible may still appear. Review
candidates to catch more."

---

## 9. Lanes, order and dependencies

```
wave 1:  L-B1 rules, compiler, surfaces || L-B3 stacks and culling backend || L-B4 eval harness, bars registered
wave 2:  L-B2 recall capture and review routes || L-BF1 exclusions and review UI || L-BF2 stacks and culling UI
wave 3:  L-B5 no-face body link (measured; may be dropped) || L-FIX
then:    the eval run against the registered bars, the owner walk
```

| Lane | Model, effort | Owns | Depends on |
|---|---|---|---|
| L-B1 | Sonnet, high (isolation of views, every surface) | the migration and wire models (first commit), `src/db/models/library_curate.py`, repos, `src/domain/library_visibility.py`, `src/services/library/exclusions.py` (rules, preview, coverage), `src/routers/curation/library_exclusions.py`, the `view_clauses` call in `src/services/library/query.py`, `PhotoDetail.excluded_by`, the send-to-project filter | P2a-2 |
| L-B2 | Sonnet, high (recall, lock rule) | `src/services/library/exclusion_sweep.py`, the assignment tick extension in `src/services/library/people/assign.py`, candidate and decision routes, `{L}/review/faces`, audit writer, nightly tick | L-B1 |
| L-B3 | Sonnet, high (near-dup engine) | `src/services/near_duplicates/**` (group kind), `src/domain/near_dup.py`, `src/services/library/{stacks,culling}.py`, `library_stack_fields.py`, stack and culling routes, `culling_profile` kind | P2a-2 |
| L-B4 | Sonnet, medium | `scripts/bench/library_eval_exclusion.py` (interleaves the Commons face set from P2a-2's L-PE into the 500k real-EXIF library by shifting each identity's photos into real events; bars, samples and Clopper-Pearson bounds in the header), `scripts/bench/library_eval_triage.py` | P2a-2 L-PE |
| L-BF1 | Sonnet, high | exclusions pages and components, review UI, banner | L-B1 merged (L-B2 routes stubbed from section 6) |
| L-BF2 | Sonnet, medium | stack badge, stack viewer, culling panel | L-B3 merged |
| L-B5 | Sonnet, high | `src/services/library/exclusion_body_link.py` behind the profile switch `body_link` | L-B2; ships only if its bar passes |

Verify commands (paste the output): L-B1 `.venv/bin/python -m pytest tests/domain/test_library_visibility.py
tests/library/test_exclusion_surfaces.py tests/library/test_send_to_project_exclusions.py -q --no-cov &&
.venv/bin/python scripts/codegen/generate_contracts.py --check`; L-B2 `... tests/library/test_exclusion_sweep.py
tests/library/test_exclusion_inherit.py`; L-B3 `... tests/curation/test_near_dup_burst.py tests/library/test_stack_top.py
tests/curation -k near_dup`; L-B4 `... tests/scripts/test_exclusion_eval.py`; L-B5 `... tests/library/test_body_link.py`;
frontend lanes `cd frontend && npm run check` plus their stubbed e2e files. Frontend lanes start after L-B1's
wire models merge.

Brief docs to read and cite: OpenSearch k-NN filtering and radial search (`min_score`), scikit-learn
`LogisticRegression`, OpenCV Laplacian (`cv2.Laplacian`) for sharpness, the near-dup engine module
docstrings, Python `holidays` (for `annual_date` holidays; MIT).

---

## 10. Tests to write red first

| Lane | Tests |
|---|---|
| L-B1 | `tests/domain/test_library_visibility.py`: each level × surface of section 2.1; forced and released; disabled rules ignored. `tests/library/test_exclusion_surfaces.py`: a route-table walk asserts every `{L}` list route builds its body through `query.build()` (so no surface can skip the predicate); a hide rule removes a photo from timeline, search, smart album, places, facets and map cells in one fixture. `tests/library/test_send_to_project_exclusions.py` |
| L-B2 | `tests/library/test_exclusion_sweep.py`: candidates written above `t_exclude`, cannot-linked faces skipped, idempotent; `tests/library/test_exclusion_inherit.py`: a newly ingested face of person A is excluded before its photo leaves the pending table (fail closed); reject returns the photo; confirm grows exemplars |
| L-B3 | `tests/curation/test_near_dup_burst.py`: window and cosine; duplicate runs unchanged (existing tests stay green); RAW+JPEG pairing; `tests/library/test_stack_top.py`: human top survives a rerun; keep-top is undoable and touches no source |
| L-B4 | the eval scripts refuse to score without their registered bars and minimum sample sizes in the header |
| L-BF1, L-BF2 | `e2e/stubbed/library/test_exclusions.py` (create with preview counts, level change, coverage waiting count shown), `test_exclusion_review.py` (confirm/reject keys), `test_stacks.py` (collapsed tiles, reasons rendered from served labels) |

Pending wave runs: the live harness ingests a Commons face-set sample into a library project, sets a
`block` and a `hide` rule, ingests 50 more photos of the person and checks every surface.

---

## 11. Eval (bars registered by L-B4 before any score)

Sets ([00](00_photo_library_plan.md) section 11.5): the **Commons face set** (CC-licensed identities with many
photos including non-frontal ones; built by P2a-2's L-PE; local only) interleaved into the 500k real-EXIF
library; **LFW** for threshold fitting on training identities only; CALFW or CPLFW only if a licence is found
on the source. PIPA is not used: its licence could not be verified from a source page. Excluded identities
are drawn from held-out identities (never used to fit `t_exclude`). Every bar: point estimate and one-sided
95 % Clopper-Pearson bound must both pass.

| Bar | Point | Bound | Minimum sample | Set |
|---|---|---|---|---|
| Recall, the identity's face side ≥ 40 px | ≥ 0.99 | lower ≥ 0.98 | 500 photos over ≥ 20 identities | Commons face set, held-out identities |
| Recall, all photos with a visible face of the identity | ≥ 0.95 | lower ≥ 0.935 | 1,000 photos | same |
| Recall, no visible face | reported | | | same; a bar only if L-B5 ships |
| False exclusion of other photos | ≤ 0.05 | upper ≤ 0.06 | 2,000 photos | the 500k library around the seeded identities |
| Excluded-identity photos in created outputs | 0 | | every seeded photo | eligibility function now; P2c memories later |
| Stack top-1 agreement | ≥ 0.60 | lower ≥ 0.56 | 500 series | photo triage benchmark if its terms allow, else the owner's own picks |

---

## 12. Acceptance walk (owner, live stack, 1600 and 430 px)

1. Pick a named public-set person; "Exclude…" → preview shows counts per surface and candidates; choose `block`.
2. Review candidates: confirm three, reject two; coverage counts move.
3. Change to `hide`: timeline, search, places, map and the person's People card drop the photos; the viewer of
   a direct link shows the banner.
4. Ingest 30 new photos of the person: none appears in any list; coverage shows them excluded.
5. Add a class rule (`show_less` on a class) and a date rule; reset all with the dry-run count.
6. Open a burst: collapsed tile with a count; the top pick's reasons read sensibly; keep the top; undo.
7. Culling on a day of photos: suggestions with values; apply rejects; undo one.
8. The eval report against section 11 bars, with load average and sample sizes.
