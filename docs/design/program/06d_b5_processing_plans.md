# B5d: request-driven processing plans (#67 P0, P2-P6, faces on resident frames)

| Field | Value |
|---|---|
| Status | plan, not started (P1 of the source plan is done) |
| Release | v0.7.0 (inside the release; R-final runs after B5; see `06_b5_overview.md` section 2) |
| Depends on | 5c merged (`06c_b5_crops_ocr_faces.md`: OCR contract settled); 5b's one decode helper (`06b_b5_decode_once.md` 5b-D1); `ref-b5a` |
| Decisions applied | #67 owner decision 2026-10-31 (design first; route work after #220 PR 1, which is merged); `faces_ocr_resident_crops_plan.md` J5 (face alignment on resident frames moves here, P5); C1 recommendation BLS, not ensembles, still measured (M1) |
| Open decisions | `processing_plans_plan.md` section 12 Q1-Q9 (present at 5d kick-off; see section 3) |
| Source IDs | former master plan (rev 4, in git history) 7.5 row 5d, 11 (#17/#18 leftover: #67 phase 2+); #67; `docs/design/processing_plans_plan.md` (the detail reference: sections 3-10); #220 step 4 |
| Verified | Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read-only `rg`, `gh issue view`; nothing run) |
| Consolidated | Consolidated 2026-11-03 |

## 1. Goal

A request names the branches it wants (`detect`, `embed_frame`, `embed_crops`, `faces`,
`face_embed`, `ocr`, `near_duplicates`, `region`); unrequested branches make no Triton call and do
no host work; decode runs once and feeds every branch; the default plan of each route reproduces
today's response. Faces and OCR for photos run as `gpu_image_prep` branches on the resident frame,
including GPU face alignment.

Source of truth for the design: `docs/design/processing_plans_plan.md`. This file sequences it
into tracks, fixes the gates to the B5 protocol and gives the briefs. Do not duplicate its design;
read sections 3-9 there before coding.

## 2. What is done, what remains

| Phase | State | Evidence |
|---|---|---|
| design doc | done (PR #426) | `processing_plans_plan.md` |
| P1 pure domain module, wire models, tests | **done**: `src/domain/processing_plan.py`, `src/schemas/processing_plan.py`, `tests/test_processing_plan.py`; no route uses them | `rg -l processing_plan src tests` |
| #220 PR 1 (P0's precondition) | **done** (`27ebeb45`) | `06c_b5_crops_ocr_faces.md` 2 |
| P0 M1 (does an ensemble skip unrequested steps) and per-plan baselines | open | plan 5.4, 10 |
| P2 `/analyze` and `/analyze/batch` on the CPU executor | open | plan 10 |
| P3 core `/ingest` family on the executor | open | plan 10 |
| P4 curation `policy_plan`, request narrowing, `gpu_image_prep` `embed_frame` flag and `selected` mask | open | plan 10 |
| P5 `faces`, `face_embed`, `ocr` as `gpu_image_prep` branches, GPU face alignment | open | plan 10 |
| P6 docs, per-plan table, close #67 | open | plan 10 |

## 3. Owner questions to answer at kick-off (from the source plan, recommendation first)

| ID | Question | Recommendation | Needed by |
|---|---|---|---|
| Q1 | remove the legacy selection flags (`enable_ocr`, `include_embedding`, `include_face_embeddings` on `/analyze`; `enable_*`, `detect_near_duplicates` on core `/ingest`) in the same change | yes (no shims) | P2 |
| Q2 | one decoder for every branch changes face and OCR inputs (OpenCV today) and needs a new parity reference | accept after the gates pass on set A (**public only**; the source plan's "and the private set" is dropped by the owner's public-data rule) | P2 |
| Q3 | curation per-request plans: narrowing only | yes; widening past the policy is a 422 | P4 |
| Q4 | `selected` on the GPU route: compiled criteria resolved by name in the API from the same detector's labels | yes, with a parity test against `ingest_policy._mask` | P4 |
| Q5 | one branch fails | 200 with `status: error` and `null` output; 502 only if every branch failed | P2 |
| Q6 | missing dependency | refuse with `missing_dependency` | P2 |
| Q7 | core `/ingest` without `embed_frame` | store no global vector; `near_duplicates` requires `embed_frame` | P3 |
| Q8 | named presets | not in v1 | none |
| Q9 | reprocess scopes as plans | decide after P4 | after P4 |

## 4. Branch, worktrees, tracks

Integration branch `int/b5d-processing-plans` from main after 5c merged. Tracks `b5d/<track>` in
`<repo>-wt-b5d-<track>`:

| Order | Track | Phases | Size | Notes |
|---|---|---|---|---|
| 1 | `p0` | P0 (runs: M1 and per-plan baselines) | S | coordinator or gate agent, GPU 0 |
| 2 | `analyze` | P2 then P3 (two commits groups, one lane) | L | P3 starts only after P2's tests are green in the lane |
| 3 | `curation` | P4 | M | independent of P2 in files; rebases on `analyze` if both touch `src/schemas/processing_plan.py` |
| 4 | `gpu` | P5 | L | based on the integration head after `analyze` merged |
| 5 | `docs` | P6 | S | last |

Typed responses (#465, owner decision 2026-11-04): the `analyze` and `curation` tracks give every
operation of the routers they rewrite a `response_model=` (audit BA-6), so #465 closes with the 5d
PR; the other routers are typed by the W5a-W5c lanes of [00_master_plan.md](00_master_plan.md).

Wire changes (`make contracts` in the same commit): `PlanReport` in route responses, the removed
legacy flags, `faces_enabled` removed from the analyze responses (#379), curation request narrowing.
Tracks that change wire models merge last in their wave (risk M19 in `11_risks_cleanup_host.md`).

## 5. Tasks

**5d-P0 C1 and per-plan baselines (no product code).**
- M1 (C1): throwaway ensemble `c1_probe` (not committed) fanning one UINT8 input to the detector and
  `mobileclip2_s2_image_encoder`; 100 requests asking only for detector outputs; read
  `nv_inference_count` per composing model from the Triton metrics port before and after. Record
  the answer in `triton_pipeline_optimization_plan.md` section 11 and close C1. The BLS
  recommendation stands either way (plans need conditions and per-detection fan-out).
- Baselines on today's code where expressible (`/faces/recognize`, `/detect`, `/analyze`) on set A
  and set B at 1 and 16 clients, 3 repetitions (M-5). Today's published comparison: `/analyze` 55.5
  img/s and `/faces/recognize` 179.3 img/s at 16 clients on COCO (source plan section 2; re-measure,
  do not carry forward).

**5d-P2 `/analyze` on the CPU executor.** Size L. Detail: source plan 5.1, 9.2-9.5.
- `ImageViews` built from 5b's single decode helper; split model clients; per-branch outcomes;
  `null` semantics; `faces_enabled` deleted; legacy flags removed (Q1); `infer_yolo_clip_cpu`
  callers moved and the method deleted; per-branch metrics (source plan section 8).
- Red: one skip-proof test per branch (no Triton call and no host work for an unrequested branch;
  source plan 9.2); decode once (9.3); wire tests (9.4). Control: unchanged behaviour for existing
  requests without a plan (9.5, P-json against `ref-b5a` minus timing and ids).
- Gates: M3 (default plan within 3 % images/s of 5d-P0, p99 not worse than 10 %); M4 (faces-only
  within 10 % of `/faces/recognize` and at least 2x the full `/analyze` row; #67 acceptance);
  P-face and P-ocr parity (Q2).

**5d-P3 core `/ingest` family on the executor.** Size M. Zero global vector removed (Q7);
`near_duplicates` dependency. Red: skip proofs; Q7 test. Gate: M3.

**5d-P4 curation: the policy is the plan.** Size M. Detail: source plan 3.5, 6.1, 6.2.
- `policy_plan` maps `all | selected | lazy | none` exactly (one function); request narrowing on
  ingest routes, bulk runs and sources (Q3); `gpu_image_prep` `embed_frame` flag and the `selected`
  mask (Q4); `ensure_loaded` per resolved branch so an unrequested branch never loads its model
  under `OP_LOAD_PROFILE=ingest`.
- Red: the existing curation characterization tests stay green (proof that ingest is unchanged);
  a widening request is a 422; the `selected` mask parity test against `ingest_policy._mask`;
  `gpu_image_prep` offline tests (source plan 9.6). Gate: M3 on the GPU route with policy `all`,
  plus a `none` and a `selected` row.

**5d-P5 faces, face_embed and OCR as GPU branches, with face alignment on the resident frame.** Size L.
- Detail: source plan 5.2, 6.3. Branch flags on `gpu_image_prep` for photos on `/analyze` and core
  `/ingest`; the OCR branch consumes the 5c OCR contract as a view of the shared frame (no second
  decode or conversion).
- Face alignment today: host `cv2.warpAffine` in `src/utils/face_align.py::align_face` (Umeyama
  similarity transform), after a CPU decode in `src/clients/fast_face_client.py`. Library first:
  evaluate CV-CUDA `warp_affine` (batched, https://cvcuda.github.io/CV-CUDA/) on the resident frame
  against the host path on the same landmarks; custom kernels only if CV-CUDA fails the gate.
- Constraint: `gpu_image_prep` is decoupled, so its BLS calls stay synchronous per worker thread
  (async BLS is not supported in decoupled mode; python_backend docs).
- Gates: ArcFace cosine p01 >= 0.995 and P-face 1.0 against `ref-b5a` (the #220 face acceptance,
  moved here); P-ocr >= 98 %; M2 per plan with the **15 % stop rule** (the BLS arm is adopted per
  plan only if it is not more than 15 % slower than the API arm; else that plan stays on the CPU
  route); B5 benchmark rule (> 2x spread) for any default change.

**5d-P6 docs and close.** `README.md` API table, `docs/CURATION.md` ingest section, docs-site
ingest pages, `docs/PERFORMANCE.md` per-plan table, the source plan's status to implemented;
`scripts/docs/check_docs_vs_code.py` 0 problems; close #67 (and the #220 faces item it absorbed).

## 6. Measurement protocol

`06_b5_overview.md` M-1..M-10 and section 6 gates; plus the source plan's M1-M4 as written there,
with the public-only change to Q2. Live tests under `docker/test/compose.yml` with its own `-p`
name (source plan 9.7), GPU 0 (GPU 2 when agreed).

## 7. Gate (5d PR)

- T0 (targeted red/green plus static checks on the touched files) per task; one offline sweep on
  the merged integration head (pre-commit `--all-files`, suite `-n 8`, contracts-check, docs check,
  CHANGELOG rules, frontend set only if `frontend/` changed).
- No separate live-harness run for 5d: the live harness runs only at B1, end of B2 and B4
  (`00a_decision_register.md` section 8.1); the skip proofs and route parity run in the B5 GPU window instead
  (skip proofs live, parity of each route's default plan: P-box, P-item, P-pe,
  P-clip, P-face, P-ocr against `ref-b5a`).
- Benchmarks M2-M4 recorded with host load.

## 8. Done when

- every route runs only its resolved plan; one skip-proof test per branch passes;
- decode once per image on every planned route (stage counters);
- M3 holds for every default plan; M4 met; P5 gates met or the plan kept on the CPU route by the
  15 % rule, recorded;
- docs updated; CI green; PR merged with a merge commit; #67 closed;
- the R-3 docs removal for 5d is done in the closing PR (`processing_plans_plan.md` and
  `triton_pipeline_optimization_plan.md`; table in `06_b5_overview.md` section 10.1); this file,
  `06_b5_overview.md` and the other 07 files are deleted after the keep-lists are moved
  (`00a_decision_register.md` "Plan lifecycle"), with their rows removed from `00a_decision_register.md` and
  `docs/design/README.md`.

### What to keep after merge

| Fact | Target |
|---|---|
| processing-plan vocabulary (branches, per-route defaults, resolved plan, skip proofs, "not run" outcome) | `docs/ARCHITECTURE.md` and the API docs; `docs/design/curation_api_contract.md` for the curation narrowing |
| M1-M4 results, P5 face-alignment gates and the 15 % rule outcome | `docs/PERFORMANCE.md` |
| removed legacy flags, `faces_enabled` removal, input-decoder change | CHANGELOG ("Breaking", "Changed"), regenerated `contracts/` |

## 9. Docs, CHANGELOG, upgrade impact, rollback

- CHANGELOG "Breaking": legacy selection flags removed (Q1), `faces_enabled` removed from analyze
  responses; "Added": processing plans on the image routes; "Changed": face and OCR inputs from the
  single decoder (Q2).
- Contracts regenerate per wire change; `docs/design/curation_api_contract.md` updated for the
  curation narrowing.
- Upgrade: `gpu_image_prep` `MODEL_VERSION` bump for P4/P5 (the seed sync replaces the installed
  model); no stored data migrates (source plan 10); core `/ingest` documents without `embed_frame`
  carry no global vector (new documents only).
- Rollback: each track reverts alone; P5 branches keep a backend switch per plan until the 15 %
  rule and the gates pass.

## 10. Risks

| Risk | Mitigation |
|---|---|
| a default plan changes an existing response | P-json and the characterization tests (9.5) |
| BLS slower than API orchestration for some plans | M2 with the 15 % stop rule, per plan |
| face crops move with the decoder change (Q2) | new parity reference after the gates; ArcFace p01 >= 0.995 |
| on-demand model loading under the ingest profile surprises a caller | `ensure_loaded` per resolved branch; an unloadable model is an `error` outcome, tested |
| contract churn across tracks | wire-changing tracks merge last; `make contracts` per commit |

## 11. Lane briefs (ready to paste)

Common block: as in `04_b3_ultralytics_one_version.md` section 17, with worktree
`<repo>-wt-b5d-<track>`, branch `b5d/<track>`, file `docs/design/program/06d_b5_processing_plans.md`
and `docs/design/processing_plans_plan.md` (read sections 3-10 first), plus: "Benchmarks and parity
per `06_b5_overview.md` against `ref-b5a`. Stack `docker compose -p op-b5d-bench`, GPU 0 only
unless the coordinator says GPU 2 is agreed. Project slugs start with `b5d-`. Run `make contracts`
in the same commit as any wire change."

- **`p0`**: "Run 5d-P0. Do not commit the `c1_probe` ensemble. Report M1's counts and the baseline
  table."
- **`analyze`**: "Do 5d-P2 (skip-proof red tests first, one per branch), then 5d-P3. Apply the
  owner's answers to Q1, Q2, Q5, Q6, Q7 as the coordinator forwards them."
- **`curation`**: "Do 5d-P4. Keep every curation characterization test green unchanged; they are
  the proof."
- **`gpu`**: "Do 5d-P5. Evaluate CV-CUDA `warp_affine` before writing an alignment kernel and put
  the comparison in the PR. Apply the 15 % rule per plan."
- **`docs`**: "Do 5d-P6 after the coordinator confirms the gate results."

## Research amendments (2026-11-04, the research integration pass)

| Where | Change |
|---|---|
| 5d-P5 | the CV-CUDA `warp_affine` evaluation is `gpu:PR-5`; uint8 SCRFD and ArcFace inputs with normalization in the graph, and SCRFD anchor decode and NMS in the graph, are adopted only on `PR-5`'s pass |
| 5d-P2 | GPU S-21: a counter `op_plan_branch_total{route, branch, outcome}` (`ran`, `not_run`, `error`); the resolved plan is stored in `jobs.params.plan` and `plan_sha256` for the ingest and reprocess kinds (GPU S-9) |
| `near_duplicates` branch | returns candidates (`near_duplicate_candidates`: id and cosine), not groups: grouping is the single-writer `near_dup_core_group` job of W4 lane ND-C (research near-duplicates, pending RD-44) |
| Q1 | the legacy fields removed in the same change include the core ingest `near_duplicate_threshold` form field (already removed by ND-C2 in W4; Q1 confirms nothing re-adds it) |
