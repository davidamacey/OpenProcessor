# Request-driven processing plans: run only the requested models and branches (#67)

Status: open (P1 done; P0 and P2 to P6 are `program/06d_b5_processing_plans.md`, v0.7.0 B5d; remove
after B5d). P1 is the pure domain module, the wire models and their tests; no route uses them yet.
Owner decision 2026-11-03: the single v0.7.0 release includes this document (it was v0.9.0 in the
2026-10-31 text); route work (P2 onward)
starts after #220 PR 1 (the uint8 OCR input contract) has landed, and GPU face alignment is part of
P5 here, not of #220. The rest is not implemented. Issue: #67. Umbrella: #40. Depends on: #220 (and through it #40). Related: #219,
#52 (selective embedding, shipped), #231.

A fresh agent with no memory of the planning conversation must be able to implement this from the
file alone. Paths are relative to the repository root. File:line references were read on `main` at
`04cb44bc`, re-checked at `1fc87ef2` (section 2.1 and 5 lines updated), and will drift; re-find by symbol before editing. Settings this plan proposes are
written in lowercase (`plan_executor`) as proposals; any env name is chosen at implementation time
and added to `env.template` and `tests/test_env_surface.py` in the same change. Routes are written
with a method only where they exist today.

Read first, do not duplicate:

- `docs/design/triton_pipeline_optimization_plan.md`: decisions 1-9 (section 1), the open
  Triton checks C1-C7 (section 5.6, log in section 11), Wave 7 (`ingest_pipeline` BLS) and the
  15 % stop rule.
- `docs/design/gpu_decode_plan.md`: why the GPU decode is a Triton Python model
  (`gpu_image_prep`) and not a DALI ensemble (section 7.4), and the as-built A/B (section 13).
- `docs/design/generic_detector_and_selective_embedding_plan.md`: the embedding policy
  `all | selected | lazy | none` this plan must keep unchanged.
- `docs/design/sp_gpu_prep_concurrency.md`: the `gpu_image_prep` memory budget and stage counters.
- `docs-site/docs/operations/ingest-load-profile.mdx`: `OP_LOAD_PROFILE` and on-demand loading.

## 1. Summary

- Every image route runs a fixed chain today, whatever the caller wants. `/analyze` runs the
  detector, a whole-frame MobileCLIP vector, a MobileCLIP vector per detection and SCRFD plus
  ArcFace on every call, then drops what was not asked for, and decodes the same bytes up to five
  times. Measured on public COCO at 16 clients, `/analyze` runs at 55.5 images/s while
  `/faces/recognize` alone runs at 179.3 (section 2).
- A **processing plan** is a short, typed list of named branches from a closed vocabulary
  (`detect`, `embed_frame`, `embed_crops`, `faces`, `face_embed`, `ocr`, `near_duplicates`,
  `region`), with a dependency table and one condition kind (`if_detected` by class name). It is
  not a user-defined graph.
- The server resolves the plan against a per-route allowlist and refuses anything it cannot run
  exactly as written (422). Unrequested branches make no Triton call and do no host work. Decode
  runs once per image and feeds every branch.
- The default plan of each route reproduces today's response for a request that sends no plan.
  Curation ingest keeps the project's ingest policy as its plan. A request plan can only narrow it.
- **C1 recommendation: BLS, not ensembles.** Plans need conditions and per-detection fan-out, and an
  ensemble cannot express either, whatever the answer to C1. The GPU route extends the existing
  decoupled BLS `gpu_image_prep` with plan flags. The CPU route is the API's own orchestration of
  direct TensorRT calls. One resolved plan drives both. C1 is still measured (section 5.4) so the
  question is closed with data.
- Cost per plan becomes visible through per-branch counters and histograms, and the response
  reports which branches ran.

## 2. Problem: fixed branches per route, with measured cost

### 2.1 What each route runs today

| Route | What the code always runs | What the request can turn off | Evidence |
|---|---|---|---|
| `POST /analyze` | detector + whole-frame MobileCLIP + MobileCLIP per detection (`infer_yolo_clip_cpu`), SCRFD + alignment + ArcFace, OCR when `enable_ocr` (default true) | only OCR; `include_embedding` and `include_face_embeddings` only hide outputs | `src/services/analyze_pipeline.py:233-298`, `src/clients/triton_client.py:219-297` |
| `POST /analyze` with `include_embedding=true` | the per-detection MobileCLIP vectors are computed **twice**: once inside `infer_yolo_clip_cpu` (`embed_boxes_from_full_res`, `triton_client.py:282-286`, result unused by `/analyze`) and again by `_run_crop_embeddings` (`analyze_pipeline.py:180-200`), which decodes the file again with PIL (`crop_batch_tensor`, `:130`) | | |
| `POST /analyze/batch` | detector + MobileCLIP (both vectors) and SCRFD + ArcFace for every image even when `include_full_results=false` returns counts only; face confidence fixed at 0.5 | OCR (default off) | `analyze_pipeline.py:362-...` (`:388`) |
| `POST /ingest` (core) | `enable_detection` or `enable_clip` runs the whole `infer_yolo_clip_cpu` (detector, frame vector, crop vectors) either way; faces at fixed 0.5 | `enable_faces`, `enable_ocr`; `enable_clip=false` writes a 512-d **zero vector** as the global embedding and still runs near-duplicate grouping on it | `src/services/visual_search/ingest_single.py:24-56`, `:146-150` |
| Curation ingest (`POST /ingest/image`, `POST /ingest/batch`, `POST /ingest/upload`, bulk) | decode once, detector, crop JPEGs, blur; PE per the project's embedding policy | per project, not per request (`IngestPolicy`, `src/services/curation/ingest_policy.py`) | `src/services/curation/ingest.py` |
| Curation ingest, GPU decode route, policy `none` | the whole-frame PE vector is computed inside `gpu_image_prep` and then dropped | | `src/services/curation/ingest_index.py:226-240` (docstring says so) |
| Curation ingest, GPU decode route, policy `selected` | every detection's crop square is embedded (`embed_crops` is one boolean, true unless `lazy`/`none`); the policy mask is applied after the PE work | | `src/services/curation/ingest_batch.py:85`, `models/gpu_image_prep/1/model.py:288-318` |

Decodes of one image's bytes on `/analyze` with OCR and embeddings: PIL in `infer_yolo_clip_cpu`
(`triton_client.py:236`), OpenCV in the face client (`src/clients/fast_face_client.py:155`), the
OCR service's validation decode (`src/services/ocr_service.py:90`), OpenCV again in
`src/clients/triton_ocr.py:33`, and PIL again in `crop_batch_tensor`. That makes five decodes for
one image. Curation ingest already decodes once (shipped in 0.6.0, `gpu_decode_plan.md` section 13).

### 2.2 Measured cost (all from `docs/PERFORMANCE.md`)

Per endpoint, "0.6.0 Wave 0: full-pipeline throughput", "Probe 3: faces and OCR": 200 images of
set A (public COCO) and 100 photos of set B (16.1 MP mean), one RTX A6000:

| Endpoint | Set A, 16 clients: images/s | API CPU s/image | Set B, 16 clients: images/s | API CPU s/image |
|---|---:|---:|---:|---:|
| `analyze` (detector, faces, MobileCLIP; no OCR) | 55.5 | 0.116 | 22.8 | 0.789 |
| `faces_recognize` | 179.3 | 0.047 | 36.8 | 0.364 |
| `detect` | 133.2 | 0.050 | 33.7 | 0.376 |
| `embed_image` | 223.0 | 0.027 | 40.2 | 0.299 |
| `analyze_ocr` | 20.0 | 0.160 | 1.2 | 2.755 |
| `ocr_predict` | 20.2 | 0.079 | 1.2 | 2.139 |

Reading:

- A caller who wants only faces pays 3.2x the throughput on COCO (55.5 against 179.3) and 1.6x on
  photos. On photos the API CPU per image doubles (0.789 against 0.364 s), and that is the repeated
  decodes: one PIL decode of an 18 MP photo is 316.5 ms on one thread ("Probe 4", primitive table).
- MobileCLIP is the largest GPU item of `/analyze`: 11.4 ms of Triton execution per COCO image at
  16 clients (31.8 ms at 1 client), against 2.3 + 2.2 ms for ArcFace and SCRFD and 3.7 ms for the
  detector. `/analyze` spends that even when the response omits every embedding.
- OCR is bound by the single `ocr_pipeline` Python instance and by the FP32 full-frame input
  (765 ms per photo; #220 PR 1 sends uint8, a quarter of the bytes). That is #220's problem, but a plan that does not ask for OCR must never pay it.
- Curation ingest shows what skipping a branch is worth. In `gpu_decode_plan.md` section 13, detect
  only (policy `lazy`) ran at 44.0 against 35.0 photos/s with embedding on set B, and at 120.3
  against 28.6 frames/s on set A ("Measured ingest speed" and the per-image cost table in
  `docs/PERFORMANCE.md` explain why: PE needs 7.39 inferences per ingested image).

### 2.3 What 0.6.0 already made selectable (keep, do not re-implement)

- Curation embedding policy `all | selected | lazy | none` (per project, revisioned):
  `src/services/curation/ingest_policy.py`.
- Reprocess scopes `detect | open_vocab | region | vlm | embed` with `EmbedPart`
  `crop | frame | region`: `src/services/curation/reprocess_models.py:25-90`, `POST /reprocess`.
- Selectable VLM and segmenter assist runs: `POST /assist/runs`,
  `docs/design/selectable_assist_plan.md`.

No route has a general declaration of which models run, and there is no per-branch cost metric.

## 3. The processing-plan model

### 3.1 Vocabulary (closed)

Branches are names from one closed list. Shared work is not a branch. It is derived from the
branches that run.

| Branch | Models | Needs | Output it produces | Routes |
|---|---|---|---|---|
| `detect` | the route's object detector (core: `TritonModelConfig.YOLO_MODEL`; curation: the profile's or the project override's detector) | | boxes, scores, class names | all |
| `embed_frame` | core: `mobileclip2_s2_image_encoder`; curation: `pe_image_encoder` | | one whole-image vector | all |
| `embed_crops` | same encoder as `embed_frame` | `detect` | one vector per detection (curation: per policy-selected detection) | all |
| `faces` | `scrfd_10g_bnkps` | | face boxes, 5-point landmarks, score, quality (quality is box-based, `fast_face_client.py` `_compute_quality`) | `/analyze`, core `/ingest` |
| `face_embed` | `arcface_w600k_r50` (+ host alignment; GPU alignment is P5 here) | `faces` | one ArcFace vector per face | `/analyze`, core `/ingest` |
| `ocr` | `ocr_pipeline` (calls `paddleocr_det_trt`, `paddleocr_rec_trt`) | | text lines, quads, scores | `/analyze`, core `/ingest` |
| `near_duplicates` | none (OpenSearch k-NN on the frame vector) | `embed_frame` | duplicate group assignment | core `/ingest` |
| `region` | none at ingest; seeds `pending_detection` for the region worker when a region profile is active | `detect` | region queue seed | curation ingest |

Derived shared stages, run at most once per image and only when a branch needs them:

| Stage | Runs when | Today's code to reuse |
|---|---|---|
| hash (imohash, dedup) | the route stores (core `/ingest`, curation) | `ingest_single._imohash`, `ingest_helpers._imohash_bytes` |
| decode (EXIF upright, uint8) | any pixel branch runs | `src/services/curation/image_decode.py` (CPU), `gpu_image_prep` (GPU) |
| detector letterbox (built as uint8; the wire datatype is the model's own, FP32 or UINT8, via `images_input` and `get_model_input_datatype`, #428; the metadata read is made only when `detect` runs) | `detect` | `preprocess_yolo_cpu`, `gpu_image_prep._detect` |
| encoder square of the frame | `embed_frame` | `preprocess_clip_cpu` (core), `pe_preprocess.whole_frame_rgb_from_image` (curation) |
| crop squares | `embed_crops` (only for the selected detections) | `embed_boxes_from_full_res` / `center_crop_cpu`; `gpu_image_prep._embed` |
| face working copy (long side at most 1024) | `faces` | `fast_face_client.recognize` (`MAX_DIM`) |
| OCR input (long side 640-960, padded to 32) | `ocr` | `triton_ocr.infer_ocr` |
| blur sums, crop JPEGs | curation `detect` (items need them for review and the crop cache) | `crop_quality`, `write_crop_cache` |

Rules:

1. **No branch decodes bytes.** Every branch receives the one decoded frame (or a view derived
   from it). A static guard test enforces this (section 9.3).
2. **Derived views are memoized per image** and built from the full-resolution frame, never
   cascaded from another model's input (`gpu_decode_plan.md` R1: cascading wrecks crops on photos).
3. **VLM is not a branch.** VLM labeling is driven by the VLM policy, the worker and assist runs.
   Adding it here would be a second way to say the same thing.
4. **Reprocess keeps its scopes.** Mapping reprocess scopes onto plans is a later, separate
   decision (section 12, Q9). The vocabularies are kept aligned (`embed_frame` = `EmbedPart`
   `frame`, `embed_crops` = `crop`) so the mapping stays mechanical.

### 3.2 Conditions

One condition kind, and only on `faces` and `ocr`:

```json
{"branches": ["detect", "ocr"], "when": {"ocr": {"if_detected": ["car", "truck"]}}}
```

`if_detected` names detector classes **by name** (CLAUDE.md: class identity is the name). It is
matched with the same normalization as the ingest policy (`src/services/curation/name_match.py`).
It needs `detect` in the plan. When no detection at or above the request's detection confidence has
one of those names, the branch does not run and the response marks it `condition_false`.
`if_detected` on OCR is the issue's "OCR only if a region was found". OCR still reads the whole
frame. OCR on detection crops is a non-goal here (section 11).

### 3.3 Wire model

One model for every route, in `src/schemas/processing_plan.py`. The vocabulary and the dependency
and allowlist tables are pure data in `src/domain/processing_plan.py`. Domain imports only `config`
and `utils`, and schemas may import domain (`scripts/codegen/check_layering.py` rules 1 and 4).

```python
# src/domain/processing_plan.py (pure; no pydantic, no I/O)
Branch = Literal['detect', 'embed_frame', 'embed_crops', 'faces', 'face_embed', 'ocr',
                 'near_duplicates', 'region']
BRANCHES: tuple[Branch, ...]                    # canonical order, also the execution order
REQUIRES: Mapping[Branch, tuple[Branch, ...]]   # embed_crops->detect, face_embed->faces, ...
CONDITIONAL: frozenset[Branch]                  # {'faces', 'ocr'}
RouteKey = Literal['analyze', 'analyze_batch', 'ingest_core', 'curation_ingest']
ALLOWED: Mapping[RouteKey, frozenset[Branch]]
DEFAULT: Mapping[RouteKey, tuple[Branch, ...]]  # curation_ingest: derived from the policy, 3.4

@dataclass(frozen=True)
class ResolvedPlan:
    route: RouteKey
    branches: tuple[Branch, ...]                 # canonical order
    if_detected: Mapping[Branch, tuple[str, ...]]  # normalized class names
    def runs(self, b: Branch) -> bool: ...
    @property
    def key(self) -> str: ...                    # 'detect+faces', the metric label

def resolve(route: RouteKey, requested: PlanInput | None, *, known_classes: Collection[str]) -> ResolvedPlan
    # raises PlanError(reason, branch) - section 4
```

```python
# src/schemas/processing_plan.py
class BranchCondition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    if_detected: list[str] = Field(min_length=1, max_length=64)

class ProcessingPlan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    branches: list[Branch] = Field(min_length=1, max_length=len(BRANCHES))
    when: dict[Literal['faces', 'ocr'], BranchCondition] = Field(default_factory=dict)

class BranchOutcome(BaseModel):
    status: Literal['ok', 'error', 'condition_false']
    error: str | None = None
    ms: float

class PlanReport(BaseModel):          # echoed in every plan-aware response
    branches: list[Branch]            # what was resolved, canonical order
    outcomes: dict[Branch, BranchOutcome]
```

How a request declares it:

| Route | Carrier | Absent means |
|---|---|---|
| `POST /analyze`, `POST /analyze/batch` (multipart) | form field `plan`, a JSON string validated with `ProcessingPlan.model_validate_json`; an invalid value is a 422 in FastAPI's validation-error shape | the route's default plan |
| `POST /ingest`, `POST /ingest/batch`, `POST /ingest/directory` (core) | form field `plan` (multipart) or body field `plan` (JSON routes) | the default plan |
| Curation `POST /ingest/image`, `POST /ingest/batch`, `POST /ingest/upload`, `POST /ingest/objects`, `POST /ingest/bulk`, registered sources | body field `plan` (form field for upload), stored on the bulk run and source record so every batch of a run resolves the same plan | the project policy's plan (3.4) |

A plan travels with the request, never through a header or an environment variable. The `/v1`
twins get the same parameter because they mount the same routers (`src/routers/v1.py`).

### 3.4 Default plan per route

The default reproduces today's **response** for a request that sends no plan. Work whose output
the response never contained is not part of the default, so defaults get cheaper while responses
stay the same.

| Route | Today's request (no plan) returns | Default plan | Work removed from the default |
|---|---|---|---|
| `POST /analyze` | detections, faces without vectors, OCR (default `enable_ocr=true`) | `detect, faces, ocr` | both MobileCLIP passes, ArcFace and alignment, four of the five decodes |
| `POST /analyze/batch` | counts (detections, faces, has_text); OCR off | `detect, faces` | MobileCLIP, ArcFace |
| `POST /ingest` (core) | detections with box vectors, frame vector, faces with vectors, OCR, near-duplicate group | `detect, embed_frame, embed_crops, faces, face_embed, ocr, near_duplicates` | the extra decodes only |
| Curation ingest | per the project policy | policy `all`: `detect, embed_frame, embed_crops` (+ `region` when a region profile is active); `selected`: same, `embed_crops` masked by the policy; `lazy`: `detect, embed_frame`; `none`: `detect` | GPU route: the dropped frame vector (`none`) and the unselected crop vectors (`selected`) |

Legacy flags are deleted in the same change. Per `docs/VISION_AND_GOALS.md` there are no shims and
no parallel parameters: `enable_ocr`, `include_embedding` and `include_face_embeddings` on
`/analyze`, and `enable_detection`, `enable_clip`, `enable_faces`, `enable_ocr` and
`detect_near_duplicates` on core `/ingest`, are replaced by plan branches. Parameters that tune a
branch rather than select it stay (`confidence`, `face_confidence`, `near_duplicate_threshold`).
`face_confidence` now also applies to `/analyze/batch` and core `/ingest`, where 0.5 is hard-coded
today. No frontend consumer uses these flags (`frontend/` has no reference). The callers to update
are the tests and scripts listed by
`rg -l "include_embedding|enable_ocr|enable_faces" tests scripts`. This is owner question Q1.

### 3.5 Curation: the policy is the plan, a request may only narrow it

Curation ingest already has a revisioned per-project plan, the ingest policy. A second source of
truth would break the no-duplicate-logic rule. So:

- `policy_plan(policy, region_profile_active) -> ResolvedPlan` (the table in 3.4) is the only
  constructor of the curation default.
- A request plan is **intersected** with the policy plan. Asking for a branch the policy plan does
  not contain is a 422 (`branch_not_in_policy`). Example: `embed_crops` on a project with policy
  `none`. Narrowing is allowed, for example a bulk run that sends `{"branches": ["detect"]}` on a
  policy `all` project: items are stored `deferred` exactly as under `lazy`/`none` and are embedded
  later by `POST /reprocess` with scope `embed`.
- The lock rule is untouched. An item with a human or imported label always gets a vector
  (`embedding_states`). The narrowing never applies to it: a narrowed `embed_crops` still embeds
  labeled items. The same holds today for `lazy`.
- `faces`, `face_embed`, `ocr` and `near_duplicates` are not in `ALLOWED['curation_ingest']`,
  because no curation store exists for them. Requesting them is a 422, never a silent no-op. Text
  reading stays a region-stage worker feature.

## 4. Server-side validation (fail closed)

`resolve()` either returns a plan that runs exactly as written or raises `PlanError`. Nothing is
added, dropped or guessed.

| Reason code | When | Response |
|---|---|---|
| `unknown_branch` | a name outside `Branch` (pydantic `Literal`) or an unknown key (`extra='forbid'`) | 422 |
| `empty_plan` | `branches: []` | 422 |
| `duplicate_branch` | a branch listed twice | 422 |
| `branch_not_supported` | a valid branch outside `ALLOWED[route]` | 422, names the branch and the route |
| `faces_disabled` | `faces` or `face_embed` requested explicitly while `OP_FACES_ENABLED=false` (#379); the default plan of a route leaves both out when faces are off. Reuses the existing `faces_disabled` refusal (`src/config/faces.py`, `src/core/error_handlers.py`), not a new code | 422 as today |
| `empty_condition` | a `when` entry whose class names are all empty after normalization | 422 |
| `plan_required` | a curation route without a plan before `policy_plan` exists (P4); the curation default comes from the project policy, so there is no static default | 422 |
| `missing_dependency` | for example `embed_crops` without `detect`, `face_embed` without `faces`, `near_duplicates` without `embed_frame`, `if_detected` without `detect`. **Refused, not auto-added** (Q6). The plan a client sends is the plan that runs. | 422, names the missing branch |
| `condition_on_unconditional` | `when` names a branch outside `CONDITIONAL`, or a branch not in `branches` | 422 |
| `unknown_class` | an `if_detected` name the detector does not label (core: the model's `labels.txt` via `src/utils/affine.py`'s label lookup; curation: the detector vocabulary, `src/services/curation/detector_vocabulary.py`) | 422, lists the names |
| `branch_not_in_policy` | curation: outside the project policy's plan (3.5) | 422 |
| `paired_branches` | core `/ingest`: `detect` without `embed_crops` or the reverse (the category indexes are vector indexes; a detection without a vector has nowhere to go) | 422 |
| model unavailable | a resolved branch's model is not READY and on-demand loading (`src/clients/model_residency.py`) fails | the branch's outcome is `error` with Triton's message. The branch never disappears from the report. |

Runtime failure of one branch: the outcome is `status: error` with the message. The branch's
output field is `null`, never an empty list, so a failure is never read as "no faces". The HTTP
status stays 200 while at least one requested branch is `ok`. When every requested branch fails it
is 502. This generalizes today's `ocr_error` (`analyze_pipeline.py:40-46`). It replaces the
fail-open face path, where a face error becomes `faces: []`
(about `analyze_pipeline.py:282-284`, `inference.py:136-200`). Retry exhaustion keeps its existing
handler (`RetryExhaustedError`, 503). Q5 asks the owner to confirm.

Output fields of a branch that was not requested are `null`, not empty: `detections`, `faces`,
`global_embedding`, `ocr` and each detection's `embedding`. The `num_*` counters are `null` when
their branch did not run. This changes `AnalyzeResponse` (`src/schemas/analyze.py`), where these
fields default to `[]` and 0 today.

## 5. Execution: one resolved plan, two backends

```
request --plan--> resolve(route, plan) --ResolvedPlan--> executor
  hash (if the route stores) -> dedup
  decode ONCE: CPU route (frames below OP_INGEST_GPU_MIN_MEGAPIXELS, CPU-only installs)
               or GPU route (gpu_image_prep, photos, when OP_INGEST_GPU_DECODE is on)
  for b in plan.branches (canonical order; independent branches overlap):
      skip unless plan.runs(b) and its condition holds
      take the views b needs from the per-image view cache
      one batched Triton call per model per request (never per crop)
  build the response from the outcomes; unrequested fields stay null
```

### 5.1 CPU route (API-orchestrated)

- New module `src/services/plan_executor.py` (no FastAPI import), holding `ImageViews`: the
  decoded frame plus memoized `letterbox(size)`, `encoder_square(model)`,
  `crop_squares(boxes, model)`, `face_working(max_side=1024)` and `ocr_input()`. Each view is built
  on first use from the full frame, and RGB/BGR order is fixed per view.
- The model clients take arrays instead of bytes. Each byte-taking entry point is split into a
  decode step (deleted once the executor is the only caller) and an array step:
  `triton_client.infer_yolo_clip_cpu` becomes separate detector and MobileCLIP calls (the box
  vectors move to `embed_crops`), `fast_face_client.recognize` gets `detect_faces(views)` and
  `embed_faces(views, faces)`, `triton_ocr.infer_ocr` gets `ocr(views)`, and
  `ocr_service.extract_text` drops its validation decode in favor of the executor's decode verdict.
  The other callers of `infer_yolo_clip_cpu` (visual search,
  `src/services/visual_search/search.py`; core batch ingest,
  `src/services/visual_search/batch_inference.py`; `src/services/inference.py:267`) move to the same
  split calls, and the method is deleted, not kept next to the new calls.
- Overlap: independent branches (`detect`, `embed_frame`, `faces`, `ocr`) run on the existing
  thread pool, as `analyze_single_image` does today. `embed_crops` and the `if_detected`
  conditions wait for `detect`.
- Curation ingest is already decode-once on this route. The plan only replaces the
  `service.policy.embedding.mode` checks (`ingest_index.py:230`, `ingest_batch.py:85`) with
  `plan.runs(...)`, so a policy and a narrowed request take the same code.

**Decode numerics.** Faces and OCR decode with OpenCV today, the detector and MobileCLIP with PIL.
One decoder for all of them is a one-time numerics change for at least one branch, so it needs a
parity reference decision like D1 of `gpu_decode_plan.md` (Q2). Gates: P-box, P-face 1.0 (face
count and box IoU), ArcFace cosine p01 at least 0.995, P-ocr text match, against the 0.8 reference
on set A.

### 5.2 GPU route (Triton BLS, `gpu_image_prep`)

`gpu_image_prep` is already the plan executor for the curation branches: a decoupled Python BLS
that decodes once, runs the detector and PE through BLS and returns only small outputs
(`models/gpu_image_prep/config.pbtxt` header, `model.py:270-318`). The plan extends it rather than
adding a new model:

- Replace the single optional input `embed_crops` (BOOL) with optional `embed_frame` (BOOL,
  default true) and `crop_select` (BOOL `[-1]`, or a compiled criteria tensor, see Q4). The model
  fills a fixed output schema with empty tensors for disabled branches, as Wave 7 of the pipeline
  plan specifies, so clients never branch on missing outputs.
- Policy `none` sends `embed_frame=false`. With no crops embedded that removes **all** PE work for
  the image. Policy `selected` gets the per-detection mask, and the mask must be decided after
  detection, inside the request. Q4 has two ways to do that.
- After #220 lands, `faces`, `face_embed` and `ocr` become GPU-route branches. SCRFD on the
  resident frame, alignment on the GPU (#220), ArcFace through BLS, and OCR from a uint8 view of
  the resident frame with width-bucketed recognition (#220). Each is one more flag on the same
  request. The frame is decoded once and never leaves the GPU.
- What stays in the client, per pipeline-plan Wave 7: identity, dedup, class mapping by name,
  policy decisions and writes. The BLS returns measurements only.

### 5.3 Ensemble or BLS (open question C1): recommendation and reasoning

C1 (`triton_pipeline_optimization_plan.md` section 5.6): does a Triton ensemble run a step whose
outputs are all unrequested? It is unanswered (section 11 of that plan).

**Recommendation: BLS for the GPU route and API orchestration for the CPU route. No ensembles for
plan execution, whatever C1 says.**

1. **Conditions.** An ensemble is a static DAG with no control flow. "OCR only if a car was
   detected" needs a branch decided at run time, and the Triton documentation points to BLS for
   conditional execution and loops.
2. **Variable fan-out.** `embed_crops` and `face_embed` take N inputs that exist only after
   detection. An ensemble cannot cut N crops from a frame (pipeline plan section 9: "DALI doing the
   per-crop work: one sample in gives one sample out").
3. **Combinatorics.** Even if C1 showed that unrequested steps are pruned, a static plan would
   need one ensemble config per branch combination on each route (up to 2^6 for `/analyze`). That
   is config sprawl with no conditions still.
4. **What exists.** No ensemble is deployed (`models/` holds TensorRT models plus two Python BLS
   models, `ocr_pipeline` and `gpu_image_prep`). The measured path is the BLS: `gpu_image_prep` runs
   43-48 photos/s in perf_analyzer at 32 streams, and 1.3x the CPU path end to end with detection and
   embedding, 1.55x detection only (`gpu_decode_plan.md` section 13).
5. **Known BLS risk.** Python BLS overhead is real. `ocr_pipeline` spends 42-46 ms per COCO image
   for about 9 ms of GPU (`docs/PERFORMANCE.md` Probe 3), and public reports put BLS behind ensembles
   (`docs/research/v060_serving_backend_trade_study.md` S10). The mitigation is the parent plan's rule: few,
   large BLS calls, batched per consumer, plus the 15 % stop rule against the API-orchestrated arm.

Measurements that confirm it (section 5.4): M1 answers C1 and is recorded in the pipeline plan's
section 11. M2 runs BLS against API orchestration per plan. M3 checks that the default plan does
not regress. If M1 shows ensembles prune unrequested steps, the recommendation does not change, but
the result feeds Wave 6 (static DALI ensembles for stateless single-model routes such as `/detect`),
which is a different question.

### 5.4 Measurements that confirm the choice (M1-M4)

Short tiers first. One full confirmation per bundle, and no full factorial.

| ID | Question | Method | Pass / decision |
|---|---|---|---|
| M1 (C1) | Does an ensemble run steps whose outputs are not requested? | Throwaway ensemble `c1_probe` (not committed to `models/`): one UINT8 input fanned to the detector and to `mobileclip2_s2_image_encoder` through two preprocess-free steps; 100 requests asking only for the detector outputs; read `nv_inference_count` per composing model from the Triton metrics port (4602) before and after | record the answer in `triton_pipeline_optimization_plan.md` section 11. The recommendation in 5.3 stands either way. |
| M2 | BLS against API orchestration per plan | set A (2,000) and set B (500): plans `faces`, `detect`, `detect+embed_frame+embed_crops`, the `/analyze` default, at 1 and 16 clients, 3 repetitions, GPU 2, same stack; images/s, p50/p99, API and Triton CPU-s per image, bytes per branch, GPU busy | BLS arm adopted per plan only if it is not more than 15 % slower than the API arm (the parent plan's stop rule); otherwise that plan stays on the CPU route |
| M3 | No regression from the plan layer | default plan of each route against `main` before the change, set A, 16 clients | within 3 % images/s, p99 not worse than 10 % |
| M4 (#67 acceptance) | Faces only is measurably cheaper | `POST /analyze` with `{"branches":["faces","face_embed"]}` against the full `/analyze` plan on set A, 16 clients; images/s and bytes per image | faces-only within 10 % of `faces_recognize` (179.3 images/s today) and at least 2x the full `/analyze` row; recorded in `docs/PERFORMANCE.md` |

## 6. Interaction with the other plans

### 6.1 Ingest load profiles (`OP_LOAD_PROFILE`)

- Under `OP_LOAD_PROFILE=ingest`, Triton preloads only the detector and the PE models, and the
  first call that needs another model loads it (`src/clients/model_residency.py`
  `ensure_loaded`). Today `/analyze` asks for SCRFD, ArcFace and MobileCLIP on every call, so a
  detection-only caller loads 6.8 + 2.8 + 3.5 GB it never uses. That is 13.1 GB, from the measured
  table in `ingest-load-profile.mdx`. With plans, `ensure_loaded` is called per **resolved
  branch**, so an unrequested branch never loads its model. This is part of the skip guarantee
  and is tested (9.2).
- The profile and the plan stay independent. The profile decides what is resident, the plan
  decides what runs. A plan never changes residency except through on-demand loading, and the
  profile never removes a branch from a plan. An unloadable model is an `error` outcome (section 4).
- `gpu_image_prep`'s device budget (`docs/design/sp_gpu_prep_concurrency.md`) reserves the decoded
  frame per photo. Fewer branches means the frame is held for less time. No budget change is
  proposed. Watch `op_gpu_prep_stage_seconds{stage="budget"}` in M2.

### 6.2 Selective embedding (#52, shipped)

- Policy modes map onto plans exactly (3.4). The mapping is one function, and the existing
  characterization tests are the proof that ingest behavior is unchanged (acceptance criterion 4 of
  #67).
- The two GPU-route wastes (2.1) are fixed by the plan flags (5.2). The CPU route already honors
  the policy before PE.
- `embedding_state` semantics do not change. A narrowed request stores `deferred` exactly as `lazy`.

### 6.3 #220 (uint8 OCR, width buckets, face alignment on resident frames) and #40, #219

- #220 changes **how** the `ocr`, `faces` and `face_embed` branches consume pixels. This plan
  changes **whether** they run. The interface between them is `ImageViews` (CPU) and the
  `gpu_image_prep` request flags (GPU). #220's plan doc (`faces_ocr_resident_crops_plan.md`, not
  written yet) should name its new OCR input as a view of the shared frame, so no plan branch
  decodes or converts the frame a second time.
- Order: P1 (pure domain module, wire models, tests) touches no route, model or existing wire and
  does not wait. #220 PR 1 fixes the OCR branch's input contract (uint8 `ocr_images` and
  `original_image`, datatype read from model metadata, `MODEL_VERSION`), so P2 onward waits for it:
  the input changes once and parity is measured once. Width-bucketed recognition (#220 PR 2) is
  inside the BLS and does not change the contract. GPU face alignment on resident frames (the old
  #220 item 4) is part of P5 here: no face route holds a GPU frame today.
- #219 (work crops at model size) and #40 (resident crops) make the views cheaper. The plan makes
  them conditional. Neither changes the plan wire.

## 7. Wire changes and contracts

| Surface | Change | Contract |
|---|---|---|
| `POST /analyze`, `POST /analyze/batch` | add form `plan`; delete `enable_ocr`, `include_embedding`, `include_face_embeddings`; response gains `plan: PlanReport`; unrequested outputs `null`; `timing` keyed by branch name (`detect`, `embed_frame`, ...) instead of `yolo_clip_ms`, `faces_ms`, `ocr_ms`, `crop_embeddings_ms` | core routes are not in `contracts/openapi/curation.json`; update `src/schemas/analyze.py`, the route docstrings, `README.md` API table and `docs/` pages that name the flags (`scripts/docs/check_docs_vs_code.py` catches stale routes, not stale flags, so grep for them) |
| `POST /ingest`, `POST /ingest/batch`, `POST /ingest/directory` (core) | add `plan`; delete the five `enable_*`/`detect_near_duplicates` flags; stop writing a zero global vector (a plan without `embed_frame` stores no global vector, Q7); response gains `plan` | same as above |
| Curation ingest routes, bulk runs, registered sources | optional `plan` on `IngestImageRequest`, `IngestBatchRequest`, the upload form, `BulkIngestRequest` (`src/routers/curation/_ingest_bulk_models.py`) and the source record; response and run status gain `plan` | `make contracts`, commit `contracts/` in the same commit (the `api-contracts-drift` hook rejects a stale one) |
| Shared | `ProcessingPlan`, `BranchCondition`, `PlanReport`, `BranchOutcome` in `src/schemas/processing_plan.py` | appear as components in `contracts/openapi/curation.json` |
| Triton | `gpu_image_prep` inputs (`embed_crops` replaced, 5.2); after #220 the face and OCR flags | header comment in `models/gpu_image_prep/config.pbtxt`; a new model version only if outputs change shape |

Bulk runs and registered sources store the request plan, not the resolved one. Each batch resolves
it against the policy that is current at that batch. A policy change during a run that removes a
branch the run asked for fails that batch with `branch_not_in_policy` instead of silently dropping
the branch. That is consistent with how bulk batches already report per-batch failures. Q3 asks the
owner to confirm.

## 8. Observability: cost per plan

Existing instruments, kept: `op_pipeline_stage_seconds{stage}` and
`op_pipeline_stage_bytes_total{stage}` (API, `src/utils/stage_timing.py`),
`op_gpu_prep_stage_seconds` and its observations counter (Triton,
`models/gpu_image_prep/1/stage_metrics.py`), `op_ingest_decode_total{route}`.

New, in `src/utils/stage_timing.py` next to the existing ones (same `prometheus_client` registry,
the same `__enter__`/`__exit__` timer that never swallows exceptions):

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `op_plan_requests_total` | counter | `route`, `plan` (`ResolvedPlan.key`, canonical `detect+faces`) | requests per resolved plan. Cardinality is bounded by the closed vocabulary and the per-route allowlist, at most 64 per route. |
| `op_plan_branch_total` | counter | `route`, `branch`, `outcome` (`ran`, `error`, `condition_false`, `not_requested`) | proves skips in production. `not_requested` costs one increment and no work. |
| `op_plan_branch_seconds` | histogram | `route`, `branch` | wall time of the branch per image, including its Triton round trip, with the same buckets as `op_pipeline_stage_seconds` |
| `op_plan_branch_bytes_total` | counter | `route`, `branch` | bytes sent to Triton by the branch (the #67 acceptance "images/s and bytes") |
| `op_plan_decodes_total` | counter | `route` | decodes performed. Divided by images it must be 1.0 (acceptance "decode once per image"). |

GPU route: `gpu_image_prep` stage names map to branches (`detect` to `detect`, `embed_frame` to
`embed_frame`, `embed_crops` plus `embed_pe` to `embed_crops`). Stages of disabled branches record
no observation, which the offline model test checks. The response `PlanReport.outcomes[*].ms` is
the per-request view of the same timers. `scripts/bench/run_baseline.py` gains a `--plan` argument
so a benchmark row is labeled by its plan key.

## 9. Test strategy

Red first: each test is watched failing before the code that makes it pass, and each skip test has
a mutation check (remove the `plan.runs` guard, and the test must fail).

### 9.1 Pure resolution (`tests/test_processing_plan.py`, no I/O; implemented in P1)

`resolve(route, requested, *, faces_enabled)` takes the deployment face switch instead of
`known_classes`; the class-name check is a separate step in the route that owns the detector
vocabulary. `PlanRequest` is the pure-domain input and `ProcessingPlan.to_request()` builds it;
`ProcessingPlan.branches` is `list[str]` so an unknown name reaches `resolve` and carries the
`unknown_branch` reason code. `if_detected` names are normalized with
`src/utils/class_names.normalize_class_name`, the normalizer `name_match` is built on (a domain
module may not import `src/services`).

- A table test per row of section 4, one case per reason code, asserting the code and the named
  branch.
- Defaults: `resolve(route, None)` equals `DEFAULT[route]` for every route. For curation, one case
  per policy mode, with and without a region profile.
- Narrowing: the intersection rules of 3.5, a widening refusal per branch, and a labeled item still
  embedded under a narrowed plan (through `embedding_states`).
- Properties over every subset of `BRANCHES`: either `PlanError` or a plan whose branches are a
  subset of `ALLOWED[route]` and closed under `REQUIRES`. `key` is canonical (order-insensitive
  input, same key).

### 9.2 Skip proofs per stage (one test per branch, the #67 acceptance "no Triton call, no host work")

Reuse the fake seams of `tests/test_analyze_routes.py` (`FakeTriton`, `FakeFaces`, `FakeOcr`),
extended to record every model name called and every preprocess function entered (spies on
`preprocess_yolo_cpu`, `preprocess_clip_cpu`, `center_crop_cpu`, the face working-copy resize, the
OCR resize, `align_faces_batch`). For each branch `b` and each route that allows it, run the default
plan minus `b` (plus its dependents) and assert:

- zero calls to `b`'s models;
- zero entries into `b`'s preprocess functions;
- zero `ensure_loaded` calls for `b`'s models under `OP_LOAD_PROFILE=ingest`;
- `op_plan_branch_total{outcome="not_requested"}` incremented and `ran` not.

Conditions: `if_detected` true runs the branch, false records `condition_false` and makes no call.
Unknown names are 422 (9.4).

### 9.3 Decode once

- Every plan on every route: a spy on the single decode entry point counts exactly one decode per
  image. `op_plan_decodes_total` equals the image count.
- Static guard (`tests/test_no_branch_decode.py`, the same AST style as
  `tests/test_no_legacy_region_scalars.py`): no module the executor dispatches to calls
  `Image.open`, `cv2.imdecode`, `decode_image` or `simplejpeg.decode_jpeg`, except the one decode
  module.

### 9.4 Wire

- 422 tests per reason code through the HTTP layer for each route (multipart `plan` JSON and JSON
  body), plus `extra='forbid'`.
- Response shape: unrequested fields are `null`, a failed branch has `status: error` and a `null`
  output, and all-failed returns 502.
- The removed flags are rejected. FastAPI ignores unknown query parameters, so the test asserts
  the route signature no longer declares them, and the docs grep in section 7 finds no reference.
- `make contracts-check` clean. `tests/test_codegen_api_contracts.py` covers the curation
  components.

### 9.5 Unchanged behavior for existing requests

The "unchanged behaviour" baseline includes the faces-off case (`OP_FACES_ENABLED=false`: faces skipped on
`/analyze`, forced off on core `/ingest`). The existing characterization suites stay green without edits to their expectations:
`tests/test_ingest_single_characterization.py`, `tests/test_ingest_batch_characterization.py`, the
curation ingest and policy tests, and the reprocess tests. Where a legacy flag is removed, only the
request changes (flag to plan) and the expected response stays byte-identical except for the
documented `null` and `timing` keys.

### 9.6 `gpu_image_prep` offline

With the suite's stand-ins for CuPy, nvJPEG and `pb_utils` (`sp_gpu_prep_concurrency.md` 6.1), test
flag decoding, the fixed output schema with empty tensors for disabled branches, a BLS call count
per flag (`embed_frame=false` and no crops means zero PE calls), and the `selected` mask's parity
with `ingest_policy._mask` on a table of candidates (Q4).

### 9.7 Live (the `docker/test/compose.yml` harness, GPUs 0 and 2, own `-p` name)

- Parity of each route's default plan against the 0.8 reference on set A: P-box, P-item, P-pe,
  P-clip, P-face, P-ocr (`docs/benchmarks/parity/`).
- Benchmarks per plan (section 5.4) recorded in `docs/PERFORMANCE.md`, with public COCO numbers
  only.

## 10. Phased rollout

Each phase is one PR or wave branch with its own gate (offline suite, pre-commit, contracts) and is
reverted by `git revert`. No phase migrates stored data. Q7 changes what new core `/ingest`
documents contain, not existing ones.

| Phase | Starts after | Content | Gate |
|---|---|---|---|
| P0 | #220 PR 1 merged | M1; per-plan baselines from today's code where expressible (`/faces/recognize`, `/detect`, `/analyze`) on set A and set B | numbers in `docs/PERFORMANCE.md`, C1 recorded |
| P1 (done) | none | `src/domain/processing_plan.py`, `src/schemas/processing_plan.py`, `tests/test_processing_plan.py`; no route uses them yet. The `stage_timing.py` metrics (section 8) and the `unknown_class`, `branch_not_in_policy` checks move to the phase whose route first uses them, so the unused-code ratchet stays clean | 9.1 green |
| P2 | P0, P1 | `faces_enabled` deleted from the analyze responses (superseded by the `PlanReport` outcome `disabled`, #379); section 8 metrics added; `/analyze` and `/analyze/batch` on the CPU executor: `ImageViews`, decode once, split model clients, legacy flags removed, `null` semantics, per-branch outcomes; `infer_yolo_clip_cpu` callers moved and the method deleted | 9.2-9.5 for these routes, M3, M4, P-face/P-ocr parity (Q2) |
| P3 | P2 | core `/ingest` family on the executor; zero global vector removed; `near_duplicates` dependency | 9.2-9.5, M3 |
| P4 | P1 | curation: `policy_plan`, request narrowing on ingest routes, bulk runs and sources; `gpu_image_prep` `embed_frame` flag and the `selected` mask (Q4); `make contracts` | curation characterization green, 9.6, M3 on the GPU route (policy `all`), plus a `none` and `selected` row |
| P5 | P2, #220 PR 1 | `faces`, `face_embed`, `ocr` as `gpu_image_prep` branches for photos on `/analyze` and core `/ingest`, including GPU face alignment on the resident frame (moved here from #220; acceptance: ArcFace cosine p01 at least 0.995, P-face 1.0) | M2 with the 15 % rule per plan; parity gates |
| P6 | P5 | docs (`README.md` API table, `docs/CURATION.md` ingest section, `docs-site` ingest pages), `docs/PERFORMANCE.md` per-plan table, this doc's status to implemented, close #67 | `scripts/docs/check_docs_vs_code.py` 0 problems |

## 11. Non-goals

- A user-defined graph or expression language. The vocabulary, the dependencies and the one
  condition kind are fixed in code. Adding a branch is a code change with tests.
- VLM, segmenter or region work inline at ingest. They stay worker and assist-run features. The
  `region` branch only seeds the queue.
- OCR or faces on detection crops (crop-scoped OCR). This is a different output contract. It could
  become a later condition variant.
- GPU-resident crops, uint8 OCR and GPU face alignment themselves: #40, #219, #220.
- Changing an embedding space or any stored vector. The decode-numerics change in Q2 affects new
  face and OCR outputs only, behind the parity gates.
- Reworking reprocess scopes or the assist-run API.
- Static DALI ensembles for single-model routes (pipeline-plan Wave 6).
- Named plan presets (Q8).

## 12. Open questions for the owner (with recommendations)

| ID | Question | Recommendation |
|---|---|---|
| Q1 | Remove the legacy selection flags (`enable_ocr`, `include_embedding`, `include_face_embeddings` on `/analyze`; `enable_*` and `detect_near_duplicates` on core `/ingest`) in the same change? | Yes, per the no-shims rule. No frontend consumer; tests and scripts are updated in the same PR. |
| Q2 | One decoder for every branch changes face and OCR inputs (OpenCV today) and needs a new parity reference, as D1 did for the GPU decode | Accept after the gates in 5.1 pass on set A and the private set (aggregates only) |
| Q3 | Curation: allow per-request plans at all, and if so narrowing only? | Narrowing only (3.5). Widening past the policy is a 422, and a mid-run policy change that removes a requested branch fails the batch. |
| Q4 | Policy `selected` on the GPU route needs a per-detection mask decided inside the BLS request | Send compiled criteria: the detector's own class ids for the policy's class names, resolved by name in the API from the same detector's labels in the same request, plus confidence, area and per-image cap. A parity test against `ingest_policy._mask` guards it. The alternative, a second stored-box request, re-decodes the image. |
| Q5 | One requested branch fails: 200 with a per-branch error, or fail the request? | 200 with `status: error` and a `null` output. 502 only when every requested branch failed. |
| Q6 | Missing dependency: refuse or auto-add? | Refuse with `missing_dependency`. The sent plan is the plan that runs. |
| Q7 | Core `/ingest` without `embed_frame`: store the image with no global vector, or refuse? | Store no global vector (no zero vector); `near_duplicates` requires `embed_frame` |
| Q8 | Named presets (`faces`, `full`)? | Not in v1. The branch list is short, and presets would be a second vocabulary. |
| Q9 | Should reprocess scopes become plans later? | Decide after P4. The names are already aligned, so the mapping stays mechanical. |
