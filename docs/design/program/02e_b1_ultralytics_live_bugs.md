# B1 track 1E: minimal Ultralytics live-bug fixes on the pinned 8.4.197

| Field | Value |
|---|---|
| Status | plan, not started |
| Batch / release | B1, part of the single v0.7.0 release |
| Depends on | B0 merged (it pins `ultralytics==8.4.197` in `requirements.txt` and `requirements-test.txt`, D-2; the pin is already on `fix/411-r2-round`); `int/b1-correctness` cut. Independent of 1A-1D |
| Decisions applied | D-2 (exact pin 8.4.197, no version move in B1), D-10 (minimal 1E: one export argument, a head check that moves unchanged into B3's module, one routing condition), D-11 (option b: bring-your-own original-YOLOv5 ONNX for the probe and the ingest secondary only, real-ONNX tests), D-12 (`nms` by capability: `nms=False` when the head has `one2one_cv2`; the export path in B1, the trainer in B3) |
| Source IDs | `model_family_versions_plan.md` section 7 Steps 1-3, C1, C2, C3 (comment only), C8, C10, J3, J4, 6.1 items 1-3 and 6; the former master plan (rev 4, in git history) 7.1 1E row and "1E tests and CI", 9.2 #447 re-scope; issues #447, N7 (`09_github_actions.md`) |
| Size | S-M |
| Lane | branch `b1/ultralytics`, worktree `<repo>-wt-b1-ultralytics`, compose project `op-b1-ultralytics` (only for the one local `families` run in the export image, done in the sweep) |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read only, with `rg`).
Consolidated 2026-11-03.

Not in B1 (moved to B3, `04_b3_ultralytics_one_version.md`): the trainer's `_export_onnx` `nms` by
capability and its pin, the `half=` to `quantize=` changes, the single export module
`export/ultralytics_export.py`, the full U-6.1 matrix, deleting `/opt/venv-y11` and the vendored
patch, GPU-14 final, U-C11, U-R8 (probe head vs served head, #368).

## 1. Goal

On the Ultralytics version that ships (8.4.197), a YOLO26 upload builds an engine with the right
head or is refused, a YOLO26 upload is no longer routed to the YOLO11 toolchain, the ingest
secondary detector maps classes by name, and the original-YOLOv5 probe works on any static input
size. Every test goes through Ultralytics' documented `nms` argument only (`end2end` is deprecated
and mapped to `nms` per the Ultralytics maintainers).

## 2. Preconditions

- Lane worktree: `git worktree add <repo>-wt-b1-ultralytics -b b1/ultralytics int/b1-correctness`.
- The venv has `ultralytics==8.4.197`, `onnx`, `onnxruntime` (from `requirements-test.txt`).
- Network for the weight fixtures (official release assets). The tests skip with an explicit reason
  when offline locally and never skip in CI.
- The original-YOLOv5 ONNX fixture is produced by the official `ultralytics/yolov5` repo's
  `export.py` in a throwaway venv (that repo pins its own `ultralytics>=8.4.192`; keep it out of the
  project venv). The probe scripts that produced the evidence are in `artifacts_local/model_versions/`
  (`scripts/y5_env.sh`, `scripts/v5probe.py`) on the owner's machine.

## 3. Documented behaviour this track relies on

| Fact | Source |
|---|---|
| `nms` export argument: "`None` exports raw one-to-many predictions for external NMS; `True` embeds NMS where supported; `False` selects the NMS-free head when available"; `end2end` deprecated and mapped to `nms` (maintainer comment) | https://docs.ultralytics.com/modes/export/ |
| `nms=False` gives `(N, 300, 6)` for YOLO26 and YOLOv10 | https://docs.ultralytics.com/guides/end2end-detection/ , https://docs.ultralytics.com/models/yolo26/ , https://docs.ultralytics.com/models/yolov10/ |
| Original-YOLOv5 checkpoints are not compatible with the Ultralytics library; YOLOv5u is | https://docs.ultralytics.com/models/yolov5/ |
| ONNX graph inputs/outputs (`onnx.load(...).graph.output`, `onnxruntime.InferenceSession.get_inputs()`) | https://onnx.ai/onnx/api/ , https://onnxruntime.ai/docs/api/python/api_summary.html |
| `actions/cache` keyed on a hash | https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows |
| pytest markers must be registered under `--strict-markers` | https://docs.pytest.org/en/stable/how-to/mark.html |

Evidence (CPU, 2026-11-03, `model_family_versions_plan.md` section 3): on 8.4.197 the repo's
`export_yolo26.py` arguments give the raw `[b, 84, anchors]` head for `yolo26n.pt` (#447);
`nms=False` gives `[b, 300, 6]` with `TopK`; a loaded `yolo26n.pt` reports `end2end=False` on
8.4.197 (C2), while `hasattr(model.model[-1], 'one2one_cv2')` is true for YOLO26 and YOLOv10 and
false for v5u, v8, v9, 11 on every version tested; the original-YOLOv5 640 static ONNX
`output0 [1, 25200, 85]` fails the probe ("Got 1280 Expected 640").

## 4. Current state (verified)

- `export/export_yolo26.py::export_fused_onnx(model, config)` calls `model.export(format='onnx',
  imgsz=IMG_SIZE, dynamic=True, batch=config['max_batch'], device='cpu', simplify=True)` (no
  `nms`). `inspect_fused_output(onnx_path)` checks only that there is one output and returns its
  per-item dims; it does not refuse a raw head.
- `src/services/training/yolo_triton_config.py`: module docstring says "`end2end=True` is the
  default -> a single output tensor of shape `(N, 300, 6)`" (false from 8.4.174 at the latest, C3);
  `YOLO26_TOPK = 300`.
- `src/services/training/triton_promote.py`: `_locate_onnx` finds the trainer's `best.onnx`; the
  ONNX is copied with `add_uint8_input` into `models/<name>/1/model.onnx` without a head check.
- `src/services/model_export.py::validate_pytorch_model` stores
  `'end2end': bool(getattr(model, 'end2end', False))`; the export task dict and the queued job dict
  carry `end2end` (`rg -n "'end2end'" src/services/model_export.py`: 3 sites). Not in `contracts/`.
- `export/export_worker.py::build_command` and `engines_to_load` route on `job.get('end2end')`
  (`EXPORT_SCRIPT_Y26` for true, `EXPORT_SCRIPT` with the vendored EfficientNMS toolchain for false).
- `src/services/curation/ingest_detect.py::WholeImageDetector.resolve_with_secondary` sets
  `item.class_id = sec_cls` and `self.registry.get(sec_cls)`: a raw model index used as a registry
  id (C10). The primary path (`assigns_class` profiles) also does `self.registry.get(cls_id)` with
  the model index, relying on the registry having been seeded in ONNX order (fixed by 1E-4b).
- `src/services/curation/probe_models.py`: `YOLOV5_OBJ_INPUT_SIZE = 1280`;
  `_build_yolov5_objectness_predictor` reads `session.get_inputs()[0].name` but letterboxes to the
  constant.
- Tests: `tests/test_validate_pytorch_model_restricted.py` exists; `tests/curation/test_probe_models.py`
  uses a fake onnxruntime. There is no `tests/export/` directory. `pyproject.toml` registers only
  the markers `integration`, `live`, `unbound` under `--strict-markers`.
- CI (`.github/workflows/ci.yml`): jobs `pytest` (offline suite, `-m 'not live'`), `pre-commit`,
  `changelog`.

## 5. Tasks

One commit per task; red test first, seen failing on the base; paired control.

### 1E-0 Weight fixtures and the `families` marker (S)

- Files: `pyproject.toml` (register `families: tests that need public Ultralytics weights or the
  official YOLOv5 ONNX fetched from the network`), a helper `tests/_ultralytics_weights.py`
  (session-scoped fixture), `tests/conftest.py` (expose the fixture), `.github/workflows/ci.yml`
  (the `pytest` job selects `-m 'not live and not families'`), the offline-suite command in
  `CLAUDE.md` and CONTRIBUTING.md (`-m 'not live and not families'`).
- Fixture rules:
  - Ultralytics weights `yolo26n.pt`, `yolo11n.pt`, `yolov10n.pt`, `yolov5nu.pt` from
    `https://github.com/ultralytics/assets/releases/download/v8.4.0/` (AGPL-3.0), original
    `yolov5n.pt` from `https://github.com/ultralytics/yolov5/releases/download/v7.0/` (AGPL-3.0),
    into `~/.cache/op-test-weights/` (override with `XDG_CACHE_HOME`). Record a sha256 per file in
    the helper (the evidence prefixes: `yolo11n.pt` 0ebbc80d, `yolo26n.pt` 9b09cc8b, `yolov5nu.pt`
    9e9c1be4, `yolov10n.pt` 11287ed0, `yolov5n.pt` 4f180cf2; take the full hashes when first
    downloading) and refuse a mismatch.
  - The original-YOLOv5 ONNX (640 static, `--simplify`) is read from the same cache dir; the CI job
    produces it (1E-6); locally the developer produces it once with the official command
    (`python export.py --weights yolov5n.pt --include onnx --imgsz 640 --device cpu --simplify`).
  - Nothing is committed under `tests/fixtures/`.
  - When a file cannot be fetched: `pytest.skip` with the reason locally; `pytest.fail` when the
    `CI` environment variable is set (GitHub Actions sets `CI=true`). No new `OP_*` variable.
- Red: none (infrastructure); the control is that the offline suite selection still collects every
  non-`families` test.

### 1E-1 #447: YOLO26 export with `nms=False` and a head check (S)

- File: `export/export_yolo26.py` (`export_fused_onnx`, `inspect_fused_output`).
- Steps: pass `nms=False` (D-12: this path serves only one-to-one heads); `inspect_fused_output`
  refuses any output whose per-item dims are not `[max_det, 6]` with `max_det = 300` (read it from
  the export call's arguments, one constant), naming the dims it found and the cause ("raw
  one-to-many head: the export did not select the NMS-free head"). This function moves unchanged
  into B3's single export module.
- Red: `tests/test_export_yolo26_head.py` (marker `families`): export `yolo26n.pt` with the
  production function and assert one output `[batch, 300, 6]` and a `TopK` op; a negative test feeds
  a raw-head ONNX (export with `nms=None`) to `inspect_fused_output` and expects the refusal. Both
  fail on the base (the base exports the raw head and does not refuse it).
- Control: `tests/test_export_config_write.py` and the other `tests/test_export_*` tests pass.

### 1E-2 Promote refuses a non-`[300, 6]` ONNX; fix the comment (S)

- Files: `src/services/training/yolo_triton_config.py` (docstring: the trainer's 8.4.48 default
  gives `[300, 6]`; newer releases need `nms=False`, which B3 passes; the promote check refuses
  anything else), `src/services/training/triton_promote.py` (read the ONNX's output with `onnx`
  before `add_uint8_input`; refuse with a promote error naming the dims).
- Red: a promote with a raw-head ONNX in a temp dir is refused (today it is copied).
- Control: a `[1, 300, 6]` ONNX promotes as before (`rg --files tests | rg promote` for the existing
  tests).

### 1E-3 U-C2: route uploads by head kind, not `end2end` (S)

- Files: `src/services/model_export.py` (`validate_pytorch_model`, the task dict, the job dict),
  `export/export_worker.py` (`build_command`, `engines_to_load`).
- Steps:
  1. `validate_pytorch_model` records `head_kind`: `'one2one'` if
     `hasattr(model.model[-1], 'one2one_cv2')`, else `'nms'`; and `ultralytics_version` from the
     checkpoint (`ckpt.get('version')`). Drop the `end2end` key everywhere (3 sites).
  2. An original-YOLOv5 checkpoint (refused by `torch_safe_load(safe_only=True)` on 8.4.174+:
     "references types outside the supported Ultralytics checkpoint format") returns a message
     naming YOLOv5u: "original YOLOv5 checkpoints are not supported by Ultralytics; retrain from
     YOLOv5u (`yolov5su.pt`), or export an ONNX with the official `ultralytics/yolov5`
     `export.py` for the probe or the ingest secondary detector".
  3. `build_command` / `engines_to_load` switch on `job['head_kind']` (`one2one` ->
     `export_yolo26.py`, `nms` -> today's `export_models.py` toolchain). A job without `head_kind`
     (queued before the upgrade) fails closed with "re-upload the model". This is a condition
     change only; B3 collapses the two commands.
- Red: `tests/test_validate_pytorch_model_families.py` (marker `families`): yolo26n and yolov10n
  give `one2one`, yolo11n and yolov5nu give `nms`, original yolov5n is refused with the YOLOv5u
  message; a `build_command` unit test with `head_kind='one2one'` picks the native script (today a
  YOLO26 upload on 8.4.197 reports `end2end=False` and goes to the YOLO11 toolchain).
- Control: `tests/test_validate_pytorch_model_restricted.py` passes (restricted load still refuses
  non-Ultralytics pickles).
- Behaviour change: a YOLOv10 upload now goes through the native NMS-free path (`[300, 6]`); say so
  in the CHANGELOG.

### 1E-4 U-C10: secondary detector class by name (S)

- File: `src/services/curation/ingest_detect.py` (`resolve_with_secondary`).
- Fix: map the secondary model's output index to a name through the model's own names (Triton
  `labels.txt` via `src/utils/class_names.py::get_class_names(model_name)`, or the ONNX `names`
  metadata the probe already reads), then resolve the registry id by name with
  `src/utils/class_names.py::resolve_class_by_name` on the registry's classes. An index with no name,
  or a name with no active registry class, leaves the primary's class unchanged and is counted in
  the ingest log (never the raw index).
- Red: `tests/test_secondary_class_by_name.py`: a registry with a gap (ids 0, 1, 3) and a secondary
  model whose index 2 is named like registry id 3: the item gets id 3 (today it gets 2, or `None`
  as the name).
- Control: a registry in model order gives the same ids as today.
- 1E-4b (next task) applies the same name-first mapping to the primary `assigns_class` path.
- Note for 1B: this file reads `self.registry`; track 1B converts it to the document snapshot after
  1E merges (merge order puts 1E first).

### 1E-4b Primary detector class by name (S)

Decision (formerly Q-1E-1, closed): the primary detector, like the secondary one, uses the model
index as the registry id. That breaks the owner rule "class identity is the name, never the index",
and it only works while the registry was seeded in ONNX order (`seed_class_registry.py --check`
detects drift after the fact). It is fixed in B1, not left to #368 or B3.

- Files: `src/services/curation/ingest_detect.py` (the `assigns_class` branch of the end2end decode:
  `entry = self.registry.get(cls_id)`, then `class_id=cls_id`, `detector_class_id=cls_id`);
  find any other registry-by-model-index site with `rg -n "registry\.get\(" src/services/curation`
  and `rg -n "assigns_class" src`. The non-assigning branch already keeps the proposer's own label
  (`proposer_label`) and is unchanged.
- Pattern: the same helpers as 1E-4: model output index -> name through the model's own names
  (`src/utils/class_names.py::get_class_names(model_name)` or the ONNX `names` metadata), then
  `resolve_class_by_name` on the registry's classes. Documented Ultralytics fact: a detection's
  class is an index into `model.names`, which is the only stable identity the model owns
  (docs.ultralytics.com, Predict mode, "Boxes": `cls`; `names`).
- Fix: `class_id` and `class_name` come from the registry class found by name. `detector_class_id`
  is the registry id of that class (not the raw index); the raw model index is never stored as an
  identity. A model name with no active registry class stays an unlabeled proposal
  (`{name}_low_conf` path rules unchanged) and is counted in the ingest log by name.
- Red: `tests/test_primary_class_by_name.py`: a registry seeded out of order (ids 0, 1, 3, model
  index 2 named like registry id 3): today the item gets the entry of registry id 2 (or none), after
  the fix id 3. Control: a registry in model order gives the same ids as today.
- Done when: no `self.registry.get(<model index>)` remains in `src/services/curation` (`rg` shows only
  by-name lookups); the CHANGELOG notes the behaviour (a registry out of ONNX order now labels
  correctly).
- Size: S. Order: after 1E-4 (same file). 1B converts the registry reads after 1E merges.

### 1E-5 U-Step 3: the YOLOv5 objectness probe reads its input size from the ONNX (S)

- File: `src/services/curation/probe_models.py` (`YOLOV5_OBJ_INPUT_SIZE`,
  `_build_yolov5_objectness_predictor`, `_letterbox_yolov5_objectness`).
- Fix: read the input size from `session.get_inputs()[0].shape` (fall back to 1280 only when the
  spatial dims are dynamic, and name it as that fallback); refuse an output that is not
  `[.., N, 5 + nc]` with `nc == len(names)` from the ONNX metadata, at build time, with a message.
- Red: `tests/test_yolov5_onnx_probe.py` (marker `families`): the real 640 static official export
  predicts a class name present in its metadata (fails today: "Got 1280 Expected 640"); a real
  `--dynamic` export still works at 1280; `apply_ensemble_nms` returns detections on Ultralytics'
  `bus.jpg` (no fakes).
- Control: `tests/curation/test_probe_models.py` (fake onnxruntime) passes.

### 1E-6 CI job `families` (S)

- File: `.github/workflows/ci.yml` (a new job `families`; B3 extends it to the full U-6.1 matrix,
  it is not rewritten).
- Steps: Python 3.12, CPU torch, `requirements-test.txt` (so the job tests the exact pin);
  `actions/cache` for `~/.cache/op-test-weights` keyed on the Ultralytics pin (hash of
  `requirements-test.txt`) and the fixture list (hash of `tests/_ultralytics_weights.py`); a step
  that, on a cache miss, produces the original-YOLOv5 ONNX in a separate venv from the official
  `ultralytics/yolov5` repo at a pinned commit SHA (record the SHA in the workflow; the evidence used
  that repo's master on 2026-11-03); then `python -m pytest tests/ -q -m families --no-cov`.
  Runtime budget 15 minutes (`timeout-minutes: 15`), CPU only.
- Red: the job fails on the base branch with the 1E tests added (they are red before 1E-1..1E-5).
- Control: the existing `pytest` job no longer runs the `families` tests and its runtime does not
  grow.
- Licence note in the workflow comment: the weights and the yolov5 repo are AGPL-3.0, used for tests
  only, not redistributed.

### 1E-7 Docs (S)

- `docs-site/docs/configuration/advanced.mdx` (`OP_INGEST_SECONDARY_*`): the secondary detector
  takes an original-YOLOv5 ONNX exported by the official repo (bring your own), with the command,
  and maps classes by name; YOLOv5u for GPU serving.
- `export/README.md`: uploads route by head kind; YOLO26 and YOLOv10 export with `nms=False`; the
  head check refuses anything but `[batch, 300, 6]`.
- `docs/CURATION.md` probe section if it states the 1280 input (`rg -n "1280|yolov5_objectness"
  docs docs-site`).
- Run `scripts/docs/check_docs_vs_code.py --only <file>` for each.

## 6. Gate for this lane

- Targeted: the four red modules with network (`-m families`), each seen red on the base, plus the
  controls; `.venv/bin/pre-commit run --files <touched files>`.
- No full suite, no `--all-files`. The sweep runs the `families` job in CI and once locally in the
  export image (`docker run --rm --cpus 8 ...` under `-p op-b1-ultralytics` naming, local-only
  image tag), see `02_b1_overview.md` section 7.

## 7. Done when

- A YOLO26 upload on 8.4.197 is routed to the native path and exports `[batch, 300, 6]`, or is
  refused; a raw-head ONNX is refused at export and at promote.
- `rg -n "'end2end'" src/services/model_export.py export/export_worker.py` finds nothing.
- The secondary detector never stores a raw model index as a class id.
- The probe works on 640 and 1280 static and on dynamic official YOLOv5 exports.
- The `families` CI job is green and under 15 minutes.

## 8. Docs, contracts, CHANGELOG

- Contracts: none (`end2end` / `head_kind` are internal task and job fields; the upload routes keep
  their shapes).
- CHANGELOG `[Unreleased]`, Fixed: "#447: YOLO26 uploads export the NMS-free `[batch, 300, 6]`
  head (`nms=False`), and any other head is refused at export and promote"; "a YOLO26 upload is no
  longer routed to the YOLO11 toolchain (routing by head kind)"; "the ingest secondary detector
  maps classes by name"; "the YOLOv5 objectness probe reads its input size from the ONNX". Changed:
  "YOLOv10 uploads use the NMS-free export path". Because there is no intermediate release (owner
  decision 3), B0 added no "known issue" line for #447; there is none to remove (check with
  `rg -n "447" CHANGELOG.md`).

## 9. Upgrade impact

- Export jobs queued before the upgrade (carrying `end2end`, no `head_kind`) fail closed with
  "re-upload the model".
- No engine is rebuilt by this track; the default detector is unchanged (that is B3).

## 10. Rollback

Revert the 1E merge commit alone; no coupling. The CI job goes with it.

## 11. Risks

| Risk | Mitigation |
|---|---|
| A checkpoint ignores `nms=False` (#447 evidence conflict, U-R7) | the head check refuses it at export time |
| The fixtures' download hosts change | sha256 pinned per file; the CI cache holds the last good copies; the skip/fail rule makes it visible |
| The yolov5 repo's requirements drift | pinned commit SHA; separate venv; cached ONNX |
| The `families` job grows the CI bill | 15-minute budget, CPU, cached weights; B3 extends it deliberately |

## 12. Corrections found while verifying

- `pyproject.toml` runs pytest with `--strict-markers`: the former master plan's marker `export` was never
  registered; this plan uses one registered marker, `families`, for every test that fetches weights.
- The former master plan put the 1E tests both under `tests/` and `tests/export/`; this plan keeps the
  four names under `tests/` (as the master's test list does) and leaves `tests/export/` to B3's
  `test_model_families.py`.
- The ingest primary path for `assigns_class` profiles also uses the model index as the registry id
  (fixed by 1E-4b).

## 13. Open questions

- **Q-1E-1 Primary detector class by name: CLOSED.** Decided: the name-first mapping applies to the
  primary `assigns_class` path too, as task 1E-4b (not left to #368 or B3).

## 13.1 What to keep after merge

| Fact or section | Target doc |
|---|---|
| head kinds (`one2one`, `nms`), the refused YOLOv5 original-weights message, upload routing by head kind | `docs/CURATION.md` (upload and export), `export/README.md` |
| class-by-name rule for primary and secondary detectors | `docs/ARCHITECTURE.md` (class identity) |
| the `families` CI job, the fixtures and the pinned yolov5 export commit | `CONTRIBUTING.md` |
| behaviour changes (YOLOv10 NMS-free path, jobs queued before upgrade re-upload) | `CHANGELOG.md` |

Done when also: the closing B1 PR deletes this file after the keep-list is moved
([00_index.md](00_index.md) "Plan lifecycle").

## 14. Lane agent brief (paste as is)

```
You are the implementation agent for B1 track 1E (Ultralytics live bugs) of OpenProcessor.
Plan: docs/design/program/02e_b1_ultralytics_live_bugs.md (this file). Execute it standalone.
Read first: CLAUDE.md, docs/VISION_AND_GOALS.md, docs/design/engineering_standards.md, and
docs/design/model_family_versions_plan.md sections 3, 4 (C1, C2, C8, C10), 5 (J3, J4), 7 Steps 1-3.

Worktree: <repo>-wt-b1-ultralytics, branch b1/ultralytics (created by the coordinator from
int/b1-correctness). Work ONLY there. Never push, never merge, never check out another branch.
Compose project name if you run any container: op-b1-ultralytics; image tags local-only (never a
public tag: watchtower re-pulls public tags). GPU: none (CPU exports only). Deadline: <fills in>.

Rules:
- Tasks 1E-0 .. 1E-7 (including 1E-4b), one commit each, conventional commits, hooks on.
- Do not change the Ultralytics version (==8.4.197 stays) and do not touch the trainer, the
  vendored patch, /opt/venv-y11 or half= (all B3).
- Use only Ultralytics' documented nms export argument (cite docs.ultralytics.com/modes/export);
  never end2end in new code or tests.
- Red test first, run on unchanged code, SEE IT FAIL, paste the red output into the commit body;
  then fix; paired control green.
- Weights come from the official release assets into ~/.cache/op-test-weights with pinned sha256;
  nothing is committed; the original-YOLOv5 ONNX is made by the official ultralytics/yolov5
  export.py in a throwaway venv, never in the project venv.
- Targeted tests only (-m families for the four new modules, plus controls) and
  .venv/bin/pre-commit run --files <touched files>. No full suite, no --all-files.
- Never use pgrep -f, pkill -f, or sleep/until polling loops.

Report at the end: commits (sha + subject), red outputs in one line each, the full sha256 of each
fixture, the yolov5 repo SHA used, the CI job runtime if you could measure it, the answer assumed
for 1E-4b, and any plan correction.
```
