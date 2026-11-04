# B3: one Ultralytics version and one detection contract

| Field | Value |
|---|---|
| Status | plan, not started |
| Release | part of the single **v0.7.0** release (B0-B5 plus R-final; D-37 approved in [00a_decision_register.md](00a_decision_register.md) section 6). No tag at the end of B3. R-final (`07_release_acceptance.md`) validates it last, after B5 |
| Depends on | B2 merged into main (decision D-13 order: B0, B1, B2, B3, B4, B5, R-final); B1-1E done (B1 1E (merged; see CHANGELOG.md)); B0 exact pin `ultralytics==8.4.197` in `requirements.txt`, `requirements-test.txt` and `pyproject.toml` (commit `d1c6bab0`) and `tests/test_requirements_contract.py` (`13_b0_close_out.md`); GPU window 1 ([00a_decision_register.md](00a_decision_register.md) section 8.1; `11_risks_cleanup_host.md`) |
| Blocks | B4 (`05_b4_stateless_api.md`) retakes its 32-worker baseline on the B3 engine; B5 (`06_b5_overview.md`) measures the stack as it stands after B4 |
| Decisions applied | D-2 (B0 pin 8.4.197 is the starting point), D-10 (B1-1E minimal; trainer `nms`, export module, full matrix are here), D-11 option b (original YOLOv5 = bring-your-own ONNX for probe and secondary only), D-12 (`nms` by capability), D-13 (order), owner decisions of 2026-11-03 items 2, 5, 6, 7 |
| Open decisions | **D-27** (U-J1 staging and U-J5..J8), **D-28** (pin choice rule and freeze date); status in [00a_decision_register.md](00a_decision_register.md) section 6, needed at B3 kick-off; section 3 holds the detail |
| Source IDs | former master plan section 7.3 (moved here), U-Step 4-9, U-J1, U-J4..J8, U-C3, U-C6, U-C7, U-C9, U-C11, U-R1..R8, U-6.1, U-6.2 (`docs/design/model_family_versions_plan.md`); GPU-14 final, GPU-18 (`docs/design/codebase_audit_plan.md` 4.3); N8; #368 first slice; #447 trainer part |
| Verified | Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read-only `rg`, nothing run). Line numbers move; re-find every site by the symbol or the `rg` pattern given |
| Consolidated | Consolidated 2026-11-03 |

## 1. Goal

"The right model with the right code." After B3:

- one exact Ultralytics release `X` in every image and requirements file (API, export, trainer);
- one detection contract for every Ultralytics family: a single fused `output0 [batch, 300, 6]`
  (`x1, y1, x2, y2, score, class`, letterboxed input pixels), produced by Ultralytics' documented
  `nms` export argument;
- no `/opt/venv-y11`, no vendored fork (`src/ultralytics_patches/`), no deprecated
  `EfficientNMS_TRT` plugin (Stage B), no box-frame guess;
- the trainer exports promoted models to the same contract on the same release;
- one family registry module (`src/domain/model_families.py`) read by the probe, the upload
  validator, the exporter and the trainer (first slice of #368);
- the CPU family matrix (U-6.1) runs in CI at the pin on every PR that touches it;
- detector throughput holds at least 95 % of today's engine, or Stage A ships alone.

Mission fit: correctness first (no silently wrong engine for any family, class by name stays),
then speed (no throughput loss; B5 then measures on this engine once).

## 2. Preconditions (check each before kick-off)

| # | Check | How |
|---|---|---|
| P1 | B2 merged into main | `git log --oneline -1 main`; the B2 PR merge commit is present |
| P2 | B1-1E merged: `nms=False` in `export_yolo26.export_fused_onnx`, `inspect_fused_output` refuses non-`[300,6]`, `head_kind` recorded by `validate_pytorch_model`, the export worker routes on `head_kind`, class by name on the secondary, YOLOv5 probe input read from the ONNX, CI job `families` exists | `rg -n "nms=False" export/export_yolo26.py`; `rg -n "head_kind" src/services/model_export.py export/export_worker.py`; `rg -n "families" .github/workflows/` |
| P3 | B0 exact pin: every requirements file says `ultralytics==8.4.197`; `tests/test_requirements_contract.py::test_ultralytics_is_one_exact_version_everywhere` exists | `rg -n "ultralytics" requirements*.txt pyproject.toml`; `rg -n "exact_version" tests/test_requirements_contract.py` |
| P4 | D-27 and D-28 answered by the owner; freeze date written into the B3 PR description | owner message |
| P5 | GPU window booked: GPU 0 for about one GPU day, announced to peer sessions; nothing else heavy from this repo at the same time | `11_risks_cleanup_host.md` |
| P6 | Reference artifacts exist or are scheduled as gate step G0 (images from main before B3, a checkpoint from the 8.4.48 trainer) | section 6 |

At `af62c947` P2 and P3 are **not** met yet (expected: B0 and B1 come first). See Corrections.

## 3. Decision detail (D-27, D-28, D-38; status lives only in [00a_decision_register.md](00a_decision_register.md) section 6)

### D-27: U-J1 staging, and U-J5..J8

| Item | Recommendation | Consequence | Needed by |
|---|---|---|---|
| U-J1 staging | One version. Build both a Stage A engine (patch kept on `X`, venv deleted) and a Stage B engine (`nms=True`, patch deleted) and judge both in **one** U-6.2 GPU gate. Ship Stage B if it passes; else revert the Stage B merges and ship Stage A | deployments rebuild the default detector engine once; one GPU day instead of two | B3 kick-off |
| U-J5 conf | keep `conf=0.25` baked into `nms=True` exports (today's YOLO11 behaviour); NMS-free YOLO26/v10 always return 300 rows, so the API confidence filter stays load-bearing; `FusedDetAdapter`'s `score > 0` stays only as a padding filter | requests below 0.25 get nothing extra, as today | B3 kick-off |
| U-J6 name | keep the Triton name `yolov11_small_trt_end2end` (and the ONNX dir `models/yolov11_small_end2end/`) | no rename churn across about 30 files; the name still means "NMS inside the engine" | B3 kick-off |
| U-J7 batch | export `batch` = Triton `max_batch_size` = TRT profile max for `nms=True`; the config writer reads `batch` from the ONNX metadata and refuses a mismatch | prevents the silent empty images past the export batch (U-C7) | B3 kick-off |
| U-J8 trainer | trainer on the same release `X` in this batch; its promote export calls the shared export settings (registry) | the trainer changes once, behind the GPU gate's promote check | B3 kick-off |

Alternative to U-J1: two separate gates (Stage A first, Stage B in a later batch). Costs a second
GPU day and a second engine rebuild for users. Not recommended.

### D-28: pin choice rule and freeze date

| Recommendation | Consequence | Needed by |
|---|---|---|
| `X` = the newest **8.4.x** release on PyPI that is at least 3 days old on the **freeze date** (set by the coordinator at B3 kick-off and written into the PR) and passes U-6.1 at that version. If no release passes, `X` = the newest that does; if none since 8.4.197 passes, `X` = 8.4.197 and Stage A/B still run on it. An 8.5.x release needs a separate owner yes. Any bump after the freeze reruns U-6.1 and U-6.2 | deterministic for an agent; the PR records `X`, the freeze date, the release-note scan and the matrix output | B3 kick-off |

Evidence: 8.4.201 passed every CPU check on 2026-11-03 (`model_family_versions_plan.md` 3.3);
Ultralytics ships several 8.4 releases a day (U-R2), hence the freeze.

### D-38 (owned by `00a_decision_register.md`; affects B3 only if placed here)

The base-image bump of #432 (Python 3.13 -> 3.14, `nvidia/tritonserver` 26.06 -> 26.09) moves
TensorRT and rebuilds every engine. `09_github_actions.md` recommends taking it in B3, which already
rebuilds engines and runs a GPU gate. If the owner places it here, it is task B3-A5 below; if not,
B3 does not touch the base images.

## 4. Branches, worktrees, tracks and merge order

- Integration branch: `int/b3-ultralytics`, cut from main **after B2 merged** (owner decision 6).
  Lane coding may start earlier only on the coordinator's say-so; such lanes rebase onto
  main-after-B2 before their merge. Review fix R-18: B3 **does** touch B2 files: B3-E2 edits
  `export/export_worker.py`, which B2 lane 5B rewrites, and Stage A edits `docker/trainer/Dockerfile`
  and the root `Dockerfile`, which 5A and 5B edit. So every B3 track re-verifies each file and symbol
  it names on the post-B2 head before coding (this file's "verified against" note predates B2), and
  only `frame` (adapters and `affine.py`, no B2 file) may code before B2 merges.
- Each track: branch `b3/<track>`, worktree `<repo>-wt-b3-<track>`, one Sonnet agent.
- Tracks merge into `int/b3-ultralytics` with `git merge --no-ff`, one merge commit per track, in
  this order. A failed stage reverts as a group (`git revert -m 1 <merge>` per track).

| Order | Track | Stage | Scope | Size | Based on |
|---|---|---|---|---|---|
| 0 | `deps` (branch `deps/pins-part-b`) | A | pins part B of [01](01_package_pins.md) section 7: the bumps that can change model output (numeric stack with numpy and pandas, OpenCV 5 with Pillow and simplejpeg, the Hugging Face stack, torch and torchvision, the onnx, onnxruntime, modelopt and TensorRT set, CuPy 14, the image-only pins), then the MLflow and Dependabot tail. Owner direction 3 of 2026-11-04: they ride this GPU gate so parity runs once. Skips `requirements-export-y11.txt` (B3-A2 deletes it) | M | `int/b3-ultralytics` (parallel with `frame`) |
| 1 | `stagea` | A | pin `X`; venv and re-exec deleted; patch runs in the main env; trainer on `X` with `nms` by capability; `pytorch_utils.py` deleted | M | `int/b3-ultralytics` after `deps` merged (both edit the manifests) |
| 2 | `families` | A | `src/domain/model_families.py` (#368 slice) and its readers; full U-6.1 matrix in `tests/export/`; CI `families` job extended | M | `b3/stagea` (needs the pin) |
| 3 | `frame` | A | GPU-14 final: each adapter declares its box frame, the `max() > 1.0` guess deleted | S | `int/b3-ultralytics` |
| - | **Stage A point** | | coordinator records the SHA of the integration head after merges 0-3 in the PR description (no tag); engine EA therefore carries every pin-series bump and the Ultralytics pin together | | |
| 4 | `export` | B | U-Step 4: one export module, one worker command, scripts, the fused YOLO11 config, J7 guard, U-C11, formats collapsed, patch deleted, engine stamp recipe | L | Stage A point |
| 5 | `decoders` | B | U-Step 5 and 6: fused-only adapter, ingest primary fused (U-C9), region detector raw-head path removed, fail closed | M | Stage A point |
| - | **GPU gate** (section 6) | | U-6.2 once for both stages | 1 GPU day | integration head |
| 6 | `docs` | shipped stage | U-Step 8, GPU-18, families table, CHANGELOG "Breaking" | S | head after the verdict |

Then one T1 sweep on the integration head (section 7), one PR to main, a merge commit.

**Batch size:** L plus about one GPU day. The two-stage split exists so that the version change
(Stage A) and the contract change (Stage B) can be judged in one gate and, if needed, separated by
reverting merge commits; this keeps D-10's promise that B1-1E's export code is moved here, not
rewritten twice (master Appendix A.5 items 1 and 2).

## 5. Tasks

Every task: red test seen failing on the track's base before the fix, plus a paired control
that passes before and after (record both outputs in the commit body or PR); `make contracts`
in the same commit as any wire change; lane checks only (section 9).

### Track `stagea` (Stage A: one version, no second environment)

**B3-A1 Pin `X` everywhere.** Size S.
- Files: `requirements.txt`, `requirements-test.txt`, `pyproject.toml` (`ultralytics==8.4.197`
  today in the dependency list), `tests/test_requirements_contract.py` (from B0, `test_ultralytics_is_one_exact_version_everywhere`).
- Pattern: exact `==` pin (U-J2); PyPI JSON for the release list
  (`https://pypi.org/pypi/ultralytics/json`).
- Steps: replace `8.4.197` with `X` in each file; rewrite the `requirements.txt` comment (it still
  says the YOLO11 toolchain needs `<8.4`, false per U-C4).
- Red: extend the pin test to also read the trainer's `ULTRALYTICS_VERSION` comment (B3-A3) and
  fail on any mismatch; red because the trainer is on 8.4.48. Control: the requirements files
  agree with each other.
- Done when: `rg -n "ultralytics[=><]" requirements*.txt pyproject.toml` shows only `==X`.

**B3-A2 Delete the second environment.** Size M.
- Files: `Dockerfile` (the `builder-export` RUN block that creates `/opt/venv-y11`, and the
  `COPY --from=builder-export ... /opt/venv-y11` line in the export stage), `requirements-export-y11.txt`
  (delete), `requirements-export.txt` (comment on the TensorRT line names the y11 file),
  `export/export_models.py` (`_ultralytics_is_pre_84`, `_Y11_VENV_PYTHON`, the `os.execv`
  re-exec, `YOLO11_EXPORT_PYTHON`), `export/yolo_trt_build.py` (comment on the venv's CPU torch),
  `docker/buildlib/dedupe_site_packages.sh` (delete if `rg dedupe_site_packages` finds no other
  caller).
- Steps: the vendored patch now runs in the main environment (it produced the same 4-output graph
  on 8.4.197 and 8.4.201, U-C4). Nothing else in Stage A changes the YOLO11 recipe.
- Red: `tests/test_dockerfile_hygiene.py` gains "no `/opt/venv-y11` in any Dockerfile" and
  "no `YOLO11_EXPORT_PYTHON` read in `export/`"; red today. Control: the existing hygiene checks
  (trainer ARG, non-root user) still pass.
- Done when: `rg -n "venv-y11|YOLO11_EXPORT_PYTHON|requirements-export-y11" --glob '!docs/**'` is empty.

**B3-A3 Trainer on `X`, promote export by capability.** Size M.
- Files: `docker/trainer/Dockerfile` (both `ARG ULTRALYTICS_SHA=` lines; add a
  `# ULTRALYTICS_VERSION=X` comment the pin test reads), `docker/trainer/trainer_fit.py::_export_onnx`,
  `docker/trainer/trainer.py` (the val-pass head toggle, `rg -n "one2one" docker/trainer/trainer.py`).
- Pattern: Ultralytics export `nms` argument ("`True` embeds NMS where supported"; `False`
  selects the NMS-free head; `None` exports the raw one-to-many head) and `quantize` replacing the
  deprecated `half` (https://docs.ultralytics.com/modes/export/, read 2026-11-03).
- Steps:
  1. `ULTRALYTICS_SHA` = the commit SHA of tag `v<X>` (`gh api repos/ultralytics/ultralytics/git/ref/tags/v<X>`), U-J8.
  2. `_export_onnx` calls the registry's export settings (B3-F1): `nms=False` if the head has the
     one-to-one branch, else `nms=True` with `agnostic_nms=True, iou=0.7, conf=0.25, max_det=300`;
     drop `half=False` (the default is FP32; precision stays with the ModelOpt/TRT step), U-C6.
  3. The trainer image cannot import `src/`. Copy the pure registry module in, exactly like
     `src/services/training/augmentation_presets.py` is copied today
     (`COPY --chown=appuser:appuser src/domain/model_families.py ./model_families.py`). The
     `stagea` agent creates `src/domain/model_families.py` with only `head_kind_of` and
     `export_kwargs` (stdlib only); the `families` track (based on `b3/stagea`) extends it.
  4. Verify the val toggle: `trainer.py` checks `hasattr(head_module, 'one2one')`, the plan's
     capability is `one2one_cv2`. Use the registry's `head_kind_of` in both places.
- Red: `tests/trainer/test_export_onnx_nms.py` with two fake models (head with and without
  `one2one_cv2`) and a recording `export`; asserts the `nms` value and no `half` kwarg; red today
  (no `nms` passed). Control: the call keeps `format='onnx', dynamic=True, batch=export_batch, simplify=True`.
- Done when: the trainer's export kwargs come from the registry; the GPU gate's G6 passes.

**B3-A4 Delete the dead module.** Size S.
- Files: `src/utils/pytorch_utils.py`, `src/utils/README.md` (two references),
  `scripts/codegen/baselines/broad_except.txt` (its line; baselines only shrink).
- Check first: `rg -n "pytorch_utils" --glob '!docs/**'` lists only those; the vulture ratchet
  (`scripts/codegen/check_vulture.py`) agrees.
- Red: none needed (deletion); control: the offline import smoke (`tests/` collection) passes.

**B3-A5 Base-image bump (only if D-38 places it in B3).** Size M.
- Files: the base `FROM` lines (root `Dockerfile`, `docker/test/`, `docker/trainer/Dockerfile`),
  `Dockerfile.triton` (`ARG TRT_VERSION`), `requirements-export.txt` (`tensorrt-cu13`), the
  pyproject `export` extra; the y11 requirements file is already gone (B3-A2).
- Rule: TensorRT pins move together in one commit; `test_tensorrt_pin_matches_the_triton_image_everywhere`
  stays green (red first: bump one pin alone and watch it fail).
- Gate effect: EA and EB are then built on the new TensorRT while E0 is on the old one; G3 and G4
  compare across both changes. Record E0 rebuilt on the new base as an extra row (E0') so the
  Ultralytics change and the TensorRT change can be told apart. All engines rebuild on upgrade
  (every group's stamp records the TensorRT version).

### Track `families` (Stage A: registry and the full CPU matrix)

**B3-F1 `src/domain/model_families.py` (#368 first slice).** Size M.
- Rule: `src/domain/` imports only `config` and `utils` (layering ratchet). This module imports
  **stdlib only**, so the trainer can copy it.
- Content: a frozen `FamilySpec` per family (`yolov5_original` (ONNX only), `yolov5u`, `yolov8`,
  `yolov9`, `yolov10`, `yolo11`, `yolo12`, `yolo26`) with `head_kind` (`one2one` | `nms` |
  `onnx_objectness`), `pt_supported`, `probe_architecture`, `decoder` (`fused` | `objectness`);
  `head_kind_of(head_module) -> str` (`one2one` iff `hasattr(head_module, 'one2one_cv2')`, U-C2);
  `export_kwargs(head_kind, max_batch) -> dict` (the J4/J5/J7 arguments of U-Step 4);
  `PROBE_ARCHITECTURES` derived from the specs.
- Readers (replace the duplicates): `src/services/curation/probe_models.py`
  (`PROBE_ARCHITECTURES` tuple), `src/routers/curation/probe.py` (imports it),
  `src/services/model_export.py::validate_pytorch_model` (the B1-1E `head_kind` logic moves here
  unchanged in behaviour), `docker/trainer/` (B3-A3), `export/ultralytics_export.py` (B3-E1).
- Red: `tests/domain/test_model_families.py`: `head_kind_of` on stub heads, `export_kwargs` per
  kind, and an AST test that no other module defines an architecture tuple
  (`rg -n "\('yolo11', 'yolo26'" src` must find only the registry). Red today. Control: the
  probe route's accepted architectures are unchanged (`yolo11`, `yolo26`, `yolov5_objectness`).
- Done when: one definition; every reader imports it.

**B3-F2 Full U-6.1 matrix at the pin.** Size M.
- File: `tests/export/test_model_families.py` (marker `families`), parametrized over
  `yolov5nu, yolov8n, yolov9t, yolov10n, yolo11n, yolo12n, yolo26n, yolo26s` plus the original
  `yolov5n` ONNX. Weights from the official release assets (`github.com/ultralytics/assets/releases`,
  AGPL-3.0); the original-YOLOv5 ONNX produced by the official `ultralytics/yolov5` repo's
  `export.py` (AGPL-3.0). Nothing committed; a session fixture caches into `~/.cache/op-test-weights`
  (the B1 fixture; extend, do not replace). `yolo12n` is optional: drop the row if the asset is
  missing at the pin and say so in the PR.
- Assertions per family, through the production functions (U-6.1 items 1-7):
  1. `.pt` load via `ultralytics.YOLO` and `validate_pytorch_model`; original YOLOv5 refused with
     a message naming YOLOv5u;
  2. `head_kind_of` equals the registry;
  3. production export (`export/ultralytics_export.py` after B3-E1; in the Stage A commit, the
     B1 `export_fused_onnx` for one2one families) at `batch=8`: one output `[batch, 300, 6]`;
     `NonMaxSuppression` count 8 for `nms`, `TopK` present for `one2one`;
  4. onnxruntime on the two Ultralytics sample images: `nms` families equal a torchvision NMS oracle over
     the raw-head export (matched fraction 1.0 at IoU >= 0.95, same class, score diff < 1e-4);
  5. over-batch: the `batch=8` ONNX at batch 10 returns empty rows 9-10 (documents U-C7), and the
     J7 writer guard refuses a config whose `max_batch_size` differs from the ONNX `batch`;
  6. original YOLOv5 ONNX: `_build_yolov5_objectness_predictor` returns a name in the metadata;
     `apply_ensemble_nms` returns detections on an Ultralytics sample image (no fakes);
  7. pin: `importlib.metadata.version('ultralytics') == X`.
- **Discrepancy rows (ledger, must be explicit):** `yolo26n` and `yolo26s` exported with
  `nms=False` and with `simplify=True` **and** `simplify=False`: all four give `[batch, 300, 6]`
  plus `TopK`. If one gives the raw head, the head check must refuse it (the test asserts the
  refusal, never a pass of a raw head). Also an AST/grep test that production code never passes
  `end2end=` (deprecated and mapped to `nms`, maintainer comment on the export page).
- Fold the B1-1E tests (`tests/test_export_yolo26_head.py`,
  `tests/test_validate_pytorch_model_families.py`) into this module only where a case becomes a
  strict duplicate; delete the duplicate in the same commit. Do not redo B1's tests.
- Red: the `yolo11n` row of item 3 fails on the Stage A tree (the patch still emits 4 outputs);
  mark it `xfail(strict=True, reason="Stage B")` in the Stage A commit and remove the mark in
  B3-E1, so the matrix proves the switch. Control: the `yolo26n` and original-YOLOv5 rows pass on both.
- Done when: the job passes at `X` in CI and once locally in the export image built from the
  integration head (CPU, `--cpus 8`).

**B3-F3 CI job `families` extended, not rewritten.** Size S.
- File: `.github/workflows/ci.yml` (the job B1 added). Conventions: `09_github_actions.md`.
- Steps: run `tests/export/` too; `actions/cache` key = pin + fixture list hash; budget stays
  15 min CPU; never skipped in CI (skips with an explicit reason only when offline locally).
- Done when: the job is green on the PR; its duration is in the PR description.

### Track `frame` (Stage A: GPU-14 final)

**B3-G1 Each adapter declares its box frame; the guess goes.** Size S.
- Files: `src/clients/model_adapters.py` (`End2EndNMSAdapter`, `FusedDetAdapter`),
  `src/utils/affine.py::inverse_letterbox_coords` (`is_pixel_coords = float(boxes_out.max()) > 1.0`),
  callers (`rg -n "inverse_letterbox_coords\(" src`).
- Design: an explicit `box_frame` (`normalized` | `pixels`) carried by the adapter (the patch
  exports `normalize_boxes=True`, so `End2EndNMSAdapter` is `normalized`; the fused contract is
  `pixels`). `inverse_letterbox_coords` takes the frame as a required argument. This supersedes
  the audit's "Triton model parameter" idea: the contract itself fixes the frame. It survives a
  Stage A fallback.
- Red (audit GPU-14): `[[.1,.2,.3,.4],[.5,.5,641/640,.9]]` with frame `normalized` returns the
  first box unchanged; red today. Control: boxes at or below 1 give today's result.
- Done when: `rg -n "max\(\) > 1|> 1\.0" src/utils/affine.py` is empty.

### Track `export` (Stage B: one export module)

**B3-E1 `export/ultralytics_export.py`.** Size L.
- Replaces: `export/export_models.py`, the end2end parts of `export/yolo_formats.py`
  (`export_onnx_end2end`, `export_trt_end2end`), the export part of `export/export_yolo26.py`;
  also review `export/yolo_export_cli.py`, `yolo_export_settings.py` (`HALF`),
  `yolo_model_config.py`, `yolo_triton_config.py`, `yolo_trt_build.py`, `yolo_export_report.py`
  and keep only what the one module needs (split by concern if a file passes the size baseline).
- Moves unchanged from B1-1E: `inspect_fused_output` and the `head_kind` routing.
- Call: `YOLO(pt).export(format='onnx', imgsz=640, dynamic=True, batch=max_batch, simplify=True,
  device='cpu', **export_kwargs(head_kind, max_batch))`; then the head check; then the existing
  ModelOpt FP16 (`ultralytics.utils.export.engine.modelopt_quantize_onnx`), `add_uint8_input`
  (`src/utils/uint8_input.py`) and the TRT build from `export_yolo26.build_engine` (profile max =
  `max_batch`); then `write_triton_config` (single `output0 [300, 6]`, output FP32, U-R4).
- Pattern: Ultralytics export docs (`nms`, `agnostic_nms`, `conf`, `iou`, `max_det`, `batch`);
  TensorRT builds `INMSLayer` from ONNX `NonMaxSuppression` (onnx-tensorrt operator table,
  TensorRT INMSLayer API; links in `model_family_versions_plan.md` section 10).
- Red: B3-F2's `yolo11n` row (remove its `xfail`). Control: the `yolo26n` row is byte-for-byte the
  same head as on Stage A.

**B3-E2 One worker command, one set of callers.** Size M.
- Files: `export/export_worker.py` (`EXPORT_SCRIPT`, `EXPORT_SCRIPT_Y26`, `build_command`,
  `engines_to_load`), `scripts/lib/export_steps.sh`, `scripts/lib/model_groups.sh`, `Makefile`
  (targets near `export_models.py ... --formats trt trt_end2end --normalize-boxes`),
  the `openprocessor` CLI `export`, `export/README.md` command lines.
- Red: `tests/export/test_export_worker_command.py`: one command for a `one2one` and an `nms`
  job; red today (two scripts). Control: `engines_to_load` names the same Triton model.

**B3-E3 Fused YOLO11 config and the J7 guard.** Size S.
- Files: `models/yolov11_small_trt_end2end/config.pbtxt` (4 outputs, FP16 `det_scores` today) ->
  `output0 [300, 6]` FP32; the config writer; `tests/test_committed_model_configs.py` renders the
  committed file from the writer.
- J7: the writer reads `batch` from the ONNX metadata and refuses `max_batch_size != batch`.
- Red: the committed-config test against the new writer; the guard test with a mismatched ONNX.

**B3-E4 U-C11: delete the installer heredoc config.** Size S.
- File: `scripts/lib/config_models.sh` (the `cat > "$config_file" << EOF` block writing
  `name: "yolov11_small_trt_end2end"` with `TYPE_FP32`). The exporter writes the config.
- Red: `tests/installer/` check that no shell file writes a `config.pbtxt` for a detector; red
  today. Control: the other heredocs in that file, if still needed, are unchanged (check each with
  `rg -n "cat > \"\$config_file\"" scripts/lib/config_models.sh`; delete those the exporters cover).

**B3-E5 Export formats collapse (wire).** Size S.
- Files: `src/routers/models_export.py` (`onnx_end2end`, `trt_end2end`), `src/schemas/model_admin.py`
  (`normalize_boxes` field), `contracts/` via `make contracts`, `docs/design/curation_api_contract.md`
  if it lists them. Keep `src/routers/models_lifecycle.py`'s `_trt_end2end` model-name suffix (U-J6).
- Red: a route test that `onnx_end2end` is now a 422; control: `onnx` and `trt` accepted.

**B3-E6 Delete the vendored fork.** Size S.
- Files: `src/ultralytics_patches/` (all), `pyproject.toml` (ruff, coverage and mypy exclusions for
  `src/ultralytics_patches`), `requirements-export.txt` and the pyproject `export` extra
  (`onnx-graphsurgeon` is used only by the patch: `rg -ln graphsurgeon src export scripts`; drop it
  if that stays true), `ATTRIBUTION.md` and `README.md` licence lines on the fork.
- Done when: `rg -n "ultralytics_patches|EfficientNMS_TRT" --glob '!docs/**' --glob '!CHANGELOG.md'` is empty.

**B3-E7 Engine stamp forces the rebuild (U-R3).** Size S.
- Files: `scripts/lib/model_stamps.sh` (`_ms_build_flags`; the stamp row is model, precision,
  reason, tensorrt, compute_cap, onnx_sha256, flags), `scripts/lib/model_setup.sh`,
  `tests/installer/test_engine_stamps.py`.
- Fact: an upgrade keeps the old ONNX, so its sha256 matches and nothing rebuilds. Add an
  export-recipe token to the yolo group's flags (for example `recipe=fused-v1`); an old row without
  it is stale; a stale yolo group must **re-export**, not only rebuild the plan (check the flow).
- Red: a stamp row from 0.6.0 for `yolov11_small_trt_end2end` reads stale; red today. Control: the
  mobileclip, faces, ocr and pe rows are unaffected.

### Track `decoders` (Stage B)

**B3-D1 Fused-only adapter.** Size S.
- Files: `src/clients/model_adapters.py` (delete `End2EndNMSAdapter`, `END2END_OUTPUTS`;
  `resolve_adapter` accepts only `[..., 300, 6]`, fail closed with the re-export command),
  `src/services/config_store/ingest_profile_validation.py` (imports `END2END_OUTPUTS` for the
  primary model check), `src/clients/triton_client.py`, `src/utils/affine.py::format_detections`
  docstrings.
- Red: a 4-tensor metadata stub now raises `unsupported_detector_contract`; control: fused metadata resolves.

**B3-D2 Ingest primary takes the fused contract (U-C9).** Size M.
- File: `src/services/curation/ingest_detect.py` (imports `END2END_OUTPUTS`); the secondary
  (`resolve_with_secondary`, original-YOLOv5 objectness) is unchanged (D-11).
- Red: a promoted YOLO26 engine (fused) as primary is accepted; red today. Control: secondary
  path tests unchanged.

**B3-D3 Region detector raw-head path removed (U-Step 6).** Size M.
- Files: `src/services/detection/cascade_detect/region_detector.py` (`[B,5,N]` decode in
  `_decode_yolo_output`, the `region_head_unrecognised` message),
  `src/services/detection/region_candidates.py::select_region_candidates` (drop the greedy NMS,
  keep the per-profile cap).
- Fail closed: a region profile whose model declares a raw head gets a clear error naming the
  export command. GPU-15 already decides pixel versus normalised per tensor; keep that.
- The GPU-15 leftover is part of this task (owner decision MQ-8, 2026-11-04): the region-profile
  validator (`src/services/config_store/profile_field_checks.py` or the profile schema) refuses a model
  with an undeclared region head at save time, not only at decode time (audit 4.3 GPU-15 "Left
  over"). Red: saving such a profile is a 422 naming the head; control: a declared fused head saves.
- Red: a raw-head declaration yields the fail-closed error (not an empty list); control: a fused
  `[B,N,6]` model gives today's candidates.

### Track `docs` (after the gate verdict)

**B3-W1 Docs for the stage that ships.** Size S.
- Files: `README.md` (the "YOLO11 exports through the EfficientNMS end2end toolchain" line and the
  licence paragraph), `docs/MIGRATION_TRITON_26.md` section 4 (still describes the venv re-exec)
  plus the GPU-18 entry (EfficientNMS_TRT deprecated since TensorRT 10.12, ONNX variant removed in
  10.16, replacement `INMSLayer`), `docs/image_size_audit.md` (venv layer gone),
  `export/README.md`, `docs-site/docs/operations/training-and-promote.mdx` (the note that a
  promoted engine does not fit the ingest detector contract), `docs-site/docs/configuration/advanced.mdx`
  (secondary detector: original-YOLOv5 ONNX only, the official export command), a "Supported model
  families" table in `docs/ARCHITECTURE.md` (from `model_family_versions_plan.md` section 8),
  `docs/PERFORMANCE.md` and a `docs/research/` note with the U-6.2 results, `CHANGELOG.md`
  "Breaking" (engines rebuilt once; export formats renamed; region profiles on raw-head engines
  fail closed until re-exported, with the command; TensorRT plugin deprecation recorded if Stage A).
- Gate: `.venv/bin/python scripts/docs/check_docs_vs_code.py` 0 problems.

## 6. The B3 GPU gate (U-6.2 once, for Stage A and Stage B)

Run by the coordinator (or one gate agent) after merges 1-5, on the integration head.

| Setting | Value |
|---|---|
| Compose project | `-p op-b3-gpu`, own port base, own volumes |
| GPU | 0 (`TRITON_GPU_ID=0`, `API_GPU_ID=0` in the stack's own env); GPU 2 only if agreed with peers |
| Budget | GPU window 1 of exactly three ([00a_decision_register.md](00a_decision_register.md) section 8.1): one GPU day for G0-G9, plus G10 (about 2 h), G11 (about 2 h) and G12 (about 1 h); announce start and end to peer sessions |
| Images | local-only tags (never `:latest`, Watchtower re-pulls it): `openprocessor-local/<image>:b3-ref` (main before B3), `:b3-stagea` (Stage A point), `:b3-stageb` (integration head) |
| Data | public only: COCO val2017 (5,000 images, about 1 GB, CC BY 4.0 annotations, per-image Flickr licences), the pinned set A (`coco_bench_2000`), the two Ultralytics sample images |
| Host record | `/proc/loadavg` at start and end of every run, foreign GPU processes (`nvidia-smi --query-compute-apps`), GPU sampled every 500 ms; a run with 1-minute load above 16 is noisy and repeated |
| Raw data | `artifacts_local/b3/gate/<step>/` (gitignored); public result files to `docs/benchmarks/b3/` |

Engines: **E0** = today's YOLO11 EfficientNMS engine (8.3.253 venv, from `:b3-ref`); **EA** =
Stage A engine (patch on `X`); **EB** = Stage B engine (`nms=True` on `X`).

| Step | Check | Pass |
|---|---|---|
| G0 | build the three image sets; from `:b3-ref` train 1 epoch in the **8.4.48 trainer** on a public COCO train subset (500 images, `fetch_coco_subset.py`, seed 20261009) and keep the `best.pt` | checkpoint exists (ledger: never verified before) |
| G1 | engine builds E0, EA, EB at production settings (max batch 64, ModelOpt FP16, uint8 input) with the image's TensorRT; record build time, plan size; read the TRT log for the `NonMaxSuppression` -> `INMSLayer` mapping (EB) | all build (ledger: TRT builds never verified) |
| G2 | Triton load + contract: EB metadata `output0 [-1, 300, 6]`; EA 4 outputs; `resolve_adapter` picks the right adapter; `/detect` on an Ultralytics sample image returns boxes; class column FP32 and integral (U-R4) | all hold (ledger: Triton serving never verified) |
| G3 | parity on COCO val2017 (5k, conf 0.25, IoU 0.7, class-agnostic): **EA vs E0** and **EB vs EA**: abs(delta mAP50-95) <= 0.3 and per-image detection agreement >= 99 % at IoU 0.95; EB vs `yolo val` of the `.pt` within 0.5 (`scripts/bench/detector_quality.py`) | both pairs pass |
| G4 | throughput: `perf_analyzer` at batch 1, 8, 32, 64 (3 repetitions each, median and min-max) and the set A ingest benchmark (3 repetitions): **EB >= 95 % of EA images/s at every batch, p99 within +10 %**; EA vs E0 recorded (same plugin, expect noise) | EB bar (ledger: embedded NMS at batch 64 never measured) |
| G5 | families: YOLO26, v10, v5u, v8, v9 build, load, and a 500-image COCO val subset mAP within 0.5 of `yolo val`; repeat the discrepancy check on GPU: `yolo26n` and `yolo26s` through the production export with `simplify` True and False all give `[300,6]` (or are refused, never served raw) | all pass |
| G6 | promote: (a) the G0 checkpoint from the 8.4.48 trainer loads and exports through the new trainer and served engine as `[300, 6]`; (b) 1 epoch at the new pin, promote, served `[300, 6]` | both (ledger: 8.4.48-trainer checkpoints never verified) |
| G7 | embeddings unaffected: PE, MobileCLIP and ArcFace cosine >= 0.999 median (p01 >= 0.995) against `:b3-ref` engines on set A (`scripts/bench/parity.py capture`/`compare`, profile `precision`); research embeddings plan section 11: the measured PE and MobileCLIP parity is written into the catalog (`embedding_model_versions.engine_parity_min_cos`); below 0.999 the catalog gets a **new model version** instead of an in-place engine swap, so every stored vector stays tied to the engine that wrote it | pass (export image pip set changed) |
| G8 | original YOLOv5 ONNX as the ingest secondary on a 500-image COCO subset: ingest completes, classes resolved by name (U-C10) | pass |
| G9 | image audit: export image size before/after (expect about -0.9 GB), `pip check` in each image, `importlib.metadata.version('ultralytics') == X` in API, export and trainer images, Trivy, image-vs-git audit | all pass |
| G10 | **B0 OCR before/after comparison** (moved out of B0; procedure and pass rule owned by `13_b0_close_out.md` B0-3, run unchanged). "Before" = the pre-B0 tree (`5b95591f`, API and Triton images plus its `models/ocr_pipeline/1/*.py`); "after" = the merged tree at the integration head. Needs: both image sets under local-only tags, the OCR engines built once and shared (B0 changes no engine), compose project `op-ocr-b0` (own ports), GPU 0 (GPU 2 per the host rules), the public image set of B0-3 step 4 (COCO-Text ids, EXIF 6 and 8 copies, RGBA PNGs). Compare boxes, text and `image_size` per image | B0-3's pass rule (boxes overlay the text on 10 opened images, rotated copies match `image_size`, no new errors), about 2 h. Result goes to `docs/PERFORMANCE.md`; a rejected OCR change is reverted by the OCR-only path in `13_b0_close_out.md` |
| G11 | **pins part B GPU smokes** ([01](01_package_pins.md) section 6; review fix R-17: the parts that need no GPU, `pip check` in each built image, the plain imports, the segmenter's weight download and the PE weight download script, run earlier in the W3 sweep on CPU image builds): the API image lists the CUDA execution provider; the export image imports `tensorrt` and both CuPy wheels on the GPU and runs one ModelOpt AutoCast export; the Triton image runs the GPU decode tests on CuPy 14; the segmenter starts on the GPU; MLflow client and server on one minor (checked inside G6's training run); the OpenCV 5 parity set includes `cv2.dnn.blobFromImage` in `src/utils/scrfd_decode.py` (reworked in OpenCV 5) through the face gate of G7 | every check passes; a failure names the bump and its fallback from 13 section 8 |
| G12 | **B2 stage-5 real-GPU smoke** ([03e](03e_stage5_gpu_queues.md) "GPU check"): one short training run through `gpu_train`, one export of a small model through `gpu_export`, one 20-image bake-off through `gpu_eval` | each job ends `completed`, its progress reached the SSE stream, its artifacts are under its storage prefix |

**Verdict.**
- All pass: ship Stage B.
- EB fails G3 or G4 but EA passes G1-G3 and G5-G9: **Stage A fallback.** Revert the `decoders`
  and `export` merge commits (`git revert -m 1`), keep `frame`; `docs` writes the Stage A text and
  the TensorRT plugin deprecation note; take the G4 numbers to the owner (U-J1). Re-run only
  G2 and G9 on the reverted head.
- EA fails: stop. Report to the owner; nothing merges to main.
- **Contingency (review fix R-17):** the window has a written cap of its 1.5 GPU days plus one day.
  On an EA failure the first action is the named fallback pins of 13 section 8 for the bumps the
  failing step points at (for example torch 2.14.1 on `cu130`, OpenCV 4.14.0.94, CuPy 13.6.0), as one
  commit on the integration branch, then a re-run of the failing steps only; a bisect over the part B
  commits is the second action, and only inside the cap. Past the cap, the coordinator reports to
  the owner instead of extending the window.

## 7. Test cycle (once per batch)

| Tier | What | Where |
|---|---|---|
| T0 | each task's red test and control | its lane |
| CI `families` | full U-6.1 at `X` | CI, plus once locally in the export image from the head (CPU, `--cpus 8`) |
| T3 | the GPU gate above, G0-G10 (GPU window 1) | coordinator, GPU 0 |
| T1 | `.venv/bin/pre-commit run --all-files`; offline suite `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live and not families and not db'` in the CI shards; `-m db`; `make contracts-check`; docs check; CHANGELOG rules; frontend set only if `frontend/` changed; the CPU part of G11 (image builds, `pip check`, imports, weight downloads; R-17) | coordinator, once, on the final head after `docs` |
| T2 | none (no cross-process change; the live harness is not run for B3) | |

Local CI parity before the push: every CI job's command runs green locally first.

## 8. Done when

- one Ultralytics version in every image and requirements file; the pin test passes in CI and in
  each image (G9);
- one detection contract served (Stage B) or the Stage A fallback recorded with the owner's yes;
- U-6.2 recorded in `docs/PERFORMANCE.md` and `docs/research/`;
- `X`, the freeze date, the release-note scan and the matrix output are in the PR description;
- `rg -n "venv-y11|ultralytics_patches|End2EndNMSAdapter|END2END_OUTPUTS" --glob '!docs/**' --glob '!CHANGELOG.md'`
  is empty (Stage B) or lists only the patch and `End2EndNMSAdapter` (Stage A);
- G10 (the B0 OCR comparison) recorded;
- CI green; the PR merged with a merge commit;
- the closing PR moves the keep-list below to its targets and deletes this file and its row in
  `00a_decision_register.md` and `docs/design/README.md` ([00a_decision_register.md](00a_decision_register.md) "Plan lifecycle").

### 8.1 What to keep after merge

| Fact or section | Target doc |
|---|---|
| the detection contract: one fused `[b,300,6]` output, adapters with declared frames, head-kind routing (tracks `frame`, `export`, `decoders`) | `docs/ARCHITECTURE.md` |
| the pin `X`, the freeze date, the U-6.1 matrix command, the families table (`docs` track) | `CONTRIBUTING.md`, `docs/ARCHITECTURE.md` |
| the G0-G10 numbers (mAP delta, agreement, throughput, embedding cosines, image size, OCR comparison) | `docs/PERFORMANCE.md`, `docs/research/` |
| the Breaking entries and the upgrade note (engine rebuild, 422 on old format names) | `CHANGELOG.md`, `INSTALLATION.md` |

## 9. Lane rules (every track)

- Lanes run only their targeted tests plus static checks on touched files
  (`.venv/bin/pre-commit run --files <touched files>`); no full suite and no `--all-files` in a lane.
- Commit per concern, hooks on, never `--no-verify`, never push, never merge, never touch another
  worktree.
- Tests that need the new pin, before the export image is rebuilt, run in a throwaway CPU container
  (`docker run --rm --cpus 8 python:3.13-slim ...` with `ultralytics==X` and CPU torch), the method
  the evidence runs used (`artifacts_local/model_versions/scripts/latest_env.sh`).
- No `pgrep -f`, `pkill -f`, `sleep` or `until` loops. Wait on a background job by its own
  notification or with the Monitor tool.

## 10. Docs, contracts, CHANGELOG

See B3-W1. Contracts change in B3-E5 (export formats, `normalize_boxes`). `ATTRIBUTION.md` drops
the vendored fork (B3-E6). One root `CHANGELOG.md` "Breaking" entry under `[Unreleased]` for the
v0.7.0 release.

## 11. Upgrade impact (0.6.0 -> 0.7.0; there is no intermediate release)

- `openprocessor upgrade` (and `repair`) sees the yolo group's stamp stale (B3-E7) and re-exports
  and rebuilds the default detector engine once.
- Region profiles that point at a raw-head engine fail closed until re-exported (CHANGELOG
  "Breaking" with the command).
- `POST /models/{model_name}/export` and `POST /models/upload` callers that used the
  `onnx_end2end` or `trt_end2end` format names get a 422; the fused names replace them.
- An ingest profile whose primary is a promoted YOLO26 engine becomes valid (U-C9).

## 12. Release checks handed to R-final

`07_release_acceptance.md` must include these; they are B3's contribution to the v0.7.0 acceptance.

| # | Check |
|---|---|
| RF-B3-1 | `installer_upgrade` from 0.6.0: the yolo stamp reads stale, the engine is re-exported and rebuilt, Triton metadata shows `output0 [-1, 300, 6]` (Stage B), `/detect` on an Ultralytics sample image returns boxes |
| RF-B3-2 | fresh install: the export image has no `/opt/venv-y11`; `ultralytics == X` in the API, export and trainer images; `pip check` clean |
| RF-B3-3 | train and promote phases: 1 epoch, promote, served `[300, 6]`, ingest primary accepts the promoted engine |
| RF-B3-4 | `POST /models/upload` of `yolo26n.pt` and `yolo11n.pt`: export worker builds a fused engine for each; an original YOLOv5 `.pt` is refused naming YOLOv5u |
| RF-B3-5 | a region profile on a raw-head engine fails closed with the re-export message |
| RF-B3-6 | the release embedding cosine gate (>= 0.999) against the 0.6.0 engines |
| RF-B3-7 | image-vs-git audit, Trivy, export image size recorded |
| RF-B3-8 | the original-YOLOv5 ONNX as ingest secondary, classes by name |

## 13. Rollback and fallback

- Before the PR merges: revert a track's merge commit; Stage B reverts as a group (section 6).
- After the PR merges and before R-final: revert the PR's merge commit on main. Dev stacks that
  rebuilt engines rebuild again with `./openprocessor repair` (the stamp forces it).
- There is no released state to roll back to inside v0.7.0 (no intermediate tags).

## 14. Risks

| ID | Risk | Mitigation |
|---|---|---|
| U-R1 / M13 | unrolled NMS (64 `NonMaxSuppression` nodes at batch 64) loses throughput | G4 bar; Stage A fallback; owner sees the numbers |
| U-R2 / M12 | an Ultralytics release changes export semantics between the matrix and the image build | exact pin, D-28 freeze, pin test in CI and in each image |
| U-R3 / M14 | deployed engines incompatible after the upgrade | fail closed at adapter resolution; stamp recipe token (B3-E7); CHANGELOG |
| U-R4 | FP16 class column | output FP32; G2 checks integral classes |
| U-R5 | original-YOLOv5 users lose nothing but get no fast serving | docs name the YOLOv5u retrain path |
| U-R6 | licence text drift when the fork goes | B3-E6 edits ATTRIBUTION and README in the deletion commit |
| U-R7 | #447 evidence conflict (`nms=False` row) | the head check refuses a raw head; matrix discrepancy rows |
| U-R8 | probe (one-to-many + NMS) versus served (one-to-one) head for YOLO26 | recorded in the registry as a known difference; decided in the #368 remainder (backlog) |
| B3-x1 | the trainer copies a `src/` file; drift if someone adds an import | the registry imports stdlib only; a test imports it in an isolated interpreter (`python -I`) with `src` absent |
| B3-x2 | CI downloads of release assets flake | `actions/cache`; a missing asset fails loudly, never skips in CI |

## 15. Not in B3 (placed elsewhere)

- #368 remainder (per-family training adapters, generic single-class exporter): backlog after B4,
  milestone v0.8.0 (platform breadth).
- GPU-15 leftover ("refuse such a model at profile validation instead of at decode time", audit
  4.3, "Not scheduled"): backlog, low; the decode path already fails closed.
- #84 wheel example: deferred until training is stable after B3; v0.8.0 (platform breadth).
- B1-1E work (#447 `nms=False`, head check, routing, class by name, YOLOv5 probe input): done in
  B1 1E (merged; see CHANGELOG.md); B3 moves it, does not redo it.
- Refuted or cut, do not redo (master Appendix B): an EfficientNMS CI check (GPU-18 is docs only);
  a region unit-guess removal before a box-format parameter.
- After B3: if Stage B shipped, close any request for EfficientNMS-specific work as obsolete
  (`09_github_actions.md`). N8 closes with the v0.7.0 tag.

## 16. Corrections (stale references found while verifying)

1. Former master plan 7.3 and 9.1 put B3 in v0.9.0; obsolete. B3 ships in v0.7.0 (owner decision 2).
2. Former master plan 7.3 cuts `int/b3-ultralytics` from main after B1 and runs it beside B2; owner
   decision 6 cuts it after B2 merged (early lane coding only on the coordinator's say-so).
3. `END2END_OUTPUTS` is also imported by `src/services/config_store/ingest_profile_validation.py`
   (not in the U-Step 5 file list). B3-D1 covers it.
4. U-Step 4's file list misses `export/yolo_export_cli.py`, `yolo_export_settings.py` (`HALF`),
   `yolo_model_config.py`, `yolo_triton_config.py`, `yolo_trt_build.py` (venv comment),
   `yolo_export_report.py`, and the wire field `src/schemas/model_admin.py::normalize_boxes`.
5. `src/utils/pytorch_utils.py` is also named in `src/utils/README.md` and
   `scripts/codegen/baselines/broad_except.txt`.
6. All three files are pinned `ultralytics==8.4.197` since commit `d1c6bab0` (`requirements.txt`,
   `requirements-test.txt` and `pyproject.toml`, with a test in
   `tests/test_requirements_contract.py`); the earlier statement that only `pyproject.toml` was
   pinned is obsolete. B3 moves all three together.
7. U-R3 says the stamp "must change"; the stamp is the ONNX sha256 plus build flags, and an upgrade
   keeps the old ONNX, so a recipe token is needed (B3-E7).
8. `docker/trainer/trainer.py` checks `hasattr(head_module, 'one2one')`; the matrix capability is
   `one2one_cv2`. Unverified which attributes each version has; B3-A3 unifies on the registry.
9. The trainer image cannot import `src/`; it copies single files (the `augmentation_presets.py`
   pattern). The source plan assumed a shared import.
10. The audit's GPU-14 final ("carry the box format as a Triton model parameter") is superseded by
    per-adapter frames (B3-G1); no Triton parameter is added.
11. `docs/MIGRATION_TRITON_26.md` section 4 still tells users the export re-execs into
    `/opt/venv-y11`.
12. `model_family_versions_plan.md` header still says "Branch read: `fix/411-r2-round` at
    `4103656b`"; former master plan 7 asked for that line to be corrected when committed. Not yet done.
13. `requirements-export.txt`'s `onnx-graphsurgeon` comment ("Required for End2End TensorRT NMS")
    points only at the patch; it can go in Stage B (B3-E6).

## 17. Lane agent briefs (ready to paste)

Common block for every brief (paste first):

```text
You are a Sonnet implementation agent for OpenProcessor B3. Read CLAUDE.md, docs/VISION_AND_GOALS.md
and docs/design/program/04_b3_ultralytics_one_version.md (your task IDs are there; it wins over
older plans). Work ONLY in your worktree <repo>-wt-b3-<track> on branch b3/<track>. Never push,
merge, rebase another branch, or touch another worktree or the live stack. Commit one concern per
commit (conventional commits), hooks on, never --no-verify. For each task: write the red test,
run it and SEE it fail for the stated reason, fix, see it pass, run the control. Run only your
targeted tests and `.venv/bin/pre-commit run --files <touched files>`; no full suite, no
--all-files. Use `make contracts` in the same commit as any wire change. Never use pgrep -f,
pkill -f, sleep or until loops; wait on background jobs by their notification. Docker: always
`docker compose -p <your project>`; never the default project name. GPU: none for this lane unless
stated. Public data only (COCO, Open Images, Ultralytics assets). No machine paths or private names
in tracked files. Deadline: <set by coordinator>. Final report: commits (SHA + subject), each red
output and green output, files touched, anything you could not do and why.
```

**`deps`** (worktree `<repo>-wt-b3-deps`, branch `deps/pins-part-b` from `int/b3-ultralytics`):
"Do the part B commits of `docs/design/program/01_package_pins.md` section 7, in that order, one
commit each, with the targeted check of section 6 after each (CPU resolver runs; no image build,
no GPU). Leave `requirements-export-y11.txt` alone (B3-A2 deletes it) and the Ultralytics pin to
`stagea`. Empty the pin test's part-B allowlist in the last numeric commit that needs it. Record in
your report the fallback of section 8 for every bump the coordinator's gate step G11 must watch."

**`stagea`** (worktree `<repo>-wt-b3-stagea`, branch `b3/stagea` from `int/b3-ultralytics` after `deps` merged):
"Do B3-A1..A4 with `X = <pin from D-28>`. Create `src/domain/model_families.py` with only
`head_kind_of` and `export_kwargs` (stdlib only) for B3-A3; the `families` lane extends it. CPU
tests needing the pin run in a throwaway `python:3.13-slim` container with `--cpus 8`. No GPU."

**`families`** (worktree `<repo>-wt-b3-families`, branch `b3/families` from `b3/stagea`):
"Do B3-F1..F3. Extend B1's `families` CI job and weight fixture; do not rewrite them. Include the
`yolo26n`/`yolo26s` x `simplify` discrepancy rows and the no-`end2end=` test. Mark the `yolo11n`
fused-export row `xfail(strict=True, reason='Stage B')`. No GPU."

**`frame`** (worktree `<repo>-wt-b3-frame`, branch `b3/frame` from `int/b3-ultralytics`):
"Do B3-G1 (GPU-14 final). The audit's red case is in `docs/design/codebase_audit_plan.md` GPU-14.
No GPU."

**`export`** (worktree `<repo>-wt-b3-export`, branch `b3/export` from the Stage A point SHA):
"Do B3-E1..E7 in that order. Remove the `xfail` from the `yolo11n` matrix row in B3-E1. Keep
`inspect_fused_output` and the `head_kind` routing behaviour exactly as B1 left them. No GPU; the
coordinator's gate builds engines."

**`decoders`** (worktree `<repo>-wt-b3-decoders`, branch `b3/decoders` from the Stage A point SHA):
"Do B3-D1..D3. Fail closed with a message naming the re-export command. Keep GPU-15's per-tensor
frame decision. No GPU."

**`docs`** (worktree `<repo>-wt-b3-docs`, branch `b3/docs` from the integration head after the gate
verdict): "Do B3-W1 for the stage the coordinator names (A or B). Use only numbers from the gate's
public result files. Run `scripts/docs/check_docs_vs_code.py` until 0 problems."

**Gate agent** (optional; coordinator may run it): "Run section 6 G0-G9 on GPU 0 with
`docker compose -p op-b3-gpu`, local-only image tags `:b3-ref`, `:b3-stagea`, `:b3-stageb`. Announce
start and end to peer sessions first; stop and ask if `nvidia-smi` shows a foreign process on GPU 0.
Record host load per run. Write raw data to `artifacts_local/b3/gate/`, public tables to
`docs/benchmarks/b3/`. Report each step's numbers against its bar and the verdict; never round or
carry a number forward."
