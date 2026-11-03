# Model families and Ultralytics versions: one toolchain, tested

Status: **plan** (no code changed). Branch read: `fix/411-r2-round` at `4103656b`
(`<repo>-wt-r2round`). Related issues: #447 (YOLO26 raw head), #368
(training model-family registry). Written 2026-11-03.

Owner requirements this plan answers:

1. "We need to ensure that we can develop and use the right model with the right ultralytics code."
2. "If we can use the latest ultralytics and have a single version that is good, but it needs to be
   tested."
3. "Our older model for the yolov5 must work as well." Clarified: Ultralytics-family models from
   YOLOv5 up to YOLO26 should be processable by the backend; for YOLOv5 the minimum is accepting a
   YOLOv5 ONNX or providing an exporter; if it is too complicated it may be left out, with the cost
   stated.

A fresh agent with no memory must be able to execute this file. Every claim below cites the run
that produced it (section 3) or an official source (section 10).

## 1. TL;DR and recommendation

**Recommendation: one Ultralytics version, pinned exactly, in every image (API, export, trainer),
and one detection contract: a single fused `output0 [batch, 300, 6]` tensor
(`x1, y1, x2, y2, score, class` in letterboxed input pixels) produced by Ultralytics' own export
options.** Delete `/opt/venv-y11`, `requirements-export-y11.txt` and the vendored
`src/ultralytics_patches/` once the GPU gate in section 6.2 passes.

Why, from the runs in section 3:

- Ultralytics' documented `nms` export argument already gives every supported family the fused
  head, on every version tested (8.4.48, 8.4.174, 8.4.197, 8.4.201; 8.3.253 for the families it
  knows):
  - `nms=True` (embedded NMS, ONNX `NonMaxSuppression`) for YOLOv5u, v8, v9, 11: output
    `[batch, 300, 6]`, and the detections equal a torchvision-NMS oracle on the raw head
    exactly (matched fraction 1.0, max score diff 0.0) on `bus.jpg` and `zidane.jpg`.
  - `nms=False` (NMS-free one-to-one head) for YOLO26 and YOLOv10: output `[batch, 300, 6]`.
- The vendored patch is not needed for any family: what it adds (TensorRT `EfficientNMS_TRT`
  baked into the ONNX) is obsolete. NVIDIA deprecated `EfficientNMS_TRT` in TensorRT 10.12 ("will
  be removed in a future release"); the image ships TensorRT 11.1.0.106. The replacement NVIDIA
  names is `INMSLayer`, which is what the TensorRT ONNX parser builds from the ONNX
  `NonMaxSuppression` op that Ultralytics `nms=True` emits (onnx-tensorrt operator table:
  NonMaxSuppression supported, FP32/FP16).
- The API already has the decoder for the fused contract (`FusedDetAdapter` in
  `src/clients/model_adapters.py`); YOLO26 is served through it today.
- Ultralytics changes export semantics inside the 8.4 line (see section 4, C1 and C2). A range
  pin (`>=8.4.174,<8.5`) therefore does not give "the right code"; an exact pin plus the CI
  matrix in section 6 does.

**YOLOv5:** two different things carry the name.

- **YOLOv5u** (Ultralytics' anchor-free re-release, `yolov5nu.pt`) is a normal Ultralytics model:
  loads, safe-loads and exports exactly like v8/11 on every version tested. Fully supported by
  the recommendation above.
- **Original YOLOv5** (`ultralytics/yolov5` repo checkpoints): the Ultralytics package refuses
  them on every version (official docs: "Models trained with the original YOLOv5 repo are not
  compatible with the Ultralytics library"; runs: `TypeError ... NOT forwards compatible`). The
  only correct path is an ONNX exported by the official `ultralytics/yolov5` repo's own
  `export.py` (output `[batch, anchors, 85]`, objectness channel, names in metadata). The repo's
  existing code reads that ONNX for the **probe** (`yolov5_objectness`) and for the **ingest
  secondary detector** (raw output plus `ensemble_nms`), but not for `/detect`, not as the ingest
  primary, not through `POST /models/upload`, and no exporter or TRT build path exists.
  Recommendation J3: keep it as "bring your own ONNX" for the probe and the secondary detector
  only, documented with the official export command, and do not build a serving path for it.

## 2. What exists today (verified)

| Area | File | What it does | Ultralytics |
|---|---|---|---|
| API runtime | `requirements.txt`, `requirements-test.txt` | `ultralytics>=8.4.174,<8.5`; image resolves 8.4.197 | 8.4.197 in `openprocessor:0.7.0-rc5` |
| Upload validation | `src/services/model_export.py::validate_pytorch_model` | `torch_safe_load(safe_only=True)`, refuses when `_SafeLoad.SUPPORTED` is false; routes on `model.end2end` | 8.4.197 |
| YOLO26 export | `export/export_yolo26.py` | `model.export(format='onnx', dynamic=True, batch=max_batch, device='cpu', simplify=True)`, no `nms` argument; then ModelOpt FP16 (`ultralytics.utils.export.engine.modelopt_quantize_onnx`), uint8 input prefix, TRT build | main env 8.4.197 |
| YOLO11 export | `export/export_models.py`, `export/yolo_formats.py` | re-execs into `/opt/venv-y11`; `Exporter` + patched `export_onnx_trt` with `topk_all=300, iou_thres=0.7, conf_thres=0.25, class_agnostic=True, normalize_boxes=True, opset=17` | 8.3.253 (`requirements-export-y11.txt`) |
| Vendored patch | `src/ultralytics_patches/` | levipereira fork extract: `TRT::EfficientNMS_TRT` ops in ONNX, 4 outputs | targets 8.3 |
| Export worker | `export/export_worker.py::build_command` | `job['end2end']` true -> `export_yolo26.py`, else -> `export_models.py` | both |
| Trainer | `docker/trainer/Dockerfile` (`ULTRALYTICS_SHA=8f5d355c...`), `docker/trainer/trainer_fit.py::_export_onnx` | git SHA pin; promote ONNX via `export(format='onnx', dynamic=True, batch=export_batch, simplify=True, half=False)`, no `nms` | **8.4.48** in `openprocessor-trainer:0.7.0-rc5` |
| Promote config | `src/services/training/yolo_triton_config.py` | renders `output0 [300, 6]`, comment says "`end2end=True` is the default" | n/a |
| Decoders | `src/clients/model_adapters.py` | `End2EndNMSAdapter` (4 tensors) and `FusedDetAdapter` (`[..., 6]`), picked from Triton metadata | n/a |
| Box frame | `src/utils/affine.py::inverse_letterbox_coords` | guesses "normalized vs pixel" from `boxes.max() > 1.0` | n/a |
| Region detector | `src/services/detection/cascade_detect/region_detector.py` | decodes `[B,5,N]` anchors head itself and `[B,N,6]`; NMS in `src/services/detection/region_candidates.py` | n/a |
| Ingest | `src/services/curation/ingest_detect.py` | primary must be the 4-tensor end2end contract (`END2END_OUTPUTS`); secondary must be raw `output0 [N, 5+nc]` (original-YOLOv5 objectness) decoded by `src/services/detection/ensemble_nms.py` (torchvision NMS) | n/a |
| Probe | `src/services/curation/probe_models.py` | `yolo11`/`yolo26` via `ultralytics.YOLO`; `yolov5_objectness` via onnxruntime, 1280 letterbox, names from ONNX metadata | API env |
| Triton configs | `models/yolov11_small_trt_end2end/config.pbtxt` (4 outputs, uint8 in, FP16 scores), `models/yolo26_small_trt/config.pbtxt` (`output0 [300,6]`) | | |
| Dead module | `src/utils/pytorch_utils.py` | imports `ultralytics.YOLO`; no importer in `src`, `scripts`, `tests` (only a ratchet baseline line) | |

Images inspected: `openprocessor-local/openprocessor-export:0.7.0-rc5` (main env ultralytics
8.4.197, torch 2.15.0+cu132, onnx 1.21.0, onnxruntime 1.24.4, TensorRT 11.1.0.106; `/opt/venv-y11`
ultralytics 8.3.253, torch 2.15.0+cpu; the venv-y11 layer is 921 MB),
`openprocessor-local/openprocessor:0.7.0-rc5` (8.4.197), `openprocessor-local/openprocessor-trainer:0.7.0-rc5`
(8.4.48, `ULTRALYTICS_SHA=8f5d355cd05b91503e7bf62429681dea4fa4b004`, Python 3.12).

Latest release on PyPI on 2026-11-03: **8.4.201** (no 8.5 yet). Ultralytics ships several 8.4
patch releases per day (8.4.197 to 8.4.201 between 2026-11-01 and 2026-11-03).

## 3. Compatibility matrix (runs on 2026-11-03, CPU only)

### 3.1 How it was run

All runs were one-at-a-time `docker run --rm` jobs, CPU only (`--cpus 8`, no GPU, no compose).
Scratch area `artifacts_local/model_versions` (not in the repo): `weights/` public weights, `scripts/` the probes,
`out/` exported ONNX, `run_*.log` the raw logs (one `RESULT {json}` line per case).

Weights (public, sha256 prefix): `yolo11n.pt` 0ebbc80d, `yolo26n.pt` 9b09cc8b, `yolov5nu.pt`
9e9c1be4, `yolov8n.pt` f59b3d83, `yolov9t.pt` 410e3cac, `yolov10n.pt` 11287ed0 (all from
`github.com/ultralytics/assets/releases/download/v8.4.0/`), original `yolov5n.pt` 4f180cf2 (from
`github.com/ultralytics/yolov5/releases/download/v7.0/`).

Environments and commands:

| Tag | Ultralytics | torch | Where | Command |
|---|---|---|---|---|
| u8448trainer | 8.4.48 | 2.15.0+cu132 | `openprocessor-trainer:0.7.0-rc5` | `docker run --rm --network none --cpus 8 -w /tmp -v artifacts_local/model_versions/weights:/w:ro -v artifacts_local/model_versions/out:/out -v artifacts_local/model_versions/scripts:/s:ro --entrypoint python openprocessor-local/openprocessor-trainer:0.7.0-rc5 /s/probe.py --tag u8448trainer` |
| u84174 | 8.4.174 | 2.15.0+cpu | throwaway venv in `python:3.13-slim` | `docker run --rm --cpus 8 -v ...:/w:ro -v ...:/out -v ...:/s:ro python:3.13-slim bash /s/latest_env.sh 'ultralytics==8.4.174' u84174` |
| u84197 | 8.4.197 | 2.15.0+cu132 | `openprocessor-export:0.7.0-rc5` main env (= API image version) | same as u8448trainer with the export image and `--tag u84197` |
| u84201 | 8.4.201 (latest) | 2.15.0+cpu | throwaway venv in `python:3.13-slim` | `bash /s/latest_env.sh 'ultralytics==8.4.201' u84201` |
| u83253 | 8.3.253 | 2.15.0+cpu | `openprocessor-export:0.7.0-rc5` `/opt/venv-y11` | `--entrypoint bash ... /s/export_image_run.sh` (runs `/opt/venv-y11/bin/python /s/probe.py --tag u83253`) |

`probe.py` per environment: (1) `ultralytics.YOLO(pt)` load and head inspection; (2)
`ultralytics.nn.tasks.torch_safe_load(pt, safe_only=True)`; (3) ONNX exports with the repo's
settings (`format='onnx', imgsz=640, dynamic=True, simplify=True, device='cpu'`, plus `batch=8`
where a max batch matters), reading the graph outputs and NMS ops with `onnx`, then running the
ONNX in onnxruntime at batch 2 on Ultralytics' `bus.jpg` and `zidane.jpg` (letterbox 640, pad 114);
(4) for `nms=True` exports an oracle: torchvision NMS (class-agnostic, conf 0.25, IoU 0.7) over the
raw-head export of the same model, matched at IoU >= 0.95 with the same class.
`patched.py` runs the repo's exact YOLO11 end2end call (`export/yolo_formats.py::export_onnx_end2end`
arguments) with `src/ultralytics_patches` from the worktree.

### 3.2 Load and restricted load

| `.pt` | 8.3.253 | 8.4.48 | 8.4.174 | 8.4.197 | 8.4.201 |
|---|---|---|---|---|---|
| original YOLOv5 (`yolov5n.pt` v7.0) | refused (`TypeError ... NOT forwards compatible`) | refused | refused | refused | refused |
| YOLOv5u / v8 / v9 / 11 | load ok (`Detect`) | ok | ok | ok | ok |
| YOLOv10 | ok (`v10Detect`, one2one) | ok | ok | ok | ok |
| YOLO26 | **unpickles** (`Detect` with one2one) but mis-exports (3.3) | ok, `end2end=True` | ok, `end2end=False` | ok, `end2end=False` | ok, `end2end=False` |
| `torch_safe_load(safe_only=True)` | returns `SafeClass` stubs for every file, no `_SafeLoad` | same as 8.3.253 (stubs, even for original YOLOv5) | same as 8.4.197 (real model; original YOLOv5 refused) | real `DetectionModel`; original YOLOv5 refused ("references types outside the supported Ultralytics checkpoint format"); `_SafeLoad.SUPPORTED=True` | same as 8.4.197 |

### 3.3 ONNX export: output head

Cells: output shape (NMS ops in graph); "oracle 1.0" = detections identical to the raw-head +
torchvision-NMS oracle on both images (matched fraction 1.0, max score diff 0.0).

| Family | Arguments | 8.3.253 | 8.4.48 | 8.4.174 | 8.4.197 | 8.4.201 |
|---|---|---|---|---|---|---|
| YOLO11 | default (repo "standard") | `[b,84,anchors]` | same | same | same | same |
| YOLO11 | `nms=True, batch=8, agnostic_nms=True, conf=.25, iou=.7, max_det=300` | `[b,300,6]` (8 NMS), oracle 1.0 | same, oracle 1.0 | same, oracle 1.0 | same, oracle 1.0 | same, oracle 1.0 |
| YOLO11 | `nms=True` class-aware | `[b,300,6]`, oracle 1.0 | same | same | same | same |
| YOLO11 | vendored patch (`export_onnx_trt`, repo args) | 4 outputs (`num_dets`, `det_boxes [b,300,4]`, `det_scores`, `det_classes`), 1 `EfficientNMS_TRT`, 443 nodes | not run | same 4 outputs, 1 `EfficientNMS_TRT` | **same 4 outputs**, 1 `EfficientNMS_TRT`, 442 nodes (patch works on 8.4.197 in the export image's main env) | 4 outputs `num_dets [b,1]`, `det_boxes [b,300,4]`, `det_scores [b,300]`, `det_classes [b,300]`, 1 `EfficientNMS_TRT` (onnxsim and graphsurgeon absent in that venv, steps skipped) |
| YOLO26 | repo `export_yolo26.py` args (no `nms`) | `[b,84,anchors]` | **`[b,300,6]`** (TopK) | **`[b,84,anchors]`** | **`[b,84,anchors]`** (#447) | **`[b,84,anchors]`** |
| YOLO26 | `nms=False` | `[b,84,anchors]` | `[b,300,6]` | `[b,300,6]` | `[b,300,6]` (TopK, meta end2end=True) | `[b,300,6]` |
| YOLO26 | `end2end=True` | rejected ("not a valid YOLO argument") | `[b,300,6]` | `[b,300,6]` | `[b,300,6]`, warns deprecated (use `nms=False`) | same |
| YOLO26 | `nms=True` | `[b,300,6]` but 3/0 dets (wrong) | `[b,300,6]` via one2one (nms ignored) | `[b,300,6]` (8 NMS), 5/3 dets | `[b,300,6]` (8 NMS), 5/3 dets | same |
| YOLOv10 | default | `[b,300,6]` (TopK) | `[b,300,6]` | `[b,84,anchors]` | `[b,84,anchors]` | `[b,84,anchors]` |
| YOLOv10 | `nms=False` | `[b,300,6]` | `[b,300,6]` | `[b,300,6]` | `[b,300,6]` | `[b,300,6]` |
| YOLOv5u | default / `nms=True` | raw / `[b,300,6]` oracle 1.0 | same | same | same | same |
| YOLOv8 | default / `nms=True` | raw / `[b,300,6]` oracle 1.0 | same | same | same | same |
| YOLOv9 | default / `nms=True` | raw / `[b,300,6]` oracle 1.0 | same | same | same | same |

Box frame: every `[b,300,6]` output is `x1,y1,x2,y2` in letterboxed input pixels (max
coordinate 572-640 in the runs), score, class as float. NMS-free heads (YOLO26/v10 `nms=False`)
always return 300 rows with real (small) scores; embedded-NMS outputs zero-pad.

### 3.4 Over-batch and dynamic shape (`overbatch.py`, 8.4.197 ONNX in onnxruntime)

| ONNX (exported `batch=8`) | batch 1 | batch 8 | batch 10 | 1x3x480x640 input |
|---|---|---|---|---|
| YOLO11 `nms=True` | dets [5] | [5,3,5,3,5,3,5,3] | **[5,3,5,3,5,3,5,3,0,0]**: images 9-10 silently empty | runs, `[1,300,6]` |
| YOLO26 `nms=False` | [300] | 8 x 300 | 10 x 300 (no unrolling) | runs, `[1,300,6]` |

This confirms C7: an embedded-NMS export must never see a batch larger than its export `batch`.

### 3.5 Original YOLOv5 ONNX (official `ultralytics/yolov5` master `export.py`)

Exported in a throwaway `python:3.13-slim` venv with `pip install -r requirements.txt` of
`ultralytics/yolov5` master (which itself pins `ultralytics>=8.4.192`; it resolved 8.4.201), from
the original `yolov5n.pt` v7.0:
`python export.py --weights yolov5n.pt --include onnx --imgsz {640,1280} --device cpu --simplify`
and `--imgsz 640 --dynamic` (script `artifacts_local/model_versions/scripts/y5_env.sh`). Then the repo's own code ran on
them in the API image (`openprocessor:0.7.0-rc5`, script `artifacts_local/model_versions/scripts/v5probe.py`):
`_build_predictor(path, 'yolov5_objectness')` on a crop of `bus.jpg`, and
`apply_ensemble_nms(raw, conf_thres=0.25)` on the raw output.

| ONNX | Output | Metadata | Probe (`yolov5_objectness`) | `apply_ensemble_nms` |
|---|---|---|---|---|
| 1280 static (9.2 MB) | `output0 [1,100800,85]` | `names`, `stride` | ok: `person`, conf 0.986, 80 names | 4 detections |
| 640 static (7.9 MB) | `output0 [1,25200,85]` | `names`, `stride` | **fails**: onnxruntime "Got 1280 Expected 640" (the probe hard-codes `YOLOV5_OBJ_INPUT_SIZE = 1280`) | 5 detections |
| 640 `--dynamic` (7.6 MB) | `output0 [batch, anchors, 85]` | `names`, `stride` | ok at 1280 | 5 detections |

So the existing original-YOLOv5 paths work on real official exports, with one defect: the probe
must read the input size from the ONNX instead of assuming 1280 (fix in step 3 of section 7).
Loading these ONNX files through `ultralytics.YOLO` was not tested; Ultralytics documents
foreign-ONNX inference only for YOLOv7, and only after an ONNX rewrite to `[1, 100, 6]`
(docs.ultralytics.com/models/yolov7). The official repo offers in-graph NMS only for TensorFlow formats (`--nms`: "TF: add NMS to
model"), so there is no first-party way to give them the fused `[300,6]` contract.


## 4. Corrections to current assumptions

C1. **"YOLO26 uses the main environment and `export_yolo26.py` gets the fused head" is false on
8.4.197.** Root cause of #447: from some 8.4 release on (8.4.48 still exported `[300,6]` by
default, 8.4.174 / 8.4.197 / 8.4.201 do not; see matrix), Ultralytics' default `nms=None` exports
the **one-to-many raw head** for every model, YOLO26 included (8.4.197 `exporter.py`:
`model.end2end = self.args.nms is False or (...axelera...)`; docs: "By default (`nms=None`),
detection models including YOLO26 export raw one-to-many predictions"; "`nms=False` for
NMS-free detections shaped `(batch, max_det, 6)`"). The fix is the documented argument
`nms=False`, which gives `[batch, 300, 6]` on every version tested, including for `yolo26n.pt`.
The #447 table's row "`yolo26n.pt` + `nms=False` -> raw head" did not reproduce: in this run
`nms=False` and `end2end=True` both gave `[batch, 300, 6]` with `TopK` ops and metadata
`end2end=True` on 8.4.197. Re-check how that row was produced before closing #447.

C2. **`model.end2end` is not a stable family flag.** On 8.4.48 a loaded `yolo26n.pt` reports
`end2end=True`; on 8.4.197 the same file reports `end2end=False` (8.4.197 `Detect.end2end` is a
property over `_end2end`, which old pickles do not carry). `validate_pytorch_model` stores
`'end2end': bool(getattr(model, 'end2end', False))` and `export_worker.build_command` routes on
it, so **on the shipped 8.4.197 a YOLO26 upload is routed to the YOLO11 EfficientNMS toolchain**
(8.3.253). That toolchain does **not** refuse YOLO26: 8.3.253 unpickles `yolo26n.pt`, but
`nms=False` and the default both give the raw `[batch, 84, anchors]` head, `end2end=True` is
rejected as an unknown argument, and `nms=True` produces `[batch, 300, 6]` with **different,
fewer detections** (3 and 0 boxes on `bus.jpg`/`zidane.jpg` where every 8.4 release finds 5 and
3; treat as wrong until a COCO check says otherwise).
A YOLO26 upload therefore builds a silently wrong engine today (or fails in the patch path). The capability
that is stable across versions is the presence of the one-to-one branch
(`hasattr(head, 'one2one_cv2')`: true for YOLO26 and YOLOv10, false for v5u/v8/v9/11 on every
version tested).

C3. **The trainer is not on `>=8.4.174`.** It runs 8.4.48 (git SHA). Its promote export passes no
`nms` argument, so promoted YOLO26 models get `[300,6]` today only because 8.4.48 still defaulted
to the NMS-free head; bumping the trainer SHA to any tested newer release silently turns
promoted engines into raw-head models. `src/services/training/yolo_triton_config.py`'s comment
"`end2end=True` is the default" is version-dependent and false from (at the latest) 8.4.174.

C4. **"The vendored patch requires <8.4" did not hold in this run.** On 8.4.201 (throwaway
venv, CPU) the repo's own call sequence (`get_cfg` + `Exporter` + patched `export_onnx_trt`,
copied from `export/yolo_formats.py::export_onnx_end2end`) produced the same 4-output graph as on
8.3.253, and so did 8.4.197 in the export image's main environment (with onnxsim and
graphsurgeon present; 442 vs 443 nodes): `num_dets [batch,1]`, `det_boxes [batch,300,4]`, `det_scores [batch,300]`,
`det_classes [batch,300]`, one `EfficientNMS_TRT` node (see matrix rows `patched_end2end`). So the
second environment is not even needed to keep today's engine. It is still the wrong thing to
keep: it is fork code monkey-patched onto Ultralytics internals (`Exporter`, `v10Detect`), and the
TensorRT plugin it targets is deprecated. Ultralytics' own `nms=True` gives the same detections
(oracle match 1.0) through a supported ONNX op.

C5. **`torch_safe_load(safe_only=True)` is not the same function across 8.4.** On 8.4.48 it
exists, has no `_SafeLoad`, and returns stub `SafeClass` objects for every checkpoint, including
an original YOLOv5 checkpoint (so `validate_pytorch_model` would refuse every upload with
"Not an Ultralytics model checkpoint"). On 8.4.174+ it rebuilds `DetectionModel` and refuses the
original-YOLOv5 pickle ("references types outside the supported Ultralytics checkpoint format").
The 8.4.174 floor is right for the API; the trainer must not share code that relies on it until
it is on the same version.

C6. **`half` is deprecated within 8.4.** 8.4.197 warns "'half' is deprecated and will be removed
in the future. Use 'quantize' instead." `export/yolo_formats.py`, `docker/trainer/trainer_fit.py`
and docs pass `half=`. Use `quantize=` (or leave precision to the ModelOpt/TRT step as
`export_yolo26.py` does) after the pin.

C7. **Embedded NMS is unrolled per export batch.** `NMSModel.forward` loops over images; under
`dynamic=True` the trace unrolls the loop to the export `batch` (8 `NonMaxSuppression` nodes at
`batch=8`). Ultralytics PR #26585 (v8.4.175) documents that images beyond the export batch come
back as all-zero rows from a dynamic ONNX; its fix chunks inside `AutoBackend`, not in the graph.
Triton does not use `AutoBackend`, so **the export `batch` must equal Triton `max_batch_size` and
the TRT profile max** (see matrix row `overbatch`). The 4-output EfficientNMS plugin has no such
constraint; this is the main performance question for the GPU gate.

C8. **Original YOLOv5 support is narrower than the probe code suggests.** The `yolov5_objectness`
predictor and `ensemble_nms` were tested only with fakes (`tests/curation/test_probe_models.py`
uses a fake onnxruntime). Against real official-repo exports (section 3.5) both paths work,
except the probe fails on any non-1280 static export because it hard-codes the input size.

C9. **Ingest cannot use a fused-head model as the primary.** `ingest_detect.py` requires the
4-tensor `END2END_OUTPUTS` for the primary, and the docs say so ("A promoted YOLO26 engine ... does
not yet fit either ingest detector contract"). With one fused contract the primary takes every
family, including YOLO26 and promoted models.

C10. **Secondary detector class id.** `WholeImageDetector.resolve_with_secondary` sets
`item.class_id = sec_cls` and looks up `self.registry.get(sec_cls)`: a raw model output index used
as a registry id, against the "class identity is the name" rule. Fix in the same series: map the
index through the model's own `labels.txt` / ONNX `names` to a name, then resolve the registry id
by name (the probe already does this correctly).

C11. **Installer config drift.** `scripts/lib/config_models.sh` writes the YOLO11 config with
`TYPE_FP32` input and FP32 `det_scores`, while the committed `models/yolov11_small_trt_end2end/config.pbtxt`
has `TYPE_UINT8` input and FP16 `det_scores`. It only writes when no config exists, but it is a
second hand-written source of the contract; delete it in the migration (the exporter writes the
config).

## 5. Judgment calls

**J1. One version or two environments.** Recommend **one exact version** (`ultralytics==<X>`,
where X is the newest release that passes section 6.1 on the day of the change; 8.4.201 passed
every CPU check here) in `requirements.txt`, `requirements-test.txt`, `requirements-export.txt`
(inherits) and the trainer (replace the git SHA with the SHA of tag `v<X>` so the manifest still
records a SHA). Two environments buy nothing: every family 8.3.253 can export, 8.4.x exports to
the same contract with documented arguments; 8.3.253 mis-exports YOLO26 (C2); and even the vendored
patch runs on 8.4.201 (C4). Staging, so the venv can go before the performance question is
settled:

- Stage A (low risk): one version; YOLO11 still exported through the patch, now on 8.4.x; delete
  `/opt/venv-y11` and the re-exec. Gate: 6.2 steps 1-3 for the patched engine (bit-for-bit same
  contract as today, so mAP must match the 8.3.253 engine within noise).
- Stage B (target): replace the patch with `nms=True` (J4), delete `src/ultralytics_patches/`,
  `End2EndNMSAdapter` and the 4-tensor configs. Gate: 6.2 steps 3-4 against the Stage-A engine.
- If Stage B misses the throughput gate, stay on Stage A and take the measurement to the owner;
  note the plugin deprecation (it will break on a future TensorRT) in the release notes.

Stage A and B can ship in the same release if 6.2 is run once for both engines.

**J2. Pin style.** Recommend exact `==` everywhere plus a CI job that installs the same pin (the
CI job today resolves the range freshly and can test a different release than the image ships).
Bump by a PR that reruns the section 6.1 matrix. Ultralytics changes export defaults and argument
names inside a minor (C1, C2, C5, C6), so a range is not a compatibility promise.

**J3. Original YOLOv5.** Options and cost:

| Option | Work | Result |
|---|---|---|
| a. Leave out entirely | delete `yolov5_objectness`, `ensemble_nms`, the secondary detector | lose the only path for existing original-YOLOv5 deployments |
| b. **Bring-your-own ONNX for probe + ingest secondary (recommended)** | document the official command (`python export.py --weights <w>.pt --include onnx --imgsz <n> [--dynamic]` in `ultralytics/yolov5`, which today requires `ultralytics>=8.4.192` itself); add real-ONNX tests; fix C10; validate shape/metadata at load | works as today, now tested; no serving via `/detect` |
| c. Exporter shipped in the export image | vendor or clone `ultralytics/yolov5` (AGPL) + pandas/seaborn/matplotlib (~100-150 MB) and wire a job type | one more toolchain to keep green; still no in-graph NMS (the repo only offers NMS for TF formats) |
| d. Full serving parity | add NMS to the original head by ONNX graph surgery, or a third Triton contract | hand-rolled graph surgery or a third decoder; against the owner's rules |

Recommend **b**. Users who want fast GPU serving of a YOLOv5-class model retrain from
`yolov5su.pt` (YOLOv5u), which the official YOLOv5 docs page itself recommends; that path is
fully covered by J1. If the owner prefers zero YOLOv5-original code, option a costs one PR
(delete three modules, their tests, `PROBE_ARCHITECTURES` entry, the `OP_INGEST_SECONDARY_*`
settings and docs).

**J4. Which `nms` per family.** Decide by capability, not name: if the head has the one-to-one
branch (`one2one_cv2`: YOLO26, YOLOv10) export `nms=False`; otherwise (`Detect` without it:
v5u, v8, v9, 11, 12) export `nms=True` with `agnostic_nms`, `iou`, `conf`, `max_det=300`. Read
the result back with `onnx` and refuse anything whose single output is not `[batch, 300, 6]`
(#447 scope item 2). Recommend class-agnostic NMS to keep today's YOLO11 behaviour
(`class_agnostic=True` in the patch call) ; the owner may prefer class-aware for multi-class
custom models; both are one argument.

**J5. Conf threshold baked into the engine.** Today's YOLO11 engine bakes `conf=0.25`; keep
`conf=0.25` for `nms=True` exports (requests below it return nothing extra, same as today).
NMS-free YOLO26/v10 always return 300 rows with real low scores (`n_dets` 300 in every run), so
the confidence filter in the API (`src/routers/detect.py`, the ingest confidence floor) is
load-bearing; keep `FusedDetAdapter`'s `score > 0` only as a padding filter.

**J6. Model names.** Keep the Triton name `yolov11_small_trt_end2end` (the name still says what it
is: NMS inside the engine). Its config changes to `output0 [300, 6]`; installed deployments must
rebuild the engine (the installer's engine stamps change with the export recipe). Renaming adds
churn across ~30 files for no behaviour gain.

**J7. Export batch.** Set Ultralytics `batch` = Triton `max_batch_size` = TRT profile max for
`nms=True` exports (C7). Add a Triton-side guard: the config writer reads the batch from the ONNX
metadata (`batch` key, present in every export) and refuses a mismatch.

**J8. Trainer version.** Move the trainer to the same release as J1 in the same series, and make
its promote export call the shared export settings (J4) so a promoted model has the same contract
as an uploaded one. Retraining should not be needed: the public checkpoints (published with
v8.4.0) load on every 8.4 release tested, but a checkpoint *trained* by the 8.4.48 trainer was not
available here; load and export one real promoted checkpoint in 6.2 step 6.

## 6. Required test matrix

### 6.1 CI (CPU, every PR that touches the pin, `export/`, adapters or `model_export.py`)

New marker `families` (offline suite excludes it only when no network; the CI job has network).
One parametrized test module, `tests/export/test_model_families.py`, over public tiny weights
`yolov5nu, yolov8n, yolov9t, yolov10n, yolo11n, yolo26n` (+ original `yolov5n` from the
`ultralytics/yolov5` v7.0 release) with these assertions per family, using the production
functions (not re-implemented calls):

1. `.pt` load through `ultralytics.YOLO` and through `validate_pytorch_model` (restricted load):
   ok for every Ultralytics family, refused for original YOLOv5 with a message naming YOLOv5u.
2. Head kind from capability equals the expected (`one2one` for v10/26, `nms` otherwise).
3. Production export to ONNX at `batch=8`: single output `[batch, 300, 6]`; `NonMaxSuppression`
   count == 8 for `nms` families, `TopK` present for `one2one` families.
4. onnxruntime on `bus.jpg` + `zidane.jpg` (Ultralytics assets): for `nms` families the detections
   equal a torchvision NMS oracle over the raw-head export (matched fraction 1.0 at IoU >= 0.95,
   same class, score diff < 1e-4) — the probe in this plan already implements that oracle.
5. Over-batch: run the `batch=8` ONNX at batch 10 and assert the guard (J7) is what prevents it in
   serving; the test documents that images 9-10 come back empty from the graph.
6. Original YOLOv5 ONNX (exported once by the official repo, checked into `tests/fixtures/` only if
   under ~8 MB, else fetched): `_build_yolov5_objectness_predictor` returns a class name present in
   the metadata, and `apply_ensemble_nms` returns detections on `bus.jpg` (no fakes).
7. A pin test: `importlib.metadata.version('ultralytics')` equals the pin in `requirements.txt`
   (catches CI/image drift).

Runtime on a CI runner: about 2-4 minutes for six tiny models (CPU export 3-12 s each here on a
loaded 48-core host with 8 CPUs).

### 6.2 Live acceptance (GPU, once per pin bump and before deleting `/opt/venv-y11`)

Run under its own compose project name on GPUs 0/2, public COCO val2017 only.

1. **Engine build** in the export image for each family at the production settings
   (`max_batch` 64 for the default detector, FP16 via ModelOpt, uint8 input): build must succeed
   with TensorRT 11.1; record build time and plan size. Check the TRT log for the
   `NonMaxSuppression` -> INMSLayer mapping.
2. **Triton load + contract**: Triton metadata shows `output0 [-1, 300, 6]`; `resolve_adapter`
   picks the fused adapter; `/detect` returns boxes for `bus.jpg`.
3. **Detection parity gate** (YOLO11 small, the default): COCO val2017 mAP50-95 of the new
   engine vs the current EfficientNMS engine (same weights, same conf 0.25 / IoU 0.7,
   class-agnostic): |delta| <= 0.3 points; per-image detection-set agreement >= 99 % at IoU 0.95.
   Also vs `yolo val` of the `.pt` (Ultralytics' own reference) within 0.5 points.
4. **Throughput / latency gate**: `perf_analyzer` at batch 1, 8, 32, 64 and the API ingest
   benchmark: new engine >= 95 % of the EfficientNMS engine's images/s at every batch, p99 within
   +10 %. This is the risk in C7 (unrolled NMS); if it fails, take J1's fallback to the owner.
5. **YOLO26 / YOLOv10 / v5u / v8 / v9**: build, load and a 500-image COCO subset mAP vs `yolo val`
   of the same `.pt` within 0.5 points.
6. **Promote path**: train 1 epoch on the public COCO subset in the trainer at the new pin,
   promote, and check the served output is `[300, 6]` (C3).
7. **Embeddings unaffected**: MobileCLIP / PE / ArcFace engines are not rebuilt by this change;
   the release acceptance cosine gate (>= 0.999 vs the previous engines on the COCO sample)
   still runs because the export image's pip set changes.
8. **Original YOLOv5 ONNX** as ingest secondary on a COCO subset: ingest completes, class names
   resolved by name (C10).
9. Image audit: export image size before/after (expect about -0.9 GB), `pip check` in each image,
   Trivy scan.

## 7. Migration plan (file by file)

Sequence: steps 1-3 can merge before any GPU run (they fix live bugs on the shipped 8.4.197).
Step 7 Stage A needs 6.2 steps 1-3 for the patched engine. Steps 4-6 (Stage B) need the full
6.2 gate. Step 8 goes with whichever stage ships; step 9 last.

**Step 1 — #447 fix (no version change).**
- `export/export_yolo26.py::export_fused_onnx`: pass `nms=False`; `inspect_fused_output` refuses
  any output whose per-item dims are not `[300, 6]` (read `max_det` from the export call).
- `docker/trainer/trainer_fit.py::_export_onnx`: pass `nms=False` when the head has
  `one2one_cv2`, else `nms=True` (J4); drop `half=False` in favour of the default FP32.
- `src/services/training/yolo_triton_config.py`: fix the "default" comment; refuse a promote ONNX
  whose output is not `[300, 6]` (promote_job reads the ONNX before copying).
- Tests: `tests/test_export_yolo26_head.py` (CPU, marked `export`): exports `yolo26n.pt` with the
  production function and asserts `[batch, 300, 6]` plus `TopK` present; a negative test feeds a
  raw-head ONNX to `inspect_fused_output` and expects a refusal (must fail before the fix).

**Step 2 — family detection by capability (C2).**
- `src/services/model_export.py::validate_pytorch_model`: replace `'end2end': getattr(model,
  'end2end')` with a `head_kind` of `one2one` / `nms` from `hasattr(model.model[-1],
  'one2one_cv2')`, and record `ultralytics_version` from the checkpoint (`ckpt['version']`).
- `export/export_worker.py::build_command`, `engines_to_load`: route on `head_kind`.
- Test: `tests/test_validate_pytorch_model_families.py` with real public checkpoints (yolo26n,
  yolo11n, yolov10n, yolov5nu, original yolov5n): asserts the head kind and that the original
  YOLOv5 file is refused with a message naming YOLOv5u. Fetch weights in a session fixture into
  a cache dir; skip with an explicit reason when offline (CI has network).

**Step 3 — C10 class-id fix** in `src/services/curation/ingest_detect.py::resolve_with_secondary`
(name first, then registry), with a test that fails today (registry with a gap).
Same step: `src/services/curation/probe_models.py::_build_yolov5_objectness_predictor` reads the
input size from `session.get_inputs()[0].shape` (fall back to 1280 only for a dynamic input) instead
of the hard-coded `YOLOV5_OBJ_INPUT_SIZE`, and refuses an output that is not `[.., N, 5+nc]` with
`nc == len(names)`; test with a real 640 static official export (fails today, section 3.5).

**Step 4 — one export module for all Ultralytics families.**
- New `export/ultralytics_export.py` (replaces `export/export_models.py`, `export/yolo_formats.py`
  end2end parts, `export/export_yolo26.py` export part): `export_fused_onnx(pt, max_batch)` calls
  `YOLO(pt).export(format='onnx', imgsz=640, dynamic=True, batch=max_batch, simplify=True,
  device='cpu', nms=<J4>, agnostic_nms=True, iou=0.7, conf=0.25, max_det=300)`, verifies the head,
  then the existing ModelOpt FP16 + `add_uint8_input` + TRT build from `export_yolo26.py::build_engine`
  (bounded profile, max = `max_batch`), then writes `config.pbtxt` via the existing
  `write_triton_config` (single `output0 [300, 6]`, FP32 output).
- Keep `onnx` and `trt` (raw-head) formats only if a consumer needs them; the region detector
  anchors path (`region_detector.py` `[B,5,N]`) is the only raw-head consumer — see step 6.
- `export/export_worker.py`: one command for every family; drop `EXPORT_SCRIPT_Y26` split.
- `scripts/lib/export_steps.sh`, `scripts/lib/model_groups.sh`, `scripts/lib/config_models.sh`
  (delete the heredoc config, C11), `Makefile` export targets, `openprocessor` CLI `export`:
  call the new module.
- `models/yolov11_small_trt_end2end/config.pbtxt`: becomes `output0 [300, 6]` FP32;
  `tests/test_committed_model_configs.py` renders it from the new writer.
- `src/routers/models_export.py`: formats collapse to `onnx` / `trt` (fused); remove
  `onnx_end2end` / `trt_end2end` names from the route and `make contracts`.

**Step 5 — decoders.**
- `src/clients/model_adapters.py`: delete `End2EndNMSAdapter` and `END2END_OUTPUTS`;
  `resolve_adapter` accepts only the fused `[..., 300, 6]` contract (fail closed otherwise).
- `src/utils/affine.py::inverse_letterbox_coords`: delete the `max() > 1.0` heuristic; input is
  always letterbox pixels.
- `src/services/curation/ingest_detect.py`: primary uses the fused contract (C9);
  `src/clients/triton_client.py`, `src/utils/affine.py::format_detections` docstrings updated.

**Step 6 — region detector.** `region_detector.py` keeps decoding `[B,N,6]`; drop the `[B,5,N]`
anchors decoder and the greedy NMS in `src/services/detection/region_candidates.py` once the region
models are exported with step 4 (`nms=True` gives class-agnostic NMS in-engine). Region profiles
that point at a raw-head engine must be re-exported; fail closed with a clear message otherwise.

**Step 7 — version pin and images.** Stage A (J1) is this step with the patch kept: the patch
moves into the main environment (it ran on 8.4.197 and 8.4.201), `export/export_models.py` loses
its re-exec guard, `/opt/venv-y11` goes. Stage B is steps 4-6 plus the patch deletion below.
- `requirements.txt`, `requirements-test.txt`: `ultralytics==<X>` (J1/J2).
- `Dockerfile` `builder-export`: delete the `/opt/venv-y11` RUN block and the `COPY
  ... /opt/venv-y11` in the export stage; `requirements-export-y11.txt` deleted;
  `docker/buildlib/dedupe_site_packages.sh` loses its only caller (delete if unused).
- `docker/trainer/Dockerfile`: `ULTRALYTICS_SHA` -> the SHA of tag `v<X>` (both ARGs);
  `tests/test_dockerfile_hygiene.py` keeps checking the ARG.
- Delete `src/ultralytics_patches/` (and README/ATTRIBUTION/README licence lines that describe
  the vendored fork), `export/export_models.py` re-exec guard, `YOLO11_EXPORT_PYTHON` env.
- Delete `src/utils/pytorch_utils.py` (no importers; confirm with the dead-code ratchet).
- `CHANGELOG.md` entry (breaking: engines must be rebuilt; export formats renamed).

**Step 8 — docs.** `README.md` (line ~370 "YOLO11 exports through the EfficientNMS end2end
toolchain", licence paragraph), `docs/MIGRATION_TRITON_26.md`, `docs/image_size_audit.md`,
`export/README.md`, `docs-site/docs/operations/training-and-promote.mdx` (the "does not fit
either ingest detector contract" note), `docs-site/docs/configuration/advanced.mdx`
(secondary detector: original-YOLOv5 ONNX only, J3), a "Supported model families" table (section
8) in `docs/ARCHITECTURE.md`. Run `scripts/docs/check_docs_vs_code.py`.

**Step 9 — #368 hook.** Put the family capability (head kind, export args, decoder) in one
registry module (`src/domain/model_families.py`) that the probe (`PROBE_ARCHITECTURES`), the
upload validator and the exporter read; this is the first slice of #368 and removes the
duplicated architecture tuples.

## 8. Supported families after the migration

| Family | Ultralytics `.pt` | Export (`nms`) | Served by Triton TensorRT | Served via ONNX | Notes |
|---|---|---|---|---|---|
| Original YOLOv5 (`ultralytics/yolov5`) | no (refused by every version) | n/a; official repo `export.py --include onnx` | no | probe (`yolov5_objectness`, onnxruntime CPU) and ingest secondary (Triton raw `output0`, client NMS via torchvision) | J3; retrain on YOLOv5u for full support |
| YOLOv5u | yes | `nms=True` | yes | yes | same as v8 |
| YOLOv8 | yes | `nms=True` | yes | yes | |
| YOLOv9 | yes | `nms=True` | yes | yes | |
| YOLOv10 | yes | `nms=False` (one-to-one) | yes | yes | NMS-free head; 300 rows always, filter by score |
| YOLO11 | yes | `nms=True` | yes (default detector) | yes | replaces the EfficientNMS patch |
| YOLO12 | yes (docs) | `nms=True` | expected | expected | not run here; add `yolo12n` to 6.1 if wanted |
| YOLO26 | yes | `nms=False` | yes | yes | trained family (trainer) |
| YOLOv3, YOLOv6, YOLOv7, RT-DETR, YOLO-NAS, YOLOE, YOLO-World | out of scope | | | | not requested |

"Served via ONNX" means the same `[300, 6]` ONNX is usable with onnxruntime (probe, promote via
Triton's onnxruntime backend). Segmentation, pose and OBB are out of scope (detection only).

## 9. Risks

- **R1 TensorRT performance of unrolled NMS** (C7): 64 `NonMaxSuppression` nodes at batch 64.
  Unknown until 6.2 step 4. Mitigation: the gate; fallback J1.
- **R2 Ultralytics churn**: defaults and argument names change inside 8.4 (C1, C2, C5, C6). Exact
  pin + 6.1 on every bump; read the release notes for "export", "nms", "end2end", "safe".
- **R3 Installed deployments**: the default detector's output contract changes; old engines fail
  closed at adapter resolution (good) and need `models install --only yolo` (or `repair`).
  The installer's engine stamp must change so `repair` rebuilds.
- **R4 FP16 class ids**: the fused output carries the class as a float; keep the output tensor
  FP32 (Ultralytics issue #24428 noted in `trainer_fit.py`), ModelOpt AutoCast must not cast the
  final Concat to FP16. Check in 6.2 step 2.
- **R5 Original YOLOv5 users**: J3 b keeps today's behaviour but no fast serving; communicate the
  YOLOv5u retrain path in docs.
- **R6 Licence text**: removing the vendored fork changes README/ATTRIBUTION statements; the repo
  remains AGPL (Ultralytics package dependency).
- **R8 Probe vs served head for YOLO26**: on 8.4.174+ `ultralytics.YOLO('yolo26*.pt').predict`
  runs the one-to-many head plus NMS (the loaded model reports `end2end=False`), while the served
  engine uses the one-to-one head. Probe confidences and served scores then come from different
  heads. Decide in #368 whether the probe sets the head explicitly (as `docker/trainer/trainer.py`
  does for val).
- **R7 #447 evidence conflict** (C1): the issue table's `nms=False` row did not reproduce; if a
  checkpoint really ignores `nms=False`, the step-1 head check catches it at export time.

## 10. Sources

- Ultralytics export docs (`nms` row: "`None` exports raw one-to-many predictions ... `True`
  embeds NMS where supported; `False` selects the NMS-free head when available"; `quantize`
  replaces deprecated `half`): https://docs.ultralytics.com/modes/export/
- Ultralytics end-to-end detection guide (default/True/False semantics, `(N, 300, 6)`):
  https://docs.ultralytics.com/guides/end2end-detection/
- YOLO26 page (`model.export(format="onnx", nms=False)`, `(N, 300, 6)`):
  https://docs.ultralytics.com/models/yolo26/
- YOLOv10 page (`nms=False` selects the NMS-free head): https://docs.ultralytics.com/models/yolov10/
- YOLOv5 page ("Models trained with the original YOLOv5 repo are not compatible with the
  Ultralytics library"; YOLOv5u anchor-free split head): https://docs.ultralytics.com/models/yolov5/
- Supported models list: https://docs.ultralytics.com/models/
- YOLOv7 page (Ultralytics' own pattern for foreign checkpoints: rewrite to `[N, 6]` ONNX):
  https://docs.ultralytics.com/models/yolov7/
- Ultralytics releases v8.4.157 (`nms` selects NMS vs NMS-free), v8.4.175 / PR #26585 (embedded
  NMS and larger batches): https://github.com/ultralytics/ultralytics/releases ,
  https://github.com/ultralytics/ultralytics/pull/26585
- PyPI latest 8.4.201: https://pypi.org/pypi/ultralytics/json
- NVIDIA EfficientNMS plugin ("deprecated since TensorRT 10.12 and will be removed"; INMSLayer
  replaces it): https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin
- onnx-tensorrt operator support (NonMaxSuppression: Y, FP32/FP16):
  https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md
- TensorRT INMSLayer API: https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_n_m_s_layer.html
- Installed source read: 8.4.197 and 8.3.253 `ultralytics/engine/exporter.py` (`NMSModel`, the
  `model.end2end = self.args.nms is False` line), 8.4.197 `nn/modules/head.py` (`Detect.end2end`
  property), `nn/tasks.py`.
- `ultralytics/yolov5` master `export.py` (ONNX metadata `stride`, `names`; NMS only for TF
  formats) and `requirements.txt` (`ultralytics>=8.4.192`).
