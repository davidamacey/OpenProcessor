# Utilities Module

Shared helpers used across `src/`. They import only the standard library,
third-party packages and `src/config`; the layer rules are in
[CONTRIBUTING.md](../../CONTRIBUTING.md#guards-and-baselines). Import from the
submodule (`from src.utils.image_processing import decode_image`); the
package `__init__` exports nothing. Each module's docstring is the reference
for its functions.

## Modules

| Module | What it does |
|---|---|
| `image_processing.py` | `decode_image` (cv2 fast path, PIL fallback), `decode_ocr_image` (EXIF-oriented decode for OCR) and `validate_image` (dimension bounds, 16 to 16384 pixels by default) |
| `affine.py` | Letterbox affine math: JPEG dimensions without a full decode, `inverse_letterbox_coords`, `format_detections_from_triton` |
| `cache.py` | `EmbeddingCache` (LRU with TTL), the shared image and text caches, the CLIP tokenizer |
| `cpu_pool.py` | Per-process thread pool and the in-flight memory gate for per-image CPU work (`OP_INGEST_INFLIGHT_MB`) |
| `cpu_preprocess.py` | CPU center crop and per-box crop embedding from full-resolution pixels |
| `pe_preprocess.py` | Canonical PE-Core-L14-336 image preprocessing for whole-frame embeddings |
| `file_read.py` | Read a batch's source files in parallel on a small thread pool |
| `class_names.py` | Class id to name resolution per model, class-name normalization and by-name matching, `confine_labels_path` (a labels file must sit inside the model repository) |
| `scrfd_decode.py` | SCRFD face detector pre- and post-processing (CPU numpy) |
| `face_align.py` | ArcFace alignment by similarity transform (Umeyama) |
| `face_response.py` | Consistency check on a face-recognition response |
| `uint8_input.py` | Turn a float image input of an ONNX graph into a uint8 input with the `/255` scaling inside the graph (used when promoting a model) |
| `atomic_write.py` | `atomic_write_bytes`, `atomic_write_json`: a temp file unique to each writer, then an atomic replace |
| `retry.py` | `retry_sync` and `is_retryable_error` |
| `search_timing.py` | Per-request stage timing for the curation kNN search routes (`Server-Timing` header) |
| `stage_timing.py` | Per-stage wall-time and byte metrics for the ingest hot path |
| `issue_ids.py` | One stable `id` per issue in every issue list a response serves |
| `upstream_errors.py` | How a VLM or segmenter failure is worded when it reaches a caller |
| `vectors.py` | Vector conversion for JSON documents (`float_list`) |
| `timeutil.py` | `now_iso`, the one UTC timestamp helper for stored `*_at` strings |

## Related documentation

- [../README.md](../README.md) - source layout
- [../../CLAUDE.md](../../CLAUDE.md) - repository layout and rules
