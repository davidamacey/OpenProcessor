# Near-duplicate detection: short-run prototypes nd:P0, nd:P1, nd:P2

| Field | Value |
|---|---|
| Status | REGISTERED 2026-11-05, before any score was read. Results sections below are empty until the run commits fill them |
| Plan | [09d_research_near_duplicates.md](../design/v070_rebuild/09d_research_near_duplicates.md) section 6, capped by [08_research_prototype_track.md](../design/v070_rebuild/08_research_prototype_track.md) section 2.1 (short-run protocol: tier 0 CPU smoke, tier 1 short seeded run, today's approach as control, early stop, at most 3 arms plus the control, no tier 2 without the coordinator) |
| Code | `scripts/research/near_dup/` (exact pins in `requirements.txt`, resolved set in `requirements.lock`), tests in `tests/research_scripts/` |
| Raw outputs | `artifacts_local/near_dup/` (gitignored); small backing JSON in `docs/benchmarks/near_duplicate_detection/` with `SHA256SUMS` |
| Budgets (tier 1) | nd:P0 0.2 GPU h + 2 CPU h; nd:P1 0.5 GPU h + 2 CPU h; nd:P2 0.2 GPU h + 1 CPU h. The SSCD copy-model arm of P2 is cut (no copy-model tier in v0.7.0) |

This note is append-only: a result is added under Results; the registration above the line is never edited after the first score.

## 1. Registration

### 1.1 Decision each prototype feeds

- P0: trust in every nd row (is the venv the product's space).
- P1: seed defaults of `near_dup_profile` (`image.dedup_cosine`, linkage, `max_group_size`) read before the W2.3a cut. If no cut meets every bar, today's cosine 0.98 stays.
- P2: whether the exact tier key is SHA-256 (RD-43, cost per MB) and whether a perceptual-hash tier (PDQ) earns a place (RD-46).

### 1.2 Control (today's approach)

Whole-frame embedding, unit norm, cosine edge at 0.98, connected components (single linkage), no guard:
`src/services/detection/frame_dedup.py::near_dup_groups` (PE-Core-L14-336, the images index `pe_embedding`)
and, for the core path, an OpenSearch `cosinesimil` score `(1 + cos) / 2` at 0.99, which is cosine 0.98.
The control is run through the product function itself on the same vectors (`group.py` imports it), so
the baseline rows are the product's numbers. The commit the product modules were imported from is stamped in every results JSON.

### 1.3 Spaces (arms by space)

| Arm | Space |
|---|---|
| `pe_frame` | PE-Core-L14-336 whole frame, product preprocessing (`src/utils/pe_preprocess.py`), the images-index space |
| `mobileclip_frame` | MobileCLIP2-S2 whole frame, 256 px, the core-path space |

Not run in this window: `PE crop` (crop level). Crop-level copies need detector boxes mapped through every
transform and belong to nd:P4 (post-tag, RG-5). P1's crop-level cut therefore stays at today's 0.98 and this is said wherever the defaults are quoted.

### 1.4 Arms by grouping rule (within a space)

| Arm | Rule |
|---|---|
| control `single` | single linkage (connected components), today's rule |
| `single_guard` | single linkage; a component larger than `G = 50` is re-split by complete linkage at the same cut (`G` is 50 because one constructed family is 11 members; the profile's 1,000 default would never trigger here) |
| `complete` | complete linkage at the cut on each connected component |

Three arms plus the control is the limit (section 2.1 of the track): the three arms above are exactly that, in each of the two spaces.

### 1.5 Sample, seed, split

- Originals: 2,000 images, 1,000 COCO val2017 plus 1,000 Open Images V7 validation (seeded, disjoint by SHA-256 from every SHA-256 listed in `scripts/datasets/manifests/*.json`). The Open Images images are the validation images already present locally (the vehicles and faces sets, 2,246 images); a fresh Open Images download was not needed. Seed `20261104` for the sample; the manifest (image ids and SHA-256) is `artifacts_local/near_dup/p1/near_dup_2000.json`, its SHA-256 is recorded in the results JSON.
- Transforms (factor-isolated, Pillow, seeded, hashed; the decode applies EXIF orientation; non-JPEG-source factors are saved as PNG): `T-q60`, `T-q30` (JPEG quality), `T-half`, `T-quarter` (Lanczos downscale, PNG), `T-crop90`, `T-crop80` (centre crop keeping that share of each side, PNG), `T-bright` (brightness x1.15, PNG), `T-rot` (3 degrees, border cropped, PNG), compounds `C-social` (half then JPEG q75) and the four `make_variant` outputs of `scripts/bench/retrieval_eval_data.py` (JPEG q60, half size, 90 % crop, brightness x1.15, each also saved as JPEG q90), reported as four separate transforms `CB-q60`, `CB-half`, `CB-crop90`, `CB-bright`, for continuity with `dedupe_r1.json`.
- Pool for a half: its originals plus every transform output of those originals (about 11 to 14 items per original). A copy is recovered when it lands in the same group as its original.
- Split: 50/50 by original, three split seeds (`1`, `2`, `3`). The cut is chosen on the calibration half; the test half is read **once**, at the chosen cut and at the control cut 0.98. Calibration curves (all cuts) are published for the calibration half only.
- Copydays (INRIA; the full local set: 157 originals and the strong-edit copies, plus the JPEG and crop sets when the INRIA mirror serves them): never used to choose a cut; reported as recall of the original-to-copy pairs and the false-group rate at the cut chosen on the constructed set and at 0.98.
- Hard negatives: all distinct-original pairs of the pool above the lowest swept cut are counted. COCO and Open Images contain their own near-duplicates (claim C7 of the plan), so the false-group rate is reported twice: raw, and after a visual check of the largest (at most 20) distinct-original pairs above 0.98 and above the chosen cut (sheets written to `artifacts_local/near_dup/p1/sheets/`, viewed by the coding agent; this is an agent read, not a blind human or hosted-judge read, and is labelled that way). The registered bar is on the raw rate with the checked-true duplicates listed.
- Cost cap: a run that would pass its row's GPU budget stops at the budget; an arm stops early when its 95 % Wilson interval lies below the control or cannot reach the margin with the remaining sample.

### 1.6 Grid

Cosine 0.900 to 0.995 step 0.005 (20 cuts). Candidate edges: exact, all pairs with cosine >= 0.85 (blocked matmul, FAISS `IndexFlatIP` range search on CPU is the reference for a cross-check).

### 1.7 Metrics

Per transform: copy recall (copy grouped with its original) with a Wilson 95 % interval. Per pool: false-group rate (share of groups containing more than one distinct original), BCubed purity and completeness, group-size distribution and largest group, share of components split by the guard, edge counts. Spread over the three split seeds is reported next to each tier-1 number.

### 1.8 Bars (per space and grouping rule, on the test half at the chosen cut)

Carried from #416 and 09d section 6.5:

1. false-group rate <= 0.5 % (raw);
2. copy recall >= 0.90 on `T-q60` and on `T-half`;
3. copy recall >= 0.80 on `T-crop90`;
4. no transform's recall below the control's recall at cosine 0.98 on the same half.

Choice rule when several cuts pass: the highest cosine that passes; single linkage over complete or guarded at equal bars; between the two spaces the chosen space is the one the plan names (the images-index space, `pe_frame`, for `image.dedup_cosine`); `mobileclip_frame` yields the core-path cut. A win at tier 1 means: the arm passes every bar at a cut the control (single, 0.98) fails at, or passes at a higher cut than 0.98 with strictly higher recall; the default cut changes only after the coordinator's tier-2 confirmation. Stop rule from the plan: no cut meets every bar, keep 0.98 and publish the curve.

### 1.9 P0 (harness and parity)

- Builders and unit tests, red then green, in `tests/research_scripts/`.
- Tier 0: the whole harness on 50 originals on CPU (no GPU), before any GPU use.
- Encoder parity (the plan's gate: 200 images encoded by the venv and by the product, median cosine >= 0.999) needs product vectors, i.e. a running product stack. This lane may not start a compose stack or build images and no stack is running, so the product-vector comparison is **not run** and is recorded as such. Substitutes, labelled as substitutes: (a) the product preprocessing function itself is imported, not re-implemented; (b) the venv's fp16 and fp32 encodings of the same 200 images (median cosine reported, on GPU 0, within the 0.2 GPU h parity budget); (c) the already-committed product-engine measurement `docs/benchmarks/v060_fp16_pe_engine.json` (FP16 TensorRT engine against the FP32 plan, 200 images: mean cosine 0.99967, minimum 0.99762) is cited as the engine-versus-reference noise level. The stop rule "median below 0.999 means the venv vectors are not used" is read as: it triggers on a measured value; with no measurement the P1 conclusions are stated as research-space conclusions, and the coordinator decides about confirming them against product vectors in tier 2.

### 1.10 P2 (tiers)

- (a) `imohash` against SHA-256 on every file of every local set used here (COCO val2017, the Open Images images, Copydays, every constructed output): count of pairs with equal `imohash` and different SHA-256 (collisions), by file size; SHA-256 cost in ms per file and MB/s, on files read after a warm-up, 3 repetitions, median and range with the load average; the cost for a 5 MB high-resolution file is extrapolated from the measured MB/s and labelled as an extrapolation unless the Commons high-resolution files are present.
- (b) PDQ (`pdqhash`, 256-bit, Hamming 0 to 64), pHash and dHash (`ImageHash`, 64-bit) alone and as an extra edge over the embedding edges at the P1 chosen cut, on the P1 test halves. Stop rule of the plan: PDQ adds less than 2 percentage points of pooled copy recall at equal false-group rate over the P1 cut, or costs more than 5 % of the per-image ingest time at high resolution: no hash tier. The per-image ingest time is 1 / 35.8 s, the mid of the measured 35.0 to 36.6 img/s on set P (`docs/research/v070_measurement_log.md` E2), i.e. 27.9 ms; 5 % of it is 1.4 ms. PDQ cost is timed on a 512 px tile (the form ingest could feed it) and on the full decode of the largest local images.
- (c) SSCD arm: cut by the owner, not run.
- GPU: P2 needs no GPU except re-using the P1 vectors; its 0.2 GPU h is not planned to be spent.

### 1.11 Environment and etiquette

Venv `$SCRATCH/venvs/op-near-dup` (python 3.12, torch 2.15.0, outside the repo); GPU 0 only with `CUDA_VISIBLE_DEVICES=0` under `flock gpu0.lock`; `nice -n 19`, at most 8 CPU threads; host load recorded next to every row; a 1-minute load above 35 holds heavy runs.

### 1.12 Dataset terms as read at download

- COCO 2017: annotations CC BY 4.0, images under their Flickr licences (as stated at cocodataset.org); local copy only.
- Open Images V7: annotations CC BY 4.0; the local images are listed CC BY 2.0 in their attribution CSVs; local copy only.
- INRIA Copydays (`https://thoth.inrialpes.fr/~jegou/data.php.html`, read 2026-11-05): "INRIA is the copyright holder of all the images included in the dataset"; free download for evaluation with a citation to Jegou, Douze, Schmid (ECCV 2008). Local copy only, never committed or redistributed; only aggregates are published. The original and strong sets came from the mirror `dl.fbaipublicfiles.com/vissl/datasets/` because the INRIA host did not answer; the JPEG and crop sets were not served by that mirror.

---

## 2. Results

(filled by the run commits)
