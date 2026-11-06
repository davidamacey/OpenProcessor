# Near-duplicate detection: short-run prototypes nd:P0, nd:P1, nd:P2

| Field | Value |
|---|---|
| Status | Registered 2026-11-05 before any score was read (commit `4481b0998`); tier-1 results of nd:P0, P1 and P2 recorded in sections 2 to 6 |
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

Measured 2026-11-05 (host clock 22:00 to 23:20), registration commit `4481b0998`, P1 code commit `a9f0647d7` (the P2 and report steps ran from later commits of the same branch `research/near-dup-proto`). Hardware: one RTX A6000 (GPU 0, driver 620.70.06, CUDA 13.5 runtime), 48 CPU threads shared with other sessions. Venv: python 3.12, torch 2.15.0, exact set in `scripts/research/near_dup/requirements.lock`. Host load average at the runs: 16 to 110 (peers); every run was `nice -n 19` with at most 8 threads; the timing rows quoted below ran at a 1-minute load of 22 to 29, and a first SHA-256 timing taken at load 74 was discarded and re-run. Raw outputs: `artifacts_local/near_dup/{p0,p1,p2}/` (the manifest SHA-256 is in `p1/near_dup_2000.json.sha256`); backing JSON: `docs/benchmarks/near_duplicate_detection/`.

### 2.1 nd:P0 harness and encoder parity

| Check | PE-Core-L14-336 (`pe_frame`) | MobileCLIP2-S2 (`mobileclip_frame`) |
|---|---|---|
| Venv vectors against the **product TensorRT plans** (`models/pe_image_encoder`, `models/mobileclip2_s2_image_encoder`), 200 COCO val2017 images, product preprocessing, plans run directly through the TensorRT Python API (`trt_vectors.py`), row cosine median / min | 0.99992 / 0.99955 | 0.99999 / 0.99975 |
| Venv fp16 autocast against venv fp32, same 200 images, median / min | 0.999998 / 0.99993 | 0.99999 / 0.99972 |
| Gate (plan: median >= 0.999) | passes | passes |

- Tier 0 (CPU, 8 originals and 40 copies, 48 items) ran the whole harness end to end before any GPU use; the research `single` partition equals the product `near_dup_groups` partition at cosine 0.98 and 0.95 (checked again on every P1 half: all true).
- Tests: 13 tests in `tests/research_scripts/test_near_dup_*.py`. The first run of the group tests failed twice, both my test arithmetic (complete linkage correctly keeps two pairs, not four singletons; a Wilson upper bound is 0.9999999999999999, not exactly 1.0); fixed in the test, not by relaxing the harness.
- **What this parity does and does not cover.** It is the product engine plus the product's CPU preprocessing function (`src/utils/pe_preprocess.py`, `src/utils/cpu_preprocess.py`), run without Triton. The live ingest path decodes on the GPU and resizes in the Python backend `gpu_image_prep`; that path was not run (no stack may be started in this lane), so a live-path parity is open. See finding 4.
- Sanity of the harness on 40 COCO images (not a result row): a PNG copy gives cosine 1.0000 in both spaces; re-saving as JPEG q95 gives a PE median of 0.981 and MobileCLIP 0.996; Gaussian noise of sigma 2/255 gives PE 0.988 and MobileCLIP 0.9998.

GPU time used (flock-held wall time): about 0.07 h (parity runs). CPU: about 0.5 CPU h (builders, tier 0, tests, venv install excluded).

### 2.2 nd:P1 copy-level cuts

Sample: 2,000 originals (1,000 COCO val2017, 1,000 Open Images), 13 transforms each = 26,000 copies, 28,000 items; three split seeds (1, 2, 3) share the same 2,000 originals, so their spread understates sampling variance and the Wilson intervals are the honest uncertainty. `CB-q60` writes the same bytes as `T-q60` (both JPEG q60), so the pooled recall counts that transform twice. Test-half numbers, mean over the three splits with [min-max]; each transform has 1,000 copies per half.

Chosen cut (highest cosine meeting every registered bar on the calibration half, every split): **PE single linkage 0.925; MobileCLIP single linkage 0.965** (MobileCLIP complete linkage 0.905). No cut meets the bars for PE with complete linkage. The guard arm (`G = 50`) is identical to single linkage in both spaces because no component reached 50 (largest group 28 in PE at 0.925, 14 in MobileCLIP): that arm was not exercised.

PE frame (the images-index space), single linkage, control cosine 0.98 against 0.925:

| Transform | Control 0.98 | 0.925 | 95 % Wilson (split 1) |
|---|---|---|---|
| T-q60 | 0.015 [0.014-0.017] | 0.921 [0.920-0.924] | 0.902-0.935 |
| T-q30 | 0.006 | 0.915 | 0.896-0.931 |
| T-half | 0.488 [0.472-0.507] | 0.998 | 0.993-0.999 |
| T-quarter | 0.043 | 0.931 | 0.915-0.946 |
| T-crop90 | 0.741 [0.734-0.746] | 0.997 | 0.991-0.999 |
| T-crop80 | 0.567 | 0.991 | 0.983-0.995 |
| T-bright | 0.988 | 1.000 | 0.996-1.000 |
| T-rot | 0.617 | 0.996 | 0.988-0.998 |
| C-social | 0.000 | 0.545 | 0.512-0.574 |
| CB-q60 / CB-half / CB-crop90 / CB-bright | 0.015 / 0.002 / 0.094 / 0.202 | 0.921 / 0.857 / 0.921 / 0.925 | |
| pooled copy recall | 0.291 [0.290-0.292] | 0.917 [0.915-0.919] | |
| false groups / groups | 0 / about 2,410 | 2 / about 1,110 (0.18 %, Wilson 0.05-0.66 %) | |
| largest group | 11 | 28 | |
| BCubed precision / recall | 0.998 / 0.886 | | |

MobileCLIP frame (the core-path space), single linkage, control 0.98 against 0.965 (core score `(1 + cos) / 2`: 0.99 against 0.9825):

| Transform | Control 0.98 | 0.965 | 95 % Wilson (split 1) |
|---|---|---|---|
| T-q60 | 0.814 [0.801-0.825] | 0.981 [0.978-0.983] | 0.972-0.989 |
| T-q30 | 0.778 | 0.976 | 0.963-0.983 |
| T-half | 0.677 [0.666-0.696] | 0.947 [0.942-0.951] | 0.936-0.963 |
| T-quarter | 0.198 | 0.673 | 0.636-0.695 |
| T-crop90 | 0.795 | 0.966 | 0.960-0.981 |
| T-crop80 | 0.578 | 0.868 | 0.858-0.899 |
| T-bright | 0.993 | 0.999 | 0.996-1.000 |
| T-rot | 0.808 | 0.967 | 0.961-0.981 |
| C-social | 0.584 | 0.932 | 0.917-0.948 |
| pooled copy recall | 0.723 [0.720-0.727] | 0.938 [0.936-0.940] | |
| false groups / groups | 0 / about 1,710 | 0 / about 1,110 (Wilson upper 0.34 %) | |
| largest group | 14 | 14 | |

Stability (cut chosen on COCO originals only / Open Images only, single linkage): PE 0.92 (two splits; none in one) / 0.93; MobileCLIP 0.96 to 0.965 / 0.98. The two sources differ by up to 0.015 (PE) and 0.02 (MobileCLIP): the cut does not transfer exactly between datasets (feeds J10).

Bars at the chosen cut on the test half: PE 0.925 passes bar 1 on the point estimate (0.18 %, but its Wilson upper bound 0.66 % is above 0.5 %), q60 0.92 and half 1.00 (>= 0.90), crop90 1.00 (>= 0.80), no transform below the control: PASS in all three splits. MobileCLIP 0.965 passes with room: q60 0.98, half 0.95, crop90 0.97, false-group rate 0.

Visual check (agent read, labelled as such): the two mixed groups of PE at 0.925 (test half, split 1) are two different photographs each of one scene: two frames of the same tennis match with different poses, and two frames of one room with a cat in different positions. They are same-scene siblings (burst level), not copies. MobileCLIP 0.965 and PE 0.98 produce no mixed group on that half. Because COCO and Open Images hold such siblings (claim C7), the raw false-group rate is an upper bound on the copy-level error and a lower bound on the burst-level merge risk: see the Copydays row.

Copydays (INRIA; local originals 157, strong-edit copies 229; never used to choose a cut): single linkage, recall of the strong-edit copies and the share of groups holding more than one original.

| Cut | PE recall (95 % Wilson) | PE mixed groups | MobileCLIP recall | MobileCLIP mixed groups |
|---|---|---|---|---|
| 0.90 | 0.166 (0.12-0.22) | 9 / 41 | 0.218 (0.17-0.28) | 8 / 47 |
| 0.925 | 0.096 (0.06-0.14) | 6 / 24 | 0.083 (0.05-0.13) | 5 / 22 |
| 0.965 | 0.026 (0.01-0.06) | 0 / 6 | 0.026 (0.01-0.06) | 0 / 5 |
| 0.98 | 0.013 (0.00-0.04) | 0 / 3 | 0.013 (0.00-0.04) | 0 / 3 |

Copydays strong edits (blur, print-and-scan, paint, strong colour) are not recovered at any cut that keeps groups pure; and at 0.925 a quarter of PE's groups on this set of holiday-photo series merge distinct photographs of the same trip. The chosen PE cut is therefore a **copy-and-recompression cut that sits on the burst boundary**.

Not run in P1: the crop level (PE crop: belongs to `nd:P4`), the DISC21 sample (after the tag), the Copydays JPEG and crop sets (the INRIA host did not answer and the mirror did not serve them), the three-seed repetition of independent samples (tier 2 is the coordinator's).

GPU time used: about 0.17 h (PE 422 s, MobileCLIP 126 s over 28,000 items, Copydays 55 s). CPU: about 1.3 CPU h (set build, sweeps, vectors' decode).

### 2.3 nd:P2 tiers

**(a) `imohash` against SHA-256**, every file of every local set (33,644 files: COCO val2017 5,000, Open Images 2,246, Copydays 386, the constructed outputs 26,000, Commons high-resolution 12; 31,626 distinct contents): **0 `imohash` collisions** (no pair with an equal `imohash` and a different SHA-256, at any file size). 2,018 files are byte copies of another file (about 2,000 by construction, `CB-q60` equals `T-q60`; the rest are duplicates inside the public sets). Stop rule P2a: zero collisions, so SHA-256 stays recommended on principle (C11: `imohash` samples 16 KiB blocks and its own README advises against file verification), not because a collision was seen.

**SHA-256 cost (RD-43)**, warm page cache, `hashlib` in a Python read loop, single thread, 3 repetitions over 2,000 files (COCO, Copydays, Commons), load 28.5: **319 MB/s median (range 288-319)**; per file: under 128 KiB 0.35 ms, 128 KiB to 1 MiB 0.61 ms, 1 to 4 MiB 3.4 ms, 4 MiB and over 14.3 ms (mean 6.2 MB). Extrapolation to a 5 MB high-resolution file: about 16 ms at this rate (labelled as extrapolation; the measured 6.2 MB files took 14.3 ms). Against the measured set-P ingest rate of about 36 images per second (28 ms per image, `v070_measurement_log.md` E2) that is a visible but bounded cost of the order of half the per-image time for a 5 MB file if hashed serially, so the lane should hash while the upload streams (the bytes pass through anyway) or in the worker. A thread's rate is machine-dependent; the first timing at load 74 gave 115 MB/s and was discarded.

**(b) Perceptual hashes** (PDQ 256-bit, pHash and dHash 64-bit; the 28,000-item pools of the three P1 test halves; embedding base: PE single linkage at 0.925). Hash alone, pooled copy recall with the number of groups that mix originals:

| Hash, Hamming <= t | Pooled recall (split 1) | Mixed groups / groups |
|---|---|---|
| PDQ t = 0 / 8 / 16 / 32 / 64 | 0.146 / 0.548 / 0.653 / 0.693 / 0.790 | 0 / 3,298 to 0 / 1,661 (none) |
| pHash t = 0 / 4 / 8 / 10 / 12 | 0.574 / 0.716 / 0.842 / 0.917 / 0.961 | 0 / 0 / 1 / 2 / 26 |
| dHash t = 0 / 4 / 8 / 10 / 12 | 0.502 / 0.724 / 0.862 / 0.929 / 0.970 | 0 / 1 / 4 / 13 / 47 |

Hash plus the embedding edge at 0.925 (PE base: pooled 0.9155, 2 mixed groups of 1,113): PDQ adds **+7.8 percentage points at t = 4 (0.993) and +8.3 at t = 64 (0.999)** with **no additional mixed group** (2 of about 1,000 at every t from 0 to 64), the same in all three splits (+8.3, +8.3, +8.0 at t = 64). The gain sits where PE is weak: `C-social` 0.55 to 1.00, `CB-half` 0.86 to 1.00. pHash and dHash as an extra edge add the same recall but begin to add mixed groups at t = 8 (pHash) and t = 4 (dHash), and collapse the set into one group by t = 20. Note the base is PE at 0.925; against the MobileCLIP 0.965 base (pooled 0.938) the headroom is smaller and was not measured.

**PDQ cost (the registered stop rule, per-image ingest 27.9 ms, 5 % = 1.4 ms), `pdqhash` 0.2.8, single thread, 5 repetitions, load 22**:

| Input | Decode | PDQ from the 512 px tile after a full decode | PDQ on a JPEG draft decode (decode included) | PDQ on the full array (decode included) | pHash / dHash after decode |
|---|---|---|---|---|---|
| COCO, 0.27 MP, 30 images | 3.5 ms | 12.8 ms | 12.8 ms | 11.4 ms | 2.2 / 1.6 ms |
| Commons, 15 MP median, 12 images | 191 ms | 223 ms | 113 ms | 1,112 ms | 86 / 70 ms |

PDQ costs 8 to 9 times the 1.4 ms allowance on a 640 px image, even before the decode, and 33 ms (tile) beyond a 191 ms decode at 15 MP. **Verdict P2b (RD-46): no hash tier.** The recall gain is real and free of new false groups in this set, but the registered cost rule fails by a wide margin; it would need a PDQ computed from the tile the pipeline already holds in GPU memory (a custom kernel, justified only after a re-registration), and the cheaper fix for the same recall gap is the embedding cut itself (finding 1: MobileCLIP at 0.965 reaches 0.938 without a hash). dHash alone at 1.6 ms is at the allowance but adds false groups.

P2c: SSCD arm cut by the owner (not run). GPU time used: 0. CPU: about 1.5 CPU h (hash of 28,000 files at 6 processes, timings).

## 3. Findings

1. **Today's PE cut of 0.98 recovers 29 % of constructed copies and 1.5 % of JPEG q60 re-encodes.** PE-Core embeddings move a lot under re-compression: median cosine to the original is 0.931 for q60, 0.912 for q30, 0.978 for half size, 0.985 for a 90 % crop, 0.997 for brightness (and 0.981 for a q95 re-save, 0.988 for sigma 2/255 noise, in the 40-image sanity run); MobileCLIP is much steadier (0.984, 0.970, 0.988, 0.985, 0.999). The control row is the product's own function on these vectors. The old baseline's "half size" and "crop" rows were also JPEG q90 re-encodes (C3), which is why `CB-half` and `CB-crop90` are so low (0.002 and 0.094): in PE a q90 re-save alone costs most of the recall.
2. **Registered tier-1 outcome: both spaces pass the bars at a lower cut than today's.** PE image level `dedup_cosine` 0.925 (pooled copy recall 0.29 to 0.92, 2 same-scene merges in 1,110 groups); core path (MobileCLIP) cosine 0.965 = score 0.9825 (0.72 to 0.94, no merge). Single linkage wins over complete and guarded at equal bars (the guard never triggered). Under the protocol (section 2.1 of the track) a tier-2 confirmation on a second seeded sample of the same size is required before any default changes: **none was run, so today's 0.98 stays unless the coordinator runs tier 2** (about 10 GPU minutes).
3. **The PE cut is a burst-boundary cut.** At 0.925 PE merges different photographs of the same scene (2 of about 1,110 groups on constructed data, 6 of 24 on Copydays) while 0.965 does not. A collapsing "dedup" at 0.925 would drop legitimate near-scene photos in a personal collection, and the burst level (`nd:P3`, California-ND, set P) was not measured. Recommended seed hypothesis if tier 2 confirms: core 0.965 (strong), PE 0.925 flagged as "copy and recompression", with the level split (`dedup` versus `leak` versus burst) kept as the plan has it; the Copydays and Open Images stability spread (0.01 to 0.02) argues for a per-project profile rather than one constant.
4. **Open, unexplained: the harness recovers far more at cosine 0.98 than the live baseline did.** With MobileCLIP at cosine 0.98 (score 0.99) the harness recovers 81 % of q60 copies, 68 % of half-size, 80 % of crops and 99 % of brightness; `dedupe_r1.json` (live ingest, 2026-10-29) grouped 24 %, 9 %, 21 % and 87 % with the original. The P0 parity (venv against the product plan on CPU-preprocessed inputs) is exact, so the gap is elsewhere: the live GPU-decode and resize path (`gpu_image_prep`), the embedding that route used, an approximate search, or the `(1 + cos) / 2` score mapping. The P1 numbers describe the research path; **the coordinator should have tier 2 run through the live ingest route before ND-A seeds any default**.
5. **PDQ is not worth its cost** (finding in 2.3); **SHA-256 is cheap enough and imohash showed no collision** (RD-43 stands on principle).
6. The cut does not transfer exactly between datasets (COCO 0.92 / 0.96, Open Images 0.93 / 0.98).

## 4. Recommended `near_dup_profile` seed defaults (for ND-A, subject to tier 2)

| Field | Today | P1 evidence (tier 1) | Recommendation |
|---|---|---|---|
| core path dedup cosine (MobileCLIP, score = (1 + cos) / 2) | 0.98 (score 0.99) | 0.965 (score 0.9825); pooled recall 0.938 [0.936-0.940] against 0.723; 0 mixed groups, Wilson upper 0.34 %; every split chose 0.965 | seed 0.965 only after a tier-2 pass on the live route; else keep 0.98 |
| image `dedup_cosine` (PE) | 0.98 | 0.925; pooled 0.917 [0.915-0.919] against 0.291; q60 0.921 (0.902-0.935); 0.18 % mixed groups (0.05-0.66 %), both same-scene; Copydays mixed groups 6 of 24 at this cut | seed 0.925 as "copy and recompression" only after tier 2 and a set-P burst check; else keep 0.98 |
| linkage | single | single wins; complete: PE no passing cut, MobileCLIP 0.905 passes at lower recall | single |
| `max_group_size` | 1000 | guard at 50 never triggered (largest group 28) | keep the plan default; no evidence either way |
| item (crop) `dedup_cosine`, `leak_cosine`, `k` | 0.98, 0.98, 64 | not run | keep 0.98, 0.98, 64 |

## 5. Open issues

- Tier 2 (second sample, through the live ingest route) is needed for any change; the live-versus-harness gap (finding 4) must be explained first.
- The crop level, the burst level, the leak cut and the DISC21 sample were not run (nd:P3..P7, after the tag).
- The Copydays JPEG and crop sets could not be downloaded (INRIA host unreachable); the sample is originals plus strong edits.
- The false-group visual check was an agent read of two sheets, not a blind human or hosted-judge read.
- `pdqhash` is the only PDQ implementation timed; a GPU-resident PDQ was not built (it would be custom code against the plan's rule).

## 6. Reproduce

```bash
# venv outside the repo (exact pins in scripts/research/near_dup/requirements.txt, resolved set in requirements.lock)
python3.12 -m venv $SCRATCH/venvs/op-near-dup
$SCRATCH/venvs/op-near-dup/bin/pip install torch==2.15.0 torchvision==0.29.2
$SCRATCH/venvs/op-near-dup/bin/pip install -r requirements-test.txt -r scripts/research/near_dup/requirements.txt
export PYTHONPATH=. HF_HUB_OFFLINE=1 FG_MOBILECLIP_CKPT=<MobileCLIP2-S2 checkpoint>
PY=$SCRATCH/venvs/op-near-dup/bin/python
# set: COCO val2017 + local Open Images images, seed 20261104
$PY scripts/research/near_dup/build_sets.py --coco-dir <val2017> --oi-dir <oi images> --manifest-dir scripts/datasets/manifests \
    --out-dir <copies dir> --manifest <manifest.json> --per-source 1000
$PY scripts/research/near_dup/sweep.py paths --manifest <manifest.json> --out paths.txt
# vectors, GPU 0 under the lock
CUDA_VISIBLE_DEVICES=0 flock $SCRATCH/gpu-locks/gpu0.lock $PY scripts/research/near_dup/encode.py --paths paths.txt --space pe_frame --out vec_pe_frame
$PY scripts/research/near_dup/sweep.py run --manifest <manifest.json> --vectors vec_pe_frame --space pe_frame --out p1_pe_frame.json \
    --registration-commit 4481b0998 --code-commit $(git rev-parse HEAD)
# parity against the product plan (venv with tensorrt-cu13==11.1.0.106 and torch)
$TRT_PY scripts/research/near_dup/trt_vectors.py --paths paths200.txt --space pe_frame --plan models/pe_image_encoder/1/model.plan --out product_pe.npy
$PY scripts/research/near_dup/parity.py compare --a venv_pe.npy --b product_pe.npy --out parity.json
# P2
$PY scripts/research/near_dup/tiers.py exact --root <name>=<dir> ... --out exact.json
$PY scripts/research/near_dup/tiers.py perceptual --manifest <manifest.json> --vectors vec_pe_frame --p1 p1_pe_frame.json --hash-cache hashes.npz --out perceptual.json
$PY scripts/research/near_dup/tiers.py cost --images <large images dir> --out cost.json
$PY scripts/research/near_dup/report.py --raw artifacts_local/near_dup --out docs/benchmarks/near_duplicate_detection
```
