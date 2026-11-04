# Registration-plate text as a per-box attribute: research and rebuild plan

| Field | Value |
|---|---|
| Superseded parts | the landing lanes, waves and file ownership of this plan's section 8, its GPU lock and its GPU-hour estimates are superseded by the program (review RR-16, owner answers of 2026-11-04): lanes in [03d_stage4_job_families.md](03d_stage4_job_families.md) (W2.3a and W2.3b), [03b_stage2_control_data.md](03b_stage2_control_data.md), [03f_stage6_scheduler_scaleout.md](03f_stage6_scheduler_scaleout.md), [05_b4_stateless_api.md](05_b4_stateless_api.md) and [06e_w5e_plate_text.md](06e_w5e_plate_text.md); one lock `gpu<N>.lock` per card shared by every topic ([traceability.md](reference/traceability.md) X-14); short tiered runs under a cap of about 10 GPU hours before the tag ([v070_rebuild/08_research_prototype_track.md.md](08_research_prototype_track.md) section 2.1). A fresh agent loads those files first |
| Status | plan (research first, then product). Nothing here is implemented. Written 2026-11-04 against `origin/main` at `497315954` |
| Issues | #401 (the feature; parked until now), #470 (a neutral security item about stored recognized text, owned by the W4 privacy lane), #220 (OCR on the GPU, B5c) |
| Absorbs | the former issue-401 plan (deleted in the commit that adds this file; its privacy model, sink inventory, key design, purge and test list are carried into sections 4, 5, 7 and 9 below, corrected where the owner's 2026-11-04 decisions changed them) |
| Supersedes | RQ-7, E-9 and OQ-6 of [09b_research_clustering.md](09b_research_clustering.md); section 6.8 and wave W12 of [cluster_search_makemodel_plan.md](../cluster_search_makemodel_plan.md) |
| Builds on | the control-plane design [02](02_control_plane_database.md), [02a](02a_postgres_schema.md), [02b](02b_jobs_locking_events.md); the wave order in [00_master_plan.md](00_master_plan.md); the B5c OCR work in [06c](06c_b5_crops_ocr_faces.md) |
| Owner direction | 2026-11-04: the plate text is a must for registration plates. Research fully to best practice first; prototypes are simple local-venv GPU scripts (no Docker builds) on set P (private vehicle photos with bursts) and public plate sets; plan first, run nothing. Decided the same day: external endpoints keep today's behaviour (OI-1 for #469, OI-2 for #471, [decision_register.md](reference/decision_register.md) section 6) |
| Audience | a fresh agent with no conversation memory. Symbols are cited by path and name; re-find line numbers with `rg` |

Wording rule for this file and for the code it proposes: prose says "registration plate" and
"plate text". Code says **identity text** (text that identifies one physical object; plates are
the worked example, serial numbers and race bibs are others) or **region text** (any text read
from a region box). Nothing in code names a vehicle or a plate; the naming scan
(`scripts/codegen/check_naming_leaks.py`, Scan B) enforces it.

---

## 1. Plain summary

- **What we want.** Every registration-plate box in a curation project carries its plate text,
  a confidence that means something, which reader produced it and from how many photos. Photos
  of the same vehicle taken in a burst agree on one reading. Users can search plates exactly,
  with one or two wrong characters, or by a fragment. Boxes with the same plate text can be
  viewed as one identity group, and later feed clustering as a "same vehicle" link.
- **What exists.** Most of the reading path is already there: the region stage finds plate boxes
  inside vehicle crops, the PP-OCRv5 pipeline on Triton reads them, a dominant-text selector and
  validity rules clean the reading, and an optional VLM reading can be compared with the OCR one.
  What is missing: a per-character confidence, burst consensus, normalized search fields, a reader
  benchmark on plates, privacy modes and a purge.
- **What I recommend.**
  1. Keep the PP-OCRv5 pipeline as the default reader until a prototype proves a dedicated plate
     reader is better on public data *and* its licence is clear. The strongest candidate is
     fast-plate-ocr (MIT, ONNX, about 0.5 ms per plate vendor-reported).
  2. Add burst consensus as a confidence-weighted, alignment-based vote over the frames of one
     vehicle (the published majority-vote result is +13 points on UFPR-ALPR), with a best-frame
     fallback.
  3. Store, per box, the raw text, a normalized text, a confusable-folded match key, the minimum
     per-character confidence, a tier, the reader and its version, and the frame counts. Index the
     normalized and folded forms as `keyword` fields with a custom normalizer, plus a `wildcard`
     sub-field for fragment search. Fuzzy search uses OpenSearch's `fuzzy` query.
  4. Privacy modes per project (`off`, `keyed`, `raw`) under a deployment ceiling, with a log
     redaction processor, a sink inventory test and a CLI-only purge. The default is `raw`, which
     is today's behaviour (J1); `keyed` stores keyed hashes only.
  5. Put the creation-time seams into the B2 rebuild now (index analysis settings, the secrets
     mount); land the feature as a wave of its own after B5c (W5e, J3); run the prototypes on
     GPU 0 or 2 during W2 to W4, when the program uses no GPU (J4).

## 2. What exists and what was measured

### 2.1 The reading path today

| Piece | Where | Fact |
|---|---|---|
| Plate example profile | the plate example under `examples/region_profiles/` | segmenter-only (SAM 3 text prompt) plus the OCR text-hint re-pass; `text_reader: "both"`; uppercase; charset `[A-Z0-9]`; length 2 to 10; `text_min_confidence: 0.6`; `text_reject_sequences`; US state names and codes as stopwords; parent classes by **name** (car, truck, bus, motorcycle). No detector ships: no permissively licensed plate detector was found |
| Profile text fields | `src/config/detection_profile.py` (`DetectionProfile`) | `text_reader`, `text_crop_margin`, `text_crop_min_height`, `text_min_height_ratio`, `text_border_margin`, `text_uppercase`, `text_charset`, `text_join`, `text_len_min`/`max`, `text_stopwords`, `text_min_confidence`, `text_format` (full-match regex), `text_placeholders`, `text_reject_sequences`, the text-hint fields |
| OCR pipeline | `models/ocr_pipeline/` (Python BLS), `models/paddleocr_det_trt/`, `models/paddleocr_rec_trt/` | PP-OCRv5 mobile detection (DB) and multilingual mobile recognition (SVTR-LCNet, 18,385 classes); uint8 inputs since #220 step 1; up to 128 lines per image; outputs `texts`, `text_scores`, `rec_scores`, boxes |
| Line confidence | `models/ocr_pipeline/1/ocr_postprocess.py` (`CTCLabelDecode.__call__`) | per character `preds.max(axis=2)`, then **the mean** over the decoded characters. One unreadable character hides inside a high mean |
| Dominant text | `src/services/detection/region_text.py` (`read_dominant_text`) | tallest line plus lines within a height ratio, border band dropped, stopwords stripped, normalized; reading confidence = `min` of the kept **line** means |
| Validity rules | `src/services/detection/region_text_rules.py` | placeholder, stock run, charset, length, `text_format` full match |
| Reader choice | `resolve_region_text` in `region_text.py` | writes `text`, `text_raw`, `text_source`, `text_engine_version`, `text_confidence`, `text_vlm`, `text_ocr`, `text_choice`, `text_vlm_invalid`, `text_disagreement`; VLM confidence arrives as a category mapped to 0.92 / 0.70 / 0.40 |
| Box element | `src/domain/region_boxes.py` (`RegionBox`) | fixed keys, written only through this module and the OCC helpers |
| Mapping | `src/clients/curation_opensearch/bodies_core.py` (`_region_boxes_mapping`) | every `text*` field is a plain `keyword` with no normalizer; the regions-tab filter is a case-insensitive `*substring*` wildcard on `region_boxes.text` (`src/services/curation/review_tab_clauses.py`, `region_text_clause`) |
| Item text | `src/domain/item_text.py` | every OCR line on the item crop is stored as `item_text_lines` and `item_text_tokens`; on a vehicle item that includes the plate |
| Worker stage | `scripts/curation/worker/region_text_stage.py` | region text and item text in one OCR pass; `_drop_region_text` strips text for a text-free profile (the hook where a mode transform belongs) |
| Human text | `src/services/curation/region_box_edits.py` (`human_text_fields`) | `text_source='human'`; readers never overwrite it |
| Bursts | `src/services/curation/item_scores/near_dup.py`, `src/services/detection/frame_dedup.py` | crop near-duplicate groups (`dup_group_id`, cosine threshold `OP_CROP_DUP_THRESHOLD`, default 0.98, "a hypothesis, not a finding"); no capture time is stored on images |
| Public plate sample | `scripts/datasets/fetch_openimages_plates.py`, `scripts/datasets/manifests/oi_plates_300.json` | 300 Open Images images with a "Vehicle registration plate" box, CC BY 2.0 images only, class resolved by display name. No text transcriptions |
| Purge precedent | `scripts/purge_face_data.py`, `src/config/faces.py`, `SECURITY.md` face section | CLI-only purge, counts-only dry run and audit, refuses while the feature is on, exit 1 on leftovers |

### 2.2 What was measured (and what was not)

| Measurement | Result | Source |
|---|---|---|
| OCR parity gate machinery (P-ocr) | width-bucketed recognition rejected: P-ocr 86.0 % exact, character edit rate 1.94 %, BLS 43.2 ms against 48.0 ms; exact-width grouping 100 % but 49.5 ms | [06c](06c_b5_crops_ocr_faces.md) section 2 (#220 comment 2026-11-02) |
| uint8 OCR inputs | bit-identical to the FP32 path | [faces_ocr_resident_crops_plan.md](../faces_ocr_resident_crops_plan.md) section 0 |
| A full-frame small-region detector (one earlier model, not shipped) | mAP@.5 0.968 (FP32 ONNX), 101.8 img/s on ORT-CUDA, A6000; test split 5,837 frames | [detector_quantization_results.md](../../research/detector_quantization_results.md) |
| **Plate read accuracy of any reader** | **never measured** on any data | - |
| Burst consensus gain | never measured | - |
| Confidence calibration of the region text | never measured | - |

Nothing in this plan's recommendations rests on a number measured in this repository. Every
reader and consensus number below is literature or vendor-reported and labelled so.

### 2.3 What does not exist

- A per-character minimum confidence anywhere in the pipeline.
- Normalized or folded plate fields, a normalizer in any index body, a `wildcard` field.
- Any keyed hash (`rg -n -i hmac src` is empty), a privacy mode, a purge for region text.
- A capture timestamp on images, so bursts can only be inferred from near-duplicate groups today.
- A reader benchmark, a transcribed public plate set, a dedicated plate reader.

## 3. Literature and industry review

### 3.1 Readers

| Reader | What it is | Licence | Reported numbers (vendor or paper) | Fit |
|---|---|---|---|---|
| PP-OCRv5 (in repo) | DB detection plus SVTR-LCNet recognition; PaddleOCR 3.0 | Apache-2.0 (repository) | multilingual mobile rec 81.29 % average, English 66.00 %; `en_PP-OCRv5_mobile_rec` 85.25 % English (vendor, [text recognition module docs](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)); technical report [arXiv 2507.05595](https://arxiv.org/abs/2507.05595) | general scene text, not trained for plates; only a line score is returned |
| fast-plate-ocr | Compact Convolutional Transformer, fixed character slots, ONNX Runtime inference, optional region head (country) | MIT ([repository](https://github.com/ankandrew/fast-plate-ocr)); package 1.1.0 (2026-03-14) on [PyPI](https://pypi.org/project/fast-plate-ocr/) | `cct-s-v2-global-model` 0.68 ms at batch 1 on an RTX 3090; legacy global model 93.3 % plate accuracy on its own validation set, "+65 countries" (vendor, [model zoo](https://ankandrew.github.io/fast-plate-ocr/latest/inference/model_zoo/)) | purpose-built, tiny, fast; training-data provenance not stated, so the weights' licence chain is unclear |
| Segmentation-free CNN plate reader (Zherzdev and Gruzdev 2018) | lightweight CNN with CTC, no RNN | paper; reference weights trained on synthetic Chinese plates | [arXiv 1806.10447](https://arxiv.org/abs/1806.10447) | historical baseline; needs retraining for US plates. Not a candidate |
| PARSeq (Bautista and Atienza, ECCV 2022) | permuted autoregressive scene-text model; per-character probabilities | Apache-2.0 code and weights ([repository](https://github.com/baudm/parseq)) | 91.9 % on standard scene-text benchmarks (paper) | strong general reader, fine-tunable; heavier than fast-plate-ocr |
| TrOCR (Li et al., AAAI 2023) | ViT encoder plus text decoder | MIT ([microsoft/unilm](https://github.com/microsoft/unilm/tree/master/trocr)) | [arXiv 2109.10282](https://arxiv.org/abs/2109.10282) | accuracy reference only: far too slow for per-box production use |
| The configured VLM | the existing verify call's reading | per catalog model | - | second opinion already wired (`text_reader: "both"`); confidence is a category, not a probability |

Multi-reader fusion is a measured win on plates: Laroca et al. (CIARP 2023,
[arXiv 2309.04331](https://arxiv.org/abs/2309.04331)) raise the mean recognition rate across
datasets from 92.4 % (best single model) to 97.6 % (best fusion), and recommend 4 to 6 models where
time allows. A 2026 pipeline (BLPR, [arXiv 2604.09927](https://arxiv.org/abs/2604.09927)) calls a
small open VLM only when the character reader's confidence is low: the same "agree or fall back"
shape the repo's `both` mode already has.

### 3.2 Multi-frame consensus

| Method | Source | Result reported |
|---|---|---|
| Per-position majority vote over the frames of one vehicle | Laroca et al. 2018, which introduced UFPR-ALPR (150 vehicles x 30 frames), [arXiv 1802.09567](https://arxiv.org/abs/1802.09567) | their system 64.89 % per frame, 78.33 % with temporal redundancy (47 of 60 test vehicles); commercial systems, OpenALPR among them, "below 70 %" (the abstract's wording, quoted instead of a per-system figure; review RR-19) |
| Temporal redundancy over frames instead of one chosen frame | Gonçalves et al., ITSC 2016, [paper](https://homepages.dcc.ufmg.br/~william/papers/paper_2016_ITSC.pdf) | about +15.3 points vehicle recognition rate |
| Character-wise temporal combination beyond plain voting | Quiala et al., IWAIPR 2023, [doi 10.1007/978-3-031-49552-6_34](https://doi.org/10.1007/978-3-031-49552-6_34) | beats the majority-vote baseline in every pipeline tested (Cuban plates) |
| Character time-series matching (Hungarian assignment) | [ResearchGate record](https://www.researchgate.net/publication/365121574) | 96.7 % on UFPR-ALPR (paper) |
| Recognizer output voting (ROVER): align hypotheses of different lengths into a word transition network, then vote with confidences | Fiscus, IEEE ASRU 1997, [doi 10.1109/ASRU.1997.659110](https://doi.org/10.1109/ASRU.1997.659110) | the standard alignment-then-vote method for unequal-length outputs |
| Group reads per vehicle, report the best plate with a weighted score across frames | Rekor (OpenALPR) agent, [JSON group results](https://docs.rekor.ai/scout/application-integration/json-group-results), [stream API `AlprGroupResult`](https://docs.rekor.ai/developers/vehicle-recognition-sdk/vehicle-recognition-stream-api/c++) | industry pattern: `best_plate`, `best_confidence`, the other candidates, member image ids |

Two lessons carry into the design: plain per-position voting needs equal-length readings, so an
alignment step (ROVER-style) is needed when a frame drops or adds a character; and the
industry products keep the per-frame reads and add a group result, rather than overwriting.

### 3.3 Post-processing patterns

- **Top-N candidates plus a format template** (OpenALPR open-source design,
  [wiki](https://github.com/openalpr/openalpr/wiki/OpenALPR-Design)): per-character top-N with
  confidences, a soft threshold that adds a blank option, then a regional template picks the
  matching candidate. The repo's `text_format` full-match regex is the template half; a
  format-constrained re-decode of the CTC output is the other half (candidate R6 below).
- **Detector then reader**: CodeProject.AI's plate module
  ([repository](https://github.com/codeproject/CodeProject.AI-ALPR)) is a plate detector followed
  by PaddleOCR (v3 and v4 configs in the repository), the same shape as this repo's region stage.

### 3.4 Search fields (OpenSearch 3.6.0, the pinned image)

- `keyword` indexes the whole value as one term; matching is exact and case sensitive "unless you
  use a normalizer"; `normalizer` is not dynamically updatable
  ([keyword](https://docs.opensearch.org/latest/mappings/supported-field-types/keyword/)).
- A **normalizer** is an analyzer that emits one token; it may use the `pattern_replace` and
  `mapping` character filters and the `asciifolding`, `uppercase`, `lowercase`, `trim` token
  filters, among others, and term queries are normalized the same way
  ([normalizers](https://docs.opensearch.org/latest/analyzers/normalizers/)). Analysis settings are
  part of the index settings, so they must be in the index body at creation (changing them later
  means closing the index or reindexing).
- `fuzzy` is a term-level query on Damerau-Levenshtein distance, at most 2 edits, with
  `transpositions`, `prefix_length` and `max_expansions`
  ([fuzzy](https://docs.opensearch.org/latest/query-dsl/term/fuzzy/),
  [fuzziness](https://docs.opensearch.org/latest/query-dsl/fuzziness/)).
- `wildcard` (since 2.15) indexes up-to-3-character substrings for fast `wildcard`, `prefix` and
  `regexp`; it supports a normalizer; it does not support `fuzzy`; exact `term` is slower than on
  `keyword` ([wildcard](https://docs.opensearch.org/latest/mappings/supported-field-types/wildcard/)).
  So the exact and fuzzy key is a `keyword` and the fragment search is a `wildcard` sub-field.
- A `text` field (analyzed, tokenized) is the wrong type for a plate: tokenization splits on
  separators and scoring is irrelevant for an identifier.

### 3.5 Datasets (with licences)

| Set | Content | Licence or terms | Use here |
|---|---|---|---|
| Open Images V7 "Vehicle registration plate" | boxes, no text; in-the-wild, many US vehicles | annotations CC BY 4.0, images filtered to CC BY 2.0 by the repo fetcher ([facts](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)) | the one permissive public set; a person transcribes locally; extend the fetcher to the train split for sample size |
| OpenALPR benchmark, end-to-end | 222 US, 108 EU, 115 Brazilian images with plate text | repository AGPL-3.0 ([repository](https://github.com/openalpr/benchmarks)); the images' own licence is not stated | local evaluation only, never redistributed; the only small public US set with text |
| UFPR-ALPR | 150 vehicles x 30 frames (4,500 images), Brazilian plates, text and corners | academic, non-commercial, on request ([paper](https://arxiv.org/abs/1802.09567)); verify current terms at request | the public multi-frame benchmark for consensus |
| RodoSol-ALPR | 20,000 toll-camera images, Brazilian and Mercosur plates, cars and motorcycles | academic, non-commercial, on request ([repository](https://github.com/raysonlaroca/rodosol-alpr-dataset)) | reader cross-check, two-row motorcycle plates |
| CCPD | over 300,000 Chinese parking images (Base, DB, Blur, FN, Rotate, Tilt, Challenge, Green) | README states MIT ([repository](https://github.com/detectRecog/CCPD)); Xu et al., ECCV 2018 | robustness strata (blur, rotation, tilt); score the alphanumeric part only |
| LPLC | 10,210 images, 12,687 plates, four legibility levels, text for 11,083 | academic, non-commercial, on request ([repository](https://github.com/lmlwojcik/lplc-dataset), [arXiv 2508.18425](https://arxiv.org/abs/2508.18425)) | confidence calibration by legibility level |
| UC3M-LP | 1,975 images, 2,547 Spanish plates | custom terms ([repository](https://github.com/ramajoballester/UC3M-LP)); read before use | optional EU cross-check |
| Set P | private high-resolution vehicle photos with bursts | private | owner-run; the real target; numbers only |
| The private plate-box set | private plate boxes for detector training | private | detector prototype only; never named, never located |

No large public US set with plate text exists in the search results (the closest are the
222-image OpenALPR set and a 2025 low-resolution video set). US accuracy claims therefore rest on
Open Images transcriptions plus set P.

## 4. Candidate designs

✅ recommended, 🟡 prototype decides, ❌ rejected.

### 4.1 Reader

| # | Design | Verdict | Why |
|---|---|---|---|
| R1 | PP-OCRv5 pipeline on the plate box (today) plus dominant-text selection | ✅ default and baseline | in place, Apache-2.0, already on Triton; handles two-line plates through line detection |
| R2 | PP-OCRv5 recognition only on the plate box (skip text detection) | 🟡 P1 | saves the detection pass; loses line selection, so header and state text may leak into the reading |
| R3 | `en_PP-OCRv5_mobile_rec` (English dictionary) instead of the multilingual model | 🟡 P1 | vendor reports 85.25 % English against 66.00 % for the multilingual model; the repo already holds `en_ppocrv5_dict.txt`; a smaller softmax (436 classes) is also faster |
| R4 | fast-plate-ocr global model, served by Triton (ONNX Runtime or a TensorRT plan) | 🟡 P1, then J6 | purpose-built and fastest; ships only if it beats R1/R3 on US-like data and the weights' training-data licence is acceptable |
| R5 | PARSeq fine-tuned on transcribed plates | 🟡 later | best accuracy ceiling for a general reader; needs transcriptions and training time; revisit if R1 to R4 fall short |
| R6 | Format-constrained re-decode of the CTC output (top-N per position plus the `text_format` regex) | 🟡 P2 | the OpenALPR pattern; helps where a jurisdiction format is known; US formats vary by state, so gains may be small |
| R7 | Agreement with the configured VLM (`both`), VLM as fallback on low confidence | ✅ keep | already wired; matches fusion and the BLPR pattern; external endpoints allowed as today (OI-1) |
| R8 | TrOCR, the 2018 segmentation-free CNN reader, commercial engines | ❌ | too slow, untrained for US plates, or not open-source |
| R9 | PP-OCRv6 recognition (Apache-2.0, released June 2026) in place of PP-OCRv5 | 🟡 P1 (owner Q-B, review RR-5) | one short comparison against R1 in `pt:P1`; adopted only on a clear win at no more than today's latency, and then for the B5c OCR model too ([06c_b5_crops_ocr_faces.md](06c_b5_crops_ocr_faces.md)) |

### 4.2 Confidence and tiers

| # | Design | Verdict |
|---|---|---|
| T1 | Per-line **minimum** per-character probability (`rec_char_min`) next to today's mean, threaded to the box as `text_conf_char_min` | ✅ (the mean hides one bad character) |
| T2 | Tiers `confident` (format passes and either readers agree after folding or `text_conf_char_min >= c_hi`), `partial` (a reading that fails that), `none` | ✅, `c_hi` calibrated on the dev half |
| T3 | Calibrated probability (temperature scaling) instead of tiers | 🟡 P2 reports ECE; tiers stay the stored contract |

### 4.3 Burst consensus

| # | Design | Verdict |
|---|---|---|
| C1 | Best frame: the reading with the highest `text_conf_char_min` in the group | ✅ fallback and baseline |
| C2 | Per-position majority vote (equal lengths only) | 🟡 P3 baseline (the literature method) |
| C3 | Confidence-weighted aligned vote: align every reading to the medoid with Levenshtein opcodes ([RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html), MIT), vote per slot weighted by per-character probability, allow an empty slot | ✅ if P3 shows it beats C1 by the bar; named the "medoid alignment vote": a star alignment to the medoid, not ROVER's progressive word transition network (review RR-19) |
| C4 | Hungarian character time-series matching | 🟡 only if C3 falls short |
| C5 | Multi-reader fusion inside the vote (each reader a voter) | 🟡 P3 arm |
| C6 | Overwrite every member's text with the consensus and keep nothing else | ❌ the per-frame read is evidence; keep it in `text_frame` |

### 4.4 Which frames form a burst (observation group)

| # | Design | Verdict |
|---|---|---|
| G1 | Same near-duplicate group (`dup_group_id`) | 🟡 P4: the threshold was never validated at crop level |
| G2 | Same source batch and capture time within a window (EXIF `DateTimeOriginal`, needs a `captured_at` seam on images) **and** vehicle-crop embedding similarity above a floor | ✅ if P4 confirms purity; photo bursts are seconds apart |
| G3 | Group by equal plate text | ❌ for consensus (circular); ✅ for identity grouping (4.6) |
| G4 | Object tracking across video frames | ❌ out of scope (photos, not video) |

### 4.5 Linking plate text to the vehicle box

| # | Design | Verdict |
|---|---|---|
| L1 | Structural: the plate box is a region box on the vehicle item, so the link is the item | ✅ (it is how the region stage works) |
| L2 | One plate box visible in two overlapping vehicle crops (two items in one image): assign it to the vehicle whose box contains it fully and whose bottom edge is nearest the plate's bottom edge; otherwise tier `none` with `text_choice = 'link_ambiguous'` | ✅, P4 measures how often it fires on set P |
| L3 | Several plate boxes in one vehicle item: only a box that dominates by area (at least 2x the next) counts; otherwise tier `none` | ✅ (carried from the former plan) |

### 4.6 Identity grouping, search and clustering

| # | Design | Verdict |
|---|---|---|
| I1 | Identity group = all `confident` boxes in the project with equal `text_fold` (raw mode) or equal `text_key` (keyed mode); a review view "same plate text" | ✅ in the feature wave; needs no clustering change |
| I2 | Must-link and cannot-link edges into clustering | 🟡 later, through the #400 edge interface; must-link needs an embedding floor and compatible coarse class by **name** |
| S1 | Exact, fuzzy (1 or 2 edits) and fragment search on the folded form, normalized by the backend before the query | ✅ |
| S2 | Substring wildcard on raw `text` (today) | ❌ replaced by S1 (no shim) |

### 4.7 Privacy

| # | Design | Verdict |
|---|---|---|
| P-a | Modes `off` / `keyed` / `raw` per project under a deployment ceiling `OP_IDENTITY_TEXT_MAX_MODE` | ✅ |
| P-b | Default mode `raw` (today's behaviour) | ✅ recommended (J1); `keyed` is the stricter alternative |
| P-c | Keyed hashing: deployment master in a read-only secrets mount, per-project salt in PostgreSQL, `HMAC-SHA256` from the standard library | ✅ (carried; the salt moves from OpenSearch to PostgreSQL, which has roles and passwords) |
| P-d | Purge by CLI only; it rotates the salt | ✅ (face precedent; the API has no authentication) |
| P-e | Identity regions read only by local VLM endpoints | ❌ dropped: the owner kept today's behaviour (OI-1, #469) |
| P-f | Contact sheets refused for hosted judges by content | ❌ dropped (OI-2, #471); the refusal by declared kind stays |
| P-g | Log redaction processor and a canary sink test | ✅ whatever the mode |

## 5. Seams the v0.7.0 rebuild must include

"Seam" means a name, type or setting that is cheapest to get right while the rebuild writes the
code around it. Each row names its owning plan file and lane. Rows marked **B2** are creation-time
(index settings, secrets, schema conventions) and belong in the rebuild's own lanes; rows marked
**W5e** land with the feature wave proposed in section 8 (J3).

### 5.1 Per-box text fields (`RegionBox` in `src/domain/region_boxes.py`; nested `region_boxes` mapping)

Existing fields keep their meaning. `text` stays "the box's resolved reading" (after consensus
when a group exists), so search, review and export read one field.

| Field | Python type | OpenSearch mapping | Written by | Meaning |
|---|---|---|---|---|
| `text` | `str \| None` | `keyword` (unchanged) | reader, consensus, human | resolved reading, display form |
| `text_raw` | `str \| None` | `keyword` (unchanged) | reader | unfiltered reading, for audit |
| `text_frame` | `str \| None` | `keyword` | reader | this frame's own resolved reading before consensus (new) |
| `text_norm` | `str \| None` | `keyword`, `normalizer: op_text_compact` | domain function from `text` | uppercase, charset-filtered, separators removed (new) |
| `text_fold` | `str \| None` | `keyword`, `normalizer: op_text_compact`; sub-field `text_fold.sub` of type `wildcard` with the same normalizer | domain function from `text_norm` and the profile's `text_confusables` | match key for exact, fuzzy and fragment search; never shown (new) |
| `text_confidence` | `float \| None` | `float` (unchanged) | reader | the reading's confidence (OCR: min of line means; VLM: category value) |
| `text_conf_char_min` | `float \| None` | `float` | reader | minimum per-character probability over the kept lines (new; needs `rec_char_min`, 5.7) |
| `text_tier` | `str \| None` | `keyword` | reader, consensus | `confident` \| `partial` \| `none` (new; closed set in `src/domain/region_text.py`) |
| `text_reader` | `str \| None` | `keyword` | reader | Triton model name of the reader, or `vlm` (new; **replaces** `text_engine_version`) |
| `text_reader_version` | `str \| None` | `keyword` | reader | model version (new; with `text_reader`, replaces `text_engine_version`, deleted in the same change) |
| `text_run_id` | `str \| None` | `keyword` | reader | `region_text_runs.id` of the pass that read it (new) |
| `text_frames` | `int \| None` | `integer` | consensus | frames in the consensus group, 1 when read alone (new) |
| `text_frames_agree` | `int \| None` | `integer` | consensus | frames whose reading equals the consensus after folding (new) |
| `text_consensus_id` | `str \| None` | `keyword` | consensus | `region_text_consensus_groups.id` (new) |
| `text_key` | `str \| None` | `keyword` | keyed mode only | `HMAC-SHA256` hex of the folded text (new) |
| `text_key_version` | `int \| None` | `integer` | keyed mode only | key derivation version (new) |
| `text_source`, `text_vlm`, `text_ocr`, `text_vlm_invalid`, `text_disagreement` | unchanged | unchanged | reader | as today |
| `text_choice` | `str \| None` | `keyword` (unchanged) | reader, consensus | closed set gains `consensus` and `link_ambiguous` |

- Owner: [02](02_control_plane_database.md) data-ownership table (OpenSearch owns every
  text value); code lane **W5e `w5e/region-text-fields`**.
- Index settings (**B2**, owner [03b](03b_stage2_control_data.md) lane 2F, which owns the
  OpenSearch bootstrap `scripts/opensearch/migrate.py`): the items index body gains, at creation,

  ```json
  "analysis": {
    "char_filter": {"op_alnum_only": {"type": "pattern_replace", "pattern": "[^\\p{L}\\p{Nd}]", "replacement": ""}},
    "normalizer": {"op_text_compact": {"type": "custom", "char_filter": ["op_alnum_only"], "filter": ["asciifolding", "uppercase"]}}
  }
  ```

  Confusable folding (O to 0, I to 1, B to 8, S to 5, Z to 2, per profile) is done in the domain
  function, not in the normalizer, because it is a per-profile setting and analysis settings are
  fixed per index. The normalizer is defence in depth so a lower-case or spaced query still
  matches. A test asserts the domain function and `_analyze` with `op_text_compact` agree on a
  fixture list.

### 5.2 PostgreSQL tables (owner [02a](02a_postgres_schema.md), new section "Region text")

No plate text is stored in PostgreSQL: one owner per value (D-40), one place to purge. A schema
test asserts these tables have no `text`-valued column other than the listed codes.

```sql
CREATE TABLE identity_text_policies (           -- privacy mode per project
  project_id          uuid PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
  mode                text NOT NULL CHECK (mode IN ('off','keyed','raw')),
  salt                bytea NULL CHECK (salt IS NULL OR octet_length(salt) = 32),
  key_version         smallint NOT NULL DEFAULT 1 CHECK (key_version >= 1),
  raw_retention_days  integer NULL CHECK (raw_retention_days IS NULL OR raw_retention_days >= 1),
  revision            bigint NOT NULL DEFAULT 1,
  updated_at          timestamptz NOT NULL DEFAULT now(),
  updated_by          text NOT NULL,
  updated_by_user_id  uuid NULL,
  CHECK (mode <> 'keyed' OR salt IS NOT NULL)
);

CREATE TABLE region_text_runs (                 -- one row per reader configuration and pass
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id             uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,      -- NULL: a stream-worker region pass
  trigger            text NOT NULL CHECK (trigger IN ('region_pass','reread','consensus','rekey')),
  reader             text NOT NULL,                                          -- Triton model name or 'vlm'
  reader_version     text NOT NULL,
  profile_config_id  uuid NULL REFERENCES config_documents(id) ON DELETE SET NULL,
  profile_revision   integer NULL,
  params             jsonb NOT NULL DEFAULT '{}',   -- thresholds, consensus method; pydantic-validated
  counts             jsonb NOT NULL DEFAULT '{}',   -- boxes_read, confident, partial, none, groups
  started_at         timestamptz NOT NULL DEFAULT now(),
  finished_at        timestamptz NULL,
  created_by         text NOT NULL
);
CREATE INDEX ix_region_text_runs_project ON region_text_runs (project_id, started_at DESC);
CREATE UNIQUE INDEX uq_region_text_runs_open_pass
  ON region_text_runs (project_id, reader, reader_version, profile_config_id, profile_revision)
  NULLS NOT DISTINCT WHERE trigger = 'region_pass' AND finished_at IS NULL;

CREATE TABLE region_text_consensus_groups (     -- one observation group (a burst of one vehicle)
  id          uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id  uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  run_id      uuid NOT NULL REFERENCES region_text_runs(id) ON DELETE CASCADE,
  basis       text NOT NULL CHECK (basis IN ('near_dup','capture_window','human')),
  method      text NOT NULL CHECK (method IN ('best_frame','position_vote','aligned_vote')),
  n_frames    integer NOT NULL CHECK (n_frames >= 1),
  n_agree     integer NOT NULL CHECK (n_agree BETWEEN 0 AND n_frames),
  confidence  real NULL,
  tier        text NOT NULL CHECK (tier IN ('confident','partial','none')),
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE region_text_consensus_members (
  group_id    uuid NOT NULL REFERENCES region_text_consensus_groups(id) ON DELETE CASCADE,
  project_id  uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  item_id     text NOT NULL,
  box_id      text NOT NULL,
  weight      real NOT NULL,
  agrees      boolean NOT NULL,
  PRIMARY KEY (group_id, item_id, box_id),
  UNIQUE (project_id, item_id, box_id)          -- a box is in at most one current group
);
```

- Every table gets the `FORCE`d row-level-security policy of 02a section 1.4.
- `project_state.key` CHECK list gains `region_text_consensus_marker` (the sweep's "new readings
  since" marker).
- A recompute replaces a group in one project-scoped transaction (delete the old group, which
  cascades its members, then insert); there is no superseded history.
- The salt is read only through `identity_text_keys.project_key()`; the policy repository's
  read model has no salt field, so no serializer can emit it (a test pins the wire key set).
- Owning lane: **W5e `w5e/region-text-store`**, as its own Alembic revision `region_text`
  (J7: the alternative is to add the tables to `0001_control_plane` in B2 lane 1A).

### 5.3 Celery job kinds and the queue (owner [02b](02b_jobs_locking_events.md) sections 2.1 and 7.3)

| Kind | Project | Queue | Exclusive group, scope | Auto-resume | Soft / hard limit | Max queue wait |
|---|---|---|---|---|---|---|
| `region_text_reread` | yes | `text_read` | `region_text_write`, '' | yes (idempotent per box, checkpoint cursor) | 24 h / 24 h 15 m | 24 h |
| `region_text_consensus` | yes | `jobs` | `region_text_write`, '' | yes (idempotent) | 3 h / 3 h 15 m | 24 h |
| `identity_text_rekey` | yes | `jobs` | `region_text_write`, '' | yes | 24 h / 24 h 15 m | 24 h |

- One exclusive group for all three, so no two of them write the same boxes at once.
- Beat tasks (queue `periodic`): `op.periodic.region_text_consensus_sweep` every 60 s starts
  `region_text_consensus` for a project whose marker is older than 30 s (the burst has settled);
  `op.periodic.identity_text_retention` daily re-keys and drops raw text older than
  `raw_retention_days` where set.
- **Queue `text_read`**: consumer `text-read-worker` (api image, prefork 1, prefetch 20, profile
  `curation`, no GPU device). The GPU is Triton's: the reader is a Triton model, so the queue
  exists to serialize bulk re-reads so they never starve `jobs`, not to own a card. A `gpu_*`
  queue with its own device is needed only if P1 picks a reader Triton cannot serve (none of the
  candidates). The `tests/test_queue_consumers_from_compose.py` rule applies.
- Per-frame reads during a region pass stay in the detection stream worker (D-53), as today.
- Kind-table module: `src/config/job_kinds/region_text.py`. Owning lane: **W5e `w5e/region-text-jobs`**.

### 5.4 OpenSearch queries (owner: the review and crops query builders)

| Search | Query | Normalization |
|---|---|---|
| exact | `term` on `region_boxes.text_fold`, inside the nested `box_query` | the backend applies the profile's fold to the query (`src/domain/region_text.py::fold`) |
| fuzzy | `fuzzy` on `region_boxes.text_fold`, `fuzziness` 1 (2 allowed), `transpositions: true`, `prefix_length: 0`, `max_expansions: 50` | same |
| fragment | `wildcard` on `region_boxes.text_fold.sub` with `*<q>*` | same, wildcard metacharacters escaped |
| keyed mode | exact and fragment refused (400 `identity_text_keyed`); "same text as box X" is a `term` on `text_key` read server-side from box X | the key never travels on the wire |

### 5.5 Config (owner [03b](03b_stage2_control_data.md) lane 2C for the body model; values set in W5e)

No new `config_documents.kind`. The reader, the tiers and the consensus settings are part of what
defines a region, so they are fields of the `region_profile` body (validated pydantic,
`extra='forbid'`). The privacy mode is **not** a config document: it holds a secret salt and its
writes are gated by a deployment ceiling, so it is the `identity_text_policies` table.

| Profile field | Type | Default | Note |
|---|---|---|---|
| `identity_text` | `bool` | `false` | marks the profile's regions as identity text (modes, keys, purge apply) |
| `text_confusables` | `dict[str, str]` | `{}` | fold map for matching only; validated idempotent and closed over `text_charset` |
| `text_recognizer` | `Literal['ocr_pipeline', 'line_reader']` | `'ocr_pipeline'` | `line_reader` reads the whole box crop with one recognition model (R2, R3, R4) |
| `text_line_reader_model`, `text_line_reader_version` | `str` | `''` | Triton name and version when `text_recognizer = 'line_reader'` |
| `text_confident_char_min` | `float` | `0.0` (no confident tier until calibrated) | `c_hi`, calibrated on the dev half; the example profile carries the calibrated value with the calibration run id in a `_comment` |
| `text_consensus_method` | `Literal['off', 'best_frame', 'position_vote', 'aligned_vote']` | `'off'` | set from P3 |
| `text_consensus_basis` | `Literal['near_dup', 'capture_window']` | `'capture_window'` | set from P4 |
| `text_consensus_window_s` | `float` | `5.0` | set from P4 |
| `text_consensus_min_frames` | `int` | `2` | |

Environment (owner [08](reference/environment_and_setup.md) section 4 delta table):

| Variable | Values | Default | Tier |
|---|---|---|---|
| `OP_IDENTITY_TEXT_MAX_MODE` | `off`, `keyed`, `raw` | `raw` (J1) | ADVANCED; read live; malformed refuses startup |
| `OP_IDENTITY_TEXT_SECRETS_DIR` | path | `/run/secrets/op_identity_text` | ADVANCED; the master is a regular non-symlink file of exactly 32 bytes |

Secrets (**B2**, owner [03a](03a_stage1_foundation.md) lane 1C, which owns services,
secrets and the installer): the installer and checkout setup create `./secrets/identity_text/master`
(32 random bytes, mode 600, never printed) through the shared env generator; compose mounts it
read-only into `api`, `curation-detection-worker` and `text-read-worker` only.

### 5.6 API and wire (owner: [curation_api_contract.md](../curation_api_contract.md); `src/services/curation/wire.py`)

None of these routes exist yet. Proposed, written without a leading slash:

| Route | Method | Answer | Note |
|---|---|---|---|
| `curation/projects/{project}/identity_text/policy` | GET | mode, ceiling, effective mode, `key_version`, `raw_retention_days`, revision | never the salt |
| `curation/projects/{project}/identity_text/policy` | PUT | 200; 409 `revision_conflict`; 422 `identity_text_mode_above_ceiling` | `expected_revision` required |
| `curation/projects/{project}/region_text/reread` | POST | 202 + job id | body: item filter, optional reader override; `Idempotency-Key` accepted (D-58) |
| `curation/projects/{project}/region_text/consensus` | POST | 202 + job id | recompute all or a filter |
| `curation/projects/{project}/region_text/runs` | GET | runs with counts | no text |
| `curation/projects/{project}/identity_text/groups` | GET | raw mode: folded text, item count, first and last `indexed_at`, a representative box; keyed mode: counts and representative box only | paged, `confident` tier only |

Query parameters on the existing crops listing (`GET {P}/crops`): `region_text`,
`region_text_match` (`exact` \| `fuzzy` \| `fragment`), `same_text_as` (`<item_id>:<box_id>`). The
regions-tab `text` filter moves to the same builder (S2 deleted).

Wire, per element of `region_boxes` (`region_boxes_to_wire`): adds `text_frame`, `text_conf_char_min`,
`text_tier`, `text_reader`, `text_reader_version`, `text_frames`, `text_frames_agree`,
`text_consensus_id`; drops `text_engine_version`. In keyed mode every text-valued field is absent
and `text_has_key: bool` is present. `text_norm`, `text_fold`, `text_key` and `text_key_version`
are never on the wire. SSE payloads carry counts only (a test pins the key set). `make contracts`
runs in the same commit as each wire change. #470's access rule (W4) attaches to this one
serializer.

### 5.7 OCR pipeline output (owner [06c](06c_b5_crops_ocr_faces.md) track `ocr`)

- New output tensor `rec_char_min` (FP32, `[128]`) in `ocr_pipeline`: the minimum per-character
  max probability over the decoded characters of each line, computed in `CTCLabelDecode` next to
  the mean. `rec_scores` and `texts` stay byte-identical (parity test on a fixed fixture).
  Padded slots are 0.0 and ignored by count, like `rec_scores`.
- Threaded through `src/clients/triton_ocr.py` and the cascade OCR recognizer (`OcrLine.char_min`).
- `MODEL_VERSION` bump. Track `ocr` already rewrites the decode loop (5c-O2), so this is one more
  output in the same change, not a second edit.

### 5.8 Privacy baseline (owner [05](05_b4_stateless_api.md) W4; proposed lane `b4/identity-text-privacy`)

- A structlog processor `redact_region_text` in `shared_processors` of `configure_logging`
  (`src/core/logging.py`), so structlog and foreign stdlib records both pass through it; it masks
  values under `text`, `text_raw`, `text_frame`, `text_vlm`, `text_ocr` and model-reply keys.
- The sink inventory (section 7.4) as a canary test.
- `OP_IDENTITY_TEXT_MAX_MODE` parsing and `SECURITY.md` "Privacy: identity text".
- A separate lane from `b4/privacy-hardening` because that lane is scoped from private notes; the
  coordinator checks the two own disjoint files before the cut.

## 6. Prototype protocol

### 6.1 Code location

**Recommendation: tracked, under `scripts/research/identity_text/`, with exact pins in
`scripts/research/identity_text/requirements.txt`.**

- Why tracked: [docs/research/CLAUDE.md](../../research/CLAUDE.md) requires every result to have a
  reproduce command from a clean checkout; the naming scan and review then cover the scripts;
  the consensus and scoring functions are pure and get offline unit tests in `tests/research/`
  (the existing research test layout), so the production port starts from tested code.
- Why exact pins: the existing research requirement files are unpinned, so a number measured
  today cannot be reproduced next month. Pins follow D-44: every line `==`, resolved at P0 and
  recorded with `pip freeze` in the research note.
- Why not gitignored: an untracked script cannot be reviewed, reproduced or ported; the privacy
  risk is in the **data**, not the code, and is handled by the data rules (6.7).
- The scripts take every dataset location as an argument and contain no private names, paths or
  plate strings; outputs go only under `artifacts_local/identity_text/<run_id>/` (gitignored); the
  scorer refuses an output path outside it.

Layout: `manifest.py` (dataset ids, sha256, split; no text), `readers/` (one module per arm,
each writing predictions JSONL), `consensus.py` (C1 to C3, pure), `score.py` (metrics, aggregates
JSON), `bench.py` (throughput), `group.py` (P4 burst grouping).

### 6.2 The venv

- A venv outside the repository (for example `$SCRATCH/venvs/identity-text`), created with
  `python3.12 -m venv`, called by its binaries, never sourced.
- Packages (exact versions resolved at P0): `onnxruntime-gpu`, `onnx`, `numpy`, `opencv-python-headless`,
  `pyclipper`, `fast-plate-ocr[onnx-gpu]` (1.1.0 today), `torch` and `torchvision` (CUDA wheels,
  for PARSeq and TrOCR), `transformers` (TrOCR), `rapidfuzz`, `statsmodels`, `scipy`, `ultralytics`
  at the program pin (8.4.197, for the P6 detector arm), `pytest`.
- The PP-OCRv5 arms run **the repo's own pre- and post-processing** (`models/ocr_pipeline/1/ocr_postprocess.py`,
  `read_dominant_text`) on ONNX Runtime with the same ONNX files the export step uses, not the
  `paddleocr` package, whose resize and decode differ; otherwise the baseline would not be the
  product.

### 6.3 Datasets and sampling

| Set | Sample | Split | Ground truth |
|---|---|---|---|
| Open Images V7 plates (train split added to the fetcher, CC BY 2.0 only) | count eligible boxes first; target 2,000 legible plates | 50/50 dev/test by image, seed `20261104` | a person transcribes locally; the test half blind (no reader output shown) |
| OpenALPR end-to-end (US, EU, BR) | all 445 images | test only | the benchmark's text |
| UFPR-ALPR (J5) | all 150 vehicles x 30 frames; also subsampled to 3, 5 and 8 frames to mimic photo bursts | the authors' split | the dataset's text |
| CCPD | 2,000 stratified over Base, Blur, Rotate, Tilt, DB, FN, Challenge | test only | alphanumeric part only (the province character is excluded and the exclusion stated) |
| LPLC (J5) | 2,000 stratified by legibility level | 50/50 | the dataset's text |
| Set P | owner's choice; suggested 1,000 vehicle crops in at least 150 bursts | 50/50 by burst | transcribed locally by the owner; never leaves the machine |
| The private plate-box set | detector training and its own holdout (P6 only) | its existing split | boxes only |

Registered before any read (phase R0, a docs commit): the bars of 6.5, the sample sizes, the seed
and the split rule. The confident-tier bar (precision 0.99) needs at least 300 confident test
readings with zero errors (rule of three) or about 475 with one error (Clopper-Pearson); register
n of at least 500 confident test readings per reader that claims the bar.

### 6.4 Prototypes

| ID | Question | Arms | Data | GPU estimate |
|---|---|---|---|---|
| P0 | Data and harness ready | fetcher train split, manifests, transcription tool (local), determinism check | all | none (CPU and person time) |
| P1 | Which reader is best on plates? | R1, R2, R3, R4 (`cct-s-v2`, `cct-xs-v2`), R5 zero-shot PARSeq, TrOCR (accuracy reference only), R7 local VLM | Open Images, OpenALPR, CCPD, LPLC, set P | about 6 h on one card |
| P2 | Does `rec_char_min` give a usable confident tier? | line mean against character minimum; `c_hi` sweep; R6 format re-decode | dev halves; read on test once | about 1 h |
| P3 | How much do bursts add? | C1, C2, C3, C5 at 1, 3, 5, 8 and 30 frames | UFPR-ALPR, set P | about 1 h (reuses P1 predictions) |
| P4 | Which frames are one vehicle, and how often is linking ambiguous? | G1 thresholds 0.95 to 0.99; G2 windows 2, 5, 10 s with embedding floors; L2 rule | set P (needs local capture times) | about 1 h (embeddings) |
| P5 | Throughput of the winning reader | ORT CUDA EP and TensorRT EP at batch 1 and 32; the OCR BLS ms as reference | 1,000 plate crops | about 1 h |
| P6 | Plate boxes: segmenter plus text hint against a detector trained on the private plate-box set | end-to-end read accuracy, not only box mAP | set P, Open Images | about 4 h (training) |

### 6.5 Metrics and bars

- **Character accuracy** = 1 - CER, CER = sum of Levenshtein distances / sum of ground-truth
  lengths, on normalized strings (`text_norm` rule).
- **Full-plate exact match** on normalized strings; also reported after confusable folding.
- **Burst-consensus gain** = exact match of the consensus per vehicle minus exact match of the best
  single frame by confidence (C1); also against the mean single-frame rate. Paired bootstrap 95 %
  interval (10,000 resamples, by vehicle).
- **Confident tier**: precision with a Clopper-Pearson interval (`statsmodels`
  `proportion_confint(method='beta')`), coverage (share of plate boxes reaching the tier),
  risk-coverage curve, ECE (15 bins).
- **Throughput**: plates per second at batch 1 and 32, p50 and p95 ms, peak GPU memory; plus the
  region-pass ms per vehicle crop.
- **Reader comparisons**: paired on the same plates, McNemar's exact test plus the paired bootstrap
  difference.
- Bars (registered in R0): a new reader is adopted only if exact match beats R1 by at least 2
  points on Open Images test **and** set P with the paired interval excluding 0, with throughput no
  worse than the OCR BLS; consensus ships if its gain over C1 is at least 2 points on both
  UFPR-ALPR and set P; the confident tier ships only with test precision lower bound at least 0.99
  and coverage of at least 30 % of plate boxes.

### 6.6 Repetition and GPU etiquette

- Accuracy runs are deterministic: run once; prove it by re-running a 200-plate subset and
  asserting byte-identical predictions. Throughput: 3 repetitions, 30 s steady state after
  warm-up, median and min-max, with the load average beside each row.
- GPUs 0 and 2 only (`CUDA_VISIBLE_DEVICES`); GPU 1 belongs to other workloads. **One prototype job
  at a time** across both cards, serialized with `flock` on the shared GPU lock file. Before a job:
  `nvidia-smi` and `uptime`; if a peer session holds the card, ask it and wait for its answer;
  never stop another stack. No `pgrep -f`, `pkill -f` or sleep/until loops in any agent brief.
- No Docker builds; no compose stack is needed for P0 to P5. P6 uses the trainer script in the
  venv, not the trainer image.

### 6.7 Stop rules

| Stop when | Then |
|---|---|
| P1: no candidate clears the reader bar | keep R1 (or R3 if it alone clears it); no dedicated reader ships |
| P1: the winner's weights or training data have unclear terms | it is not a default; J6 decides an opt-in |
| P2: confident-tier lower bound below 0.99 at 30 % coverage | no `confident` tier; identity groups are a review view over `partial` readings only, never edges |
| P3: consensus gain below the bar | ship C1 (best frame) only |
| P4: burst purity below 0.95 for every basis | consensus stays off by default; manual groups only |
| Any prototype exceeds twice its GPU estimate | stop, write down why, ask |
| A plate string or a set-P path appears in a tracked file | stop, remove it before any commit, rewrite history if it was committed |

### 6.8 Results recording

- Public aggregates go to `docs/research/identity_text_eval.md` (new): setup, method, results,
  findings, open issues, reproduce; a row in `docs/research/README.md` and in
  `docs/design/README.md`.
- Set P appears by its numbers only (counts, rates, intervals); no folder or file names, no
  images, no example readings.
- **Plate strings never appear in any committed file**: not in notes, manifests, tests, fixtures,
  issue comments or commit messages. Transcriptions and predictions live only in
  `artifacts_local/identity_text/`. Tests use canaries that fail every plate format: a character outside `[A-Z0-9]`, or
  12 characters (review 3.7; [06e_w5e_plate_text.md](06e_w5e_plate_text.md)).
- Per-run JSON, logs and GPU samples stay in `artifacts_local/`; never bulk-deleted.

## 7. Production design

### 7.1 Flow

1. **Find plate boxes**: the region stage on vehicle items (segmenter, text hint, or a deployment's
   own detector) as today.
2. **Read**: the detection stream worker reads each plate box with the profile's recognizer;
   `resolve_region_text` gains `text_conf_char_min`, `text_tier`, `text_reader`,
   `text_reader_version`, `text_run_id`, `text_frame` (= the frame's own reading), `text_norm` and
   `text_fold`; the linking rules L2 and L3 run here. Writes go through `region_boxes` and the OCC
   helpers; human and import-validated boxes are never overwritten (`src/clients/occ_locks.py`).
   The worker sets `region_text_consensus_marker`.
3. **Consensus**: the sweep starts `region_text_consensus`; it forms observation groups (P4 basis),
   runs the P3 method in `src/domain/region_text.py` (pure), writes the group rows and, per member
   box, `text`, `text_choice = 'consensus'`, `text_frames`, `text_frames_agree`,
   `text_consensus_id`, `text_tier`. A human-typed member reading is authoritative: the group takes
   it and no automated vote overrides it.
4. **Mode transform** at the `_drop_region_text` hook: `raw` stores as above; `keyed` computes
   `text_key` and drops every text-valued field and the overlapping item text lines (section 7.4,
   S3); `off` reads nothing for identity regions.
5. **Search and groups**: section 5.4 and 5.6.
6. **Clustering**: nothing in v0.7.0; the identity groups are the evidence a later #400 edge
   consumer reads.

### 7.2 Keys (keyed mode, carried from the former plan with the salt moved)

```text
master       = 32 bytes, <OP_IDENTITY_TEXT_SECRETS_DIR>/master
salt         = secrets.token_bytes(32)        per project, identity_text_policies.salt; replaced on purge
project_key  = HMAC-SHA256(master, b"op/identity-text/project/v1\x00" + salt)
text_key     = HMAC-SHA256(project_key, b"k\x00" + key_version + b"\x00" + text_fold.encode()).hexdigest()
```

Standard library only (`hmac`, `hashlib`, `secrets`; [hmac](https://docs.python.org/3/library/hmac.html),
RFC 2104), as a single-block HMAC PRF in the manner of NIST SP 800-108r1. An attacker needs the
master (host mount) and the salt (PostgreSQL); purge replaces the salt, so a later index copy plus
the master cannot test old keys. The plate space is small enough to enumerate, so a key is only as
safe as the master. Distance-1 neighbourhood keys (SymSpell-style) are deferred (J8).

### 7.3 Modes, ceiling and transitions

| From | To | Action |
|---|---|---|
| `raw` | `keyed` | `identity_text_rekey` computes keys from stored text, then removes text fields and overlapping item text lines; dry-run counts first; resumable |
| `keyed` | `raw` | forward only: new reads store text; old boxes keep keys only; allowed only under a `raw` ceiling |
| any | `off` | identity regions are not read; stored data stays until purge; status reports counts |
| any | purge | requires effective mode `off` in `.env` and in the running API |

Effective mode = `min(project mode, ceiling)` on an ordered enum (`off < keyed < raw`). Project
creation writes the policy row explicitly (create-only first write), so no project is without
one; an identity-text profile with no row refuses identity reads (fail closed).

### 7.4 Sink inventory (the canary test's table)

| # | Sink | `keyed` rule |
|---|---|---|
| S1 | `region_boxes[].text`, `text_raw`, `text_frame`, `text_vlm`, `text_ocr`, `text_norm`, `text_fold` | never written |
| S2 | `text_vlm_invalid`, `text_choice` | codes only (asserted) |
| S3 | `item_text_lines`, `item_text_tokens` on the parent item | drop lines whose box is at least half inside an identity box; recompute tokens from kept lines |
| S4 | human-typed text | keyed then discarded; the edit response does not echo it |
| S5 | region text repair (`src/services/curation/region_text_repair.py`) | skips identity boxes |
| S6 | logs (API, detection worker, VLM worker, text-read worker) | redaction processor plus no-content logging; canary test |
| S7 | job rows (`jobs.params`, `progress`, `result`, `error_message`) and `events.payload` | counts and codes only; canary test scans them |
| S8 | `region_text_runs.params`, `counts` | no text (schema and canary tests) |
| S9 | VLM requests | crop pixels go to the configured endpoint as today (OI-1) |
| S10 | exports (YOLO, single-class, classification) | no text in any mode (a test iterates every writer with a non-empty registry assertion) |
| S11 | combine target | identity fields stripped; the target re-reads if it wants them |
| S12 | clone settings | mode copied, fresh salt generated |
| S13 | core `/ocr` index | out of curation scope; owned by the W4 privacy lane (#472) |
| S14 | research artefacts | `artifacts_local/` only |

### 7.5 Purge (CLI only)

`./openprocessor identity-text purge --project <slug> [--dry-run | --yes]`, script
`scripts/purge_identity_text.py` modelled on `scripts/purge_face_data.py`: refuses unless the
effective mode is `off` and no `region_text_write` job is active; deletes S1 to S4 content, keys
and consensus groups; verifies every post-count is 0 (else exit 1); rotates the salt last; appends
a counts-only line to `data/audit/identity_text_purge.jsonl`. Master rotation is a separate,
documented-destructive CLI action. Out of reach (documented): snapshots and backups, older logs,
downloaded copies, `artifacts_local/`.

## 8. Landing lanes

| When | Lane | Owns | Depends on | Proves |
|---|---|---|---|---|
| now (docs) | R0 registration | `docs/research/identity_text_eval.md` (bars, n, seed) | this plan | committed before any read |
| W2.1 | **B2 lane 1C** (`b2/compose-installer`) adds the secret | `./secrets/identity_text/` generation in the shared env generator, compose mounts | 03a | installer test: file exists, mode 600, 32 bytes, never printed |
| W2.2 | **B2 lane 2F** adds the analysis settings | items index body: `op_alnum_only`, `op_text_compact` | 03d | `_analyze` parity test against the domain normalizer (`-m db` is not needed; OpenSearch fake or live) |
| W2 to W4 (GPU idle in the program) | research: P0 to P6 | `scripts/research/identity_text/`, `tests/research/test_identity_text_*.py`, `artifacts_local/` | J4, J5 | section 6 |
| W4 | `b4/identity-text-privacy` | `src/core/logging.py` processor, the env parser `src/config/identity_text.py`, canary and sink tests, `SECURITY.md` section | B2 merged | 5.8 tests red then green |
| W5c | **B5c track `ocr`** adds `rec_char_min` | `models/ocr_pipeline/1/*`, `config.pbtxt`, `MODEL_VERSION`, `src/clients/triton_ocr.py`, the cascade OCR recognizer | 07c | parity: `rec_scores` and `texts` byte-identical |
| W5e (new wave, J3) | `w5e/region-text-store` | Alembic revision `region_text`, repositories, RLS tests | W5c | 5.2 |
| W5e | `w5e/region-text-fields` | `src/domain/region_boxes.py`, `src/domain/region_text.py` (normalize, fold, tiers, consensus), `region_text.py`, `region_text_stage.py`, mapping, wire, contracts, review and crops query builders | store lane's model names | 5.1, 5.4, 5.6 |
| W5e | `w5e/region-text-jobs` | `src/config/job_kinds/region_text.py`, the three handlers, beat entries, `text-read-worker` in compose, the reread and consensus routes | store lane | 5.3 |
| W5e | `w5e/identity-text-modes` | `identity_text_keys.py`, policy routes, rekey, purge script and CLI, combine and clone stripping | fields and jobs lanes | 7.3 to 7.5 |
| W5e | `w5e/region-text-reader` (only if P1 adopts R3 or R4) | Triton model directory for the reader, export step, installer model group | P1, J6 | P5 numbers reproduced on Triton |
| W6 | R-final row | acceptance: read a public sample, consensus, exact/fuzzy/fragment search, keyed mode and a purge leaving 0 | W5e | [07](07_release_acceptance.md) |

W5e runs one full sweep, like every wave, and one short GPU gate in GPU window 2 (the reader on
Triton and the P-ocr gate for `rec_char_min`).

## 9. Risks

| Risk | Mitigation |
|---|---|
| Plate text can identify a vehicle and, through it, a person; deployments may be subject to data-protection rules | modes, ceiling, purge, redaction and the sink test give operators the controls; `SECURITY.md` states what each mode stores and what the purge cannot reach. The plan does not decide legal questions for operators |
| `raw` as the default keeps readable text in an index without authentication by default | the ceiling lets an operator forbid `raw` deployment-wide; J1 asks the owner to confirm the default |
| Keyed mode gives false comfort: the plate space is small, so keys fall to anyone holding master and salt | the master never sits with the data; purge rotates the salt; docs say so |
| Hosted VLM endpoints receive plate pixels when external use is enabled | owner decision OI-1 (keep today's behaviour); the W4 `SECURITY.md` statement says it plainly |
| Readings leak through a sink nobody listed | the canary test walks every written body, log, job row and event; a new sink without a row fails CI |
| Item OCR lines keep the plate after the box is keyed | S3 rule and a search test (`?item_text=`), not only a field test |
| Consensus merges two vehicles into one group | P4 purity bar; the embedding floor; per-frame `text_frame` kept so a wrong group is reversible |
| A reader's weights carry unclear training-data terms | J6; default stays Apache-2.0 PP-OCRv5 |
| Request-only datasets restrict use to academic, non-commercial research | J5; only aggregates are published, with citations |
| US plates are under-represented in public sets | Open Images transcription plus set P; claims are scoped to the sets measured |
| Confusable folding merges truly different plates (for example a real O and 0) | folding is for matching only; display and raw text are never folded; fuzzy matches are suggestions |
| `rec_char_min` changes other OCR consumers | parity test pins `rec_scores` and `texts` |
| Analysis settings missed at index creation | B2 lane 2F seam; otherwise a close-and-reopen or reindex later |

## 10. Owner questions

| J | Question | Recommendation |
|---|---|---|
| J1 | Default privacy mode for identity-text projects, and the ceiling default: `raw` (today's behaviour) or `keyed`? | `raw` for both, consistent with keeping today's behaviour (OI-1); operators lower the ceiling |
| J2 | Approve the privacy model (modes, ceiling, master plus salt, CLI-only purge, sink test) as carried from the former plan, without the local-only VLM rule (dropped by OI-1)? | yes |
| J3 | Where does the feature land: a new wave W5e after W5d and before W6, folded into W5d, or after the tag? | W5e: "a must" implies inside v0.7.0, and it needs B5c's OCR change first |
| J4 | Run prototypes P0 to P6 during W2 to W4 on GPU 0 or 2 (an exception to MQ-9, research after the tag)? | yes: the program uses no GPU in those waves, and W5e needs the answers |
| J5 | May the owner request and use UFPR-ALPR and LPLC (academic, non-commercial terms) for internal evaluation, publishing aggregates with citations? | yes for UFPR-ALPR (the only public multi-frame set) and LPLC; RodoSol only if P1 needs more two-row plates |
| J6 | If a dedicated reader (fast-plate-ocr) wins P1 but its training data's terms are not stated, may it ship as an opt-in installer model group, or not at all? | opt-in only, off by default, with the provenance gap stated in `ATTRIBUTION.md` |
| J7 | The four PostgreSQL tables: their own revision in W5e, or in `0001_control_plane` now? | their own revision in W5e (B2 scope stays tight; Alembic adds them cleanly) |
| J8 | Keyed-mode distance-1 neighbourhood keys (near-match and veto without text)? | defer; decide on P2's rate of distance-1 misreads among confident readings |
| J9 | Store the issuing jurisdiction (state or country) when a reader provides it? | not in v0.7.0; P1 records whether the fast-plate-ocr region head is accurate enough to justify a field |
| J10 | Raw-mode retention: optional or required? | optional (NULL keeps text until purge), so `raw` stays today's behaviour |
| J11 | Consensus writes the group's reading into `text` and keeps the frame's own in `text_frame`: agreed? | yes |
| J12 | May a detector trained on the private plate-box set be used in P6 and in the owner's deployments (never shipped publicly)? | yes; the public example keeps the segmenter and text hint |
| J13 | Add a `captured_at` field (EXIF `DateTimeOriginal` with sub-seconds) to images, owned by the ingest lane, for burst grouping (G2)? | yes, if P4 shows G2 beats G1 |

## 11. Required changes to other plan files (for the integration agent)

| File | Change |
|---|---|
| [02_control_plane_database.md](02_control_plane_database.md) | data-ownership table: region text values in OpenSearch; reader runs, consensus groups and identity-text policies in PostgreSQL |
| [02a_postgres_schema.md](02a_postgres_schema.md) | new section "Region text" with the four tables of 5.2, their RLS, `region_text_consensus_marker` in the `project_state` CHECK, the "no text columns" schema test |
| [02b_jobs_locking_events.md](02b_jobs_locking_events.md) | kind-table rows of 5.3; queue row `text_read` / `text-read-worker`; the two beat tasks |
| [03a_stage1_foundation.md](03a_stage1_foundation.md) | lane 1C: the `identity_text` secret in the shared env generator and the three compose mounts |
| [03b_stage2_control_data.md](03b_stage2_control_data.md) | lane 2F: `op_alnum_only` and `op_text_compact` in the items index body; lane 2C: room in the `region_profile` body model for the 5.5 fields (values land in W5e) |
| [05_b4_stateless_api.md](05_b4_stateless_api.md) and [00_master_plan.md](00_master_plan.md) W4 | lane `b4/identity-text-privacy` (5.8) |
| [06c_b5_crops_ocr_faces.md](06c_b5_crops_ocr_faces.md) | track `ocr`: the `rec_char_min` output and its parity test (5.7) |
| [00_master_plan.md](00_master_plan.md) | wave W5e (section 8), its sweep and its slot in GPU window 2; the prototype window during W2 to W4 |
| [decision_register.md](reference/decision_register.md) section 6 | decision rows for J1 to J13 once answered; MQ-9 exception for J4 |
| [environment_and_setup.md](reference/environment_and_setup.md) | config delta rows for `OP_IDENTITY_TEXT_MAX_MODE` and `OP_IDENTITY_TEXT_SECRETS_DIR` |
| [traceability.md](reference/traceability.md), [github_actions.md](reference/github_actions.md) | #401: from "parked" to W5e with this plan; milestone v0.7.0 on the owner's go |
| [prior_plans_reconciliation.md](reference/prior_plans_reconciliation.md) | the W12 and "plate identity" rows point here |
| [07_release_acceptance.md](07_release_acceptance.md) | one acceptance row (section 8, W6) |
| [09b_research_clustering.md](09b_research_clustering.md) | RQ-7, E-9, OQ-6 and their stop rule point here; E-9 is replaced by P1 to P4 |
| [cluster_search_makemodel_plan.md](../cluster_search_makemodel_plan.md) | section 6.8 and W12 marked superseded by this plan |
| [01_package_pins.md](01_package_pins.md) | `rapidfuzz` (MIT) as a product dependency if C3 ships; research pins stay out of the product files |
| `ATTRIBUTION.md` | PP-OCRv5 English model if R3 ships; the reader of J6 if it ships |
