# OpenProcessor: evaluate EmbeddingGemma 2 for similarity and text semantic search (issue #347)

Status: **PLAN ONLY, not implemented.** Written 2026-11-04 against branch `int/b1-correctness`
at `81c4e4ef`. Nothing in the repository was changed and no model was downloaded or run for
this pass.

Builds on (does not repeat): `docs/research/embedding_model_eval_plan.md` (the model facts,
scenarios S1 to S7, the TensorRT ladder, the metrics schema and the ledger; written 2026-10-28/29),
`docs/design/research_clustering_plan.md` (RQ-3b and experiment E-4, bars in 4.6, stop rules in
7.1), `docs/research/wb2_standard_benchmarks.md` (the 0.7.0 encoder study that kept PE-Core-L),
`docs/research/retrieval_baseline.md` (the measured search baseline) and the owner comments on
#347 of 2026-10-28 and 2026-10-29. This gist is the execution-ready per-issue version: what to
build, in which order, where the code lives, which bars decide, and when to stop.

Reader: a fresh implementation agent with no memory of this session. Claims about the tree are
cited `path:line` at the commit above; claims about the model are vendor claims unless measured,
and the eval plan holds their sources. Re-read the model card, the licence text and the issue
trackers at execution; this ecosystem moves weekly.

In this gist the model is called by its public name from the issue title only; the arms are
`EG2-T` (the 270M text encoder) and `EG2-TV` (text plus the 170M vision tower), as in the eval
plan.

---

## 0. TL;DR

**What I recommend.**

1. **Run the evaluation in three gated stages, cheapest and most likely to pay first**, and stop
   at the first stage that shows nothing:
   - **Stage A (gates, CPU plus minutes of GPU):** licence read in full, precision sanity
     (bf16 against fp32, fp16 expected to fail), loader sanity (reproduce one vendor-reported text
     benchmark within a stated tolerance), and harness parity (the offline MobileCLIP arm must
     reproduce the measured core search numbers on the same 2,000 COCO images).
   - **Stage B (text only, mostly CPU):** placements A and B of the eval plan (OCR / item text
     and VLM caption search) with scenarios S2, S3 and the text half of S6. This is where the
     model is most likely to add something, because today OCR search is keyword-only.
   - **Stage C (text plus vision, GPU in the research window RW-2):** S4 cross-modal, S5
     clustering on real curation crops, S1 speed, the image half of S6, and the reference arms.
   - **Stage D (TensorRT and shipping design):** only for a placement that passed Stage B or C.
2. **Put the harness in the repository** (`scripts/research/embedding_eval/`), not in
   `$HOME/eval_work/scripts/` as the eval plan says, so every number has a reproduce command from
   a clean checkout (`docs/research/CLAUDE.md`, "Every result has a reproduce command"). Reuse the
   tested metric code (`scripts/bench/retrieval_eval_metrics.py`,
   `scripts/research/standard_benchmarks/metrics.py`, `scripts/bench/cluster_eval.py`); write
   no new recall, nDCG or bootstrap.
3. **Do not reuse the WB2 encoder registry as is**: it runs every encoder under fp16 autocast
   (`scripts/research/fine_grained/models.py:67`) and feeds fixed squares
   (`models.py:56-71`). The vendor says this model must not run in fp16, and it takes images
   through its own token-budget processor. The new harness declares a dtype and an input path per
   arm and refuses fp16 for this arm.
4. **Judge it as a complement, never as a replacement.** The owner decided on 2026-10-29: if
   useful, ship it as an optional embedding model, off by default, never replacing PE-Core. So
   the go/no-go is per placement (A item text, B captions, C second image view, D clustering
   view) with the eval plan's bars (`embedding_model_eval_plan.md` 3.6), registered in a docs
   commit before any score is read. RQ-3b's replacement bar does not apply to this arm.
5. **If a placement passes**, the shipping design is: a new optional vector field with its own
   per-field embedding state, a query path that names its field, a generic Triton model pair
   (TensorRT on GPU, onnxruntime on one CPU instance) behind a setting, the backfill as part of
   the Celery `op.jobs.reprocess` family after B2, and the image path reading the GPU-resident
   frame in `gpu_image_prep` (one decode). That design belongs to a follow-up implementation issue
   (J6), not to #347.

**What I am NOT deciding** (section 11): stage gating (J1), whether TensorRT runs only after a
quality pass (J2), the reference-arm scope (J3), GPU and window (J4), how the model name may
appear in new tracked files (J5), where the second-field machinery lands (J6), publication of
local-only rows (J7), the caption VLM (J8), the bar registration (J9), multilingual S7 (J10).

**Stop rule** (section 10): if Stage B gives no Go or Conditional for placements A and B, and a
Stage C screen of S4 on the 2,000-image sample loses to PE-Core and MobileCLIP with no fusion gain,
the verdict is no-go: record it, close #347, no TensorRT work, no reference arms beyond what ran.

**Budget.** Agent time about 5 working days (stages A to C), plus at most one day per TensorRT
tier per arm in stage D. GPU about 4 to 6 hours in RW-2 (encodes, VLM captions for 2,000 images,
latency sweeps, reference arms). CPU about 10 hours (scoring, CPU latency, BM25 index). No person
time except reading the licence (owner).

---

## 1. Context: why the obvious approaches are wrong here

- **"Add it to the WB2 harness and run CARS196/CUB/SOP."** Three problems. The WB2 encoder
  wrapper autocasts to fp16 (`scripts/research/fine_grained/models.py:67`), which the vendor
  warns produces NaN or silently degraded vectors for this family (eval plan 1.1, "Precision";
  5.3, the fp16-overflow row). The wrapper takes fixed-size squares
  (`models.py:64-71`), while this model sizes images by a token budget. And CARS196 is at its
  ceiling for every CLIP-type encoder (Recall@1 0.954 to 0.975, `wb2_standard_benchmarks.md`
  9.1, item 4), so it cannot rank a new one. WB2 itself measured image-to-image only, never text
  search (`wb2_standard_benchmarks.md` 9.6, item 5), and text is the main reason to look at this
  model.
- **"Treat it as the RQ-3b encoder challenger."** RQ-3b's ship bar is a replacement bar: BCubed F
  +0.03 on crops, no loss on CUB/SOP, a text tower, encode time within 2x of PE-Core-L "on the B5
  pipeline" (`research_clustering_plan.md` 4.6). By owner decision this arm can only complement
  PE-Core, so a replacement bar answers the wrong question; and "on the B5 pipeline" is not
  measurable for an arm with no TensorRT engine. It still produces one RQ-3b row (E-4 reads it),
  but #347's verdict is per placement.
- **"Scripts in a throwaway directory outside the repo."** The eval plan puts the harness under
  `$HOME/eval_work/scripts/eg2/` (eval plan 3.1, 3.5). That makes the results unreproducible from
  a checkout, against the research rules. Data and vectors stay outside the repo; code does not.
- **"Name the model in the code."** The naming-leak scan's domain pattern matches the model
  family's name in any tracked file (`scripts/codegen/check_naming_leaks.py:61`); the only
  exemption is the eval-plan note itself (`scripts/codegen/naming_leak_allowlist.txt:41`, which
  also says "generic plumbing must not hardcode it"). Arm keys and Triton model names must be
  generic and the Hugging Face id must come from the command line or an environment variable.
- **"The CPU text path can run bf16 like the GPU."** The host CPU is a Xeon E5-2680 v3 (AVX2 only,
  no AVX-512, no bf16 or VNNI instructions; read with `lscpu` on 2026-11-04). bf16 on it is
  emulated and slow, and INT8 gains little without VNNI. The CPU text arm must run fp32 (ONNX
  Runtime or PyTorch), and its latency is the number that decides placement A's CPU option.

---

## 2. What is there today (read from the code at `81c4e4ef`)

### 2.1 Encoders and search paths in the product

| Fact | Where |
|---|---|
| Curation item vector field is `pe_embedding`, 1024-d, one field | `src/config/curation.py:44`, `:157`; mapping `src/clients/curation_opensearch/base.py:37` |
| `embedding_state` records why an item has no vector, keyed to that one field | `src/domain/embedding_state.py:3-10`, `:32-55`, `:84` |
| PE text encoder: backends `triton` (`pe_text_encoder`, onnxruntime on one CPU instance), `torch`, `onnx`; `OP_PE_TEXT_BACKEND=auto` tries Triton then torch | `src/clients/pe_encoder.py:14-42`, `:167-188` |
| An in-process ONNX text session per uvicorn worker put container RSS at about 84 GiB with 32 workers, so `auto` no longer uses it | `src/clients/pe_encoder.py:29-36` |
| `pe_text_encoder` Triton config: `max_batch_size: 32`, `KIND_CPU` | `models/pe_text_encoder/config.pbtxt:20`, `:43-46` |
| Text queries LRU-cached (256), context length 32, tokens trimmed to the last EOT | `src/clients/pe_encoder.py:45-50`, `:118` |
| Image encoders in the model repository: `pe_image_encoder`, `mobileclip2_s2_image_encoder`, `mobileclip2_s2_text_encoder` | `models/` |
| Curation semantic text search | `src/services/curation/semantic_search.py:332` |
| OCR text: `item_text_lines` (display) and `item_text_tokens` (normalized keyword tokens); matching is lexical | `src/domain/item_text.py:5-6`, `:38-39` |
| GPU preprocessing: one decode into a CuPy frame, PE squares resampled from it, encoder called through BLS | `models/gpu_image_prep/1/model.py:277`, `:286-300`, `:346` |
| UINT8-input export trick and FP16 baking for TensorRT 11 (strongly typed) | `export/uint8_input.py`, `export/trt_utils.py`, `export/build_pe_trt.sh` |

### 2.2 Measured baselines this plan compares against

| Number | Value | Source |
|---|---|---|
| Core text-to-image, MobileCLIP, 2,000 COCO val2017 images, first caption | recall@1 0.567 [0.546, 0.588], recall@10 0.886 [0.872, 0.901], MRR@10 0.671, nDCG@10 0.723 | `docs/research/retrieval_baseline.md:36-41`; raw `docs/benchmarks/retrieval_baseline/text-core_c1_r1.json` |
| Curation text-to-crop, PE-Core, same images | recall@10 0.733 (crops, not whole images; not an encoder comparison) | `retrieval_baseline.md:40`, `:51-56` |
| Image-to-image (category Jaccard >= 0.5 proxy), core MobileCLIP | precision@10 0.476 | `retrieval_baseline.md:64-73` |
| Whole-image PE text-to-image on the same sample | **not measured** ("the experiment that separates them ... (not done)") | `retrieval_baseline.md:55-56` |
| Encoder throughput, research harness, batch 32, fp16, shared GPU | PE-Core-L 133 img/s, 1,835 MiB; MobileCLIP2-S2 815 img/s, 629 MiB | `wb2_standard_benchmarks.md` 9.5 |
| Encoder decision of 0.7.0 | PE-Core-L stays; no challenger passes B1 | `wb2_standard_benchmarks.md` 9.6 |
| Text search latency (91k items) and text encode on a cache miss (about 32 ms through Triton) | as recorded | `docs/research/v070_measurement_log.md` section D (cited by eval plan 2.1) |

The retrieval sample is pinned in `docs/benchmarks/retrieval_baseline/sample_ids.json` (2,000
COCO val2017 images, seed 20261028, with sha256 per file) and re-derived by
`scripts/bench/retrieval_eval.py prep` (`retrieval_eval.py:1-18`, `:52`).

### 2.3 Research code that this plan reuses

| Piece | Where | Use here |
|---|---|---|
| recall@k, MRR@10, nDCG@10, bootstrap CI over queries | `scripts/bench/retrieval_eval_metrics.py:11-56`; tests `tests/test_retrieval_eval_metrics.py` | S2, S3, S4, S6 |
| Recall@K over a gallery, paired bootstrap of two hit vectors, k-means NMI | `scripts/research/standard_benchmarks/metrics.py:19`, `:130`, `:146` | S4 image-to-image deltas, CUB/SOP rows |
| Average-linkage operating cut and partition scores | `scripts/research/standard_benchmarks/score_run.py:46-102` | S5 |
| Cluster-eval harness (pairwise, BCubed, AMI, strata) | `scripts/bench/cluster_eval.py`, `cluster_eval_metrics.py` | S5 |
| Run envelope with git sha, `nvidia-smi`, synthetic refusal | `scripts/bench/scorecard.py:97-160` | every result JSON |
| Decode once, upright RGB, all variants from that frame, sharded resumable cache | `scripts/research/standard_benchmarks/embed_run.py:1-12`, `:45-72` | pattern for the image encode step |
| PE-Core-L and MobileCLIP2-S2 loaders pinned to the production checkpoints | `scripts/research/fine_grained/models.py:37-53`, `:81-92` | baseline arms (with their own fp16 autocast kept, since that is how WB2 measured them; state it) |

### 2.4 What does not exist

No harness for text-to-text or OCR-text retrieval; no BM25 baseline; no offline text-to-image
harness (the retrieval harness goes through the API); no arm with a non-square image processor;
no second item vector field; no per-field embedding state.

---

## 3. Stage A: gates (before any benchmark)

Every gate writes one result JSON. A failed gate stops the arm and is recorded with its cause.

| Gate | What | Pass | Notes |
|---|---|---|---|
| A0 licence | The owner (or the agent, reporting to the owner) reads the model card licence and the licence text and the prohibited-use policy that the card links, in full | compatible with distributing an optional, off-by-default integration in an AGPL repo, with an ATTRIBUTION row | the card says Apache 2.0 but links other terms (eval plan 1.1, "License"); a mismatch is a no-go for shipping, not for measuring |
| A1 revision pin | record the Hugging Face commit sha of the weights and of the processor config; pin `transformers` and `sentence-transformers` versions | recorded | the card carries a deprecation notice pointing elsewhere (eval plan 1.2) |
| A2 precision | on 200 real texts (COCO captions, OCR strings) and 200 real images: cosine of bf16 against fp32 vectors | mean >= 0.999, min >= 0.99, no NaN | a deliberate fp16 run is recorded as the expected failure (or as a surprise, against the community ONNX claim of cosine 0.9998, eval plan 1.4) |
| A3 one space | text vectors from the 270M text-only load against the 440M text+vision load on 1,000 texts | cosine >= 0.999 | the vendor guide says they share one space; if not, the cross-modal scenarios use the 440M text path only |
| A4 loader sanity | reproduce one vendor-reported text task with the maintained `mteb` package ([MTEB](https://github.com/embeddings-benchmark/mteb), Apache-2.0): one small retrieval or STS task named on the card, with the card's prompts | within 1.0 point of the vendor value | catches a missing task prefix (the card says omitting prefixes lowers quality) or a wrong pooling; if no small task matches, record "no anchor" |
| A5 harness parity | the offline MobileCLIP2-S2 arm (production checkpoint, `FG_MOBILECLIP_CKPT`) on the 2,000-image retrieval sample, first caption | recall@1 and recall@10 inside the bootstrap intervals of the measured core numbers (0.546 to 0.588; 0.872 to 0.901) | if offline disagrees with the API path, the offline harness (decode, resize, normalization, ranking) is wrong; fix it before reading any new arm |

A5 is the most important gate: it is the only check that the offline harness ranks the same way
the product does.

---

## 4. Harness design

### 4.1 Where the code lives

```text
scripts/research/embedding_eval/
  __init__.py
  arms.py          arm registry: key, loader, dtype, device, input path (square | processor),
                   prefixes (query / document / none), native dim, Matryoshka dims
  data.py          dataset loaders from pinned manifests (COCO sample, Open Images LN sample,
                   scene-text set, curation crops); returns ids, texts, image paths
  encode.py        decode each image once (PIL, ImageOps.exif_transpose, RGB) and feed every
                   requested arm from that frame; sharded, resumable float32 .npy per
                   (dataset, arm, modality); writes vector metadata (arm, revision, dim, dtype)
  score_xmodal.py  S4: text-to-image, image-to-text, image-to-image, rank fusion
  score_text.py    S2, S3: text-to-text, caption index vs image index, lexical baselines
  score_cluster.py S5: partitions for the curation crops, then scripts/bench/cluster_eval.py
  storage.py       S6: Matryoshka slicing with re-normalization, int8 scalar quantization,
                   index size and k-NN timing in a throwaway OpenSearch
  latency.py       S1: one process per (arm, device, batch), fresh CUDA context
  gates.py         Stage A gates
  report.py        tables from the result JSONs (never typed by hand)
  measured_run.sh  wrapper: GPU lock, host and GPU telemetry, refusal when a foreign
                   process holds the GPU (eval plan section 6 rules)
tests/research/test_embedding_eval.py
```

Dependencies go into a new research requirements file
`scripts/research/embedding_eval_requirements.txt` (research venv outside the repo, as the other
research lists say): `torch`, `transformers`, `sentence-transformers`, `open_clip_torch`, `timm`,
`numpy`, `scipy`, `faiss-cpu`, `onnxruntime`, `nvidia-ml-py`, `mteb`, `pillow`, `pycocotools`.
Nothing goes into the product requirements.

Arm keys are generic: `emb_t` (EG2-T), `emb_tv` (EG2-TV), `pe_core_l`, `mobileclip2_s2`, and for
reference arms `ref_vl_2b`, `ref_vl_8b`, `ref_nv_1b`, `ref_nv_omni`. The model id of every
non-baseline arm comes from an environment variable (`EMB_EVAL_<KEY>_HF_ID`) or `--hf-id`, so no
tracked file spells it (J5).

### 4.2 Arm contract

```text
encode_text(texts, role) -> float32 [n, d], L2-normalized      role in {query, document, none}
encode_images(frames)    -> float32 [n, d], L2-normalized      frames are decoded RGB arrays
```

- `role` selects the vendor prefix (eval plan 1.1, "Task prefixes"). PE-Core and MobileCLIP
  ignore it. A test asserts the prefix reaches the tokenizer for `emb_t`.
- dtype is per arm: `emb_t` and `emb_tv` declare bf16 on GPU and fp32 on CPU; requesting fp16
  raises. Baselines keep the dtype WB2 used (fp16 autocast) and say so in every row.
- Images: square arms receive a centre square cut from the frame (the production rule,
  `fine_grained/preprocess.py`); processor arms receive the frame and apply the vendor processor
  with an explicit token budget (`max_soft_tokens` 280 by default; 140 and 560 in the S1 sweep).
- The vector metadata (arm, revision, dim, Matryoshka cut, normalization) is written next to each
  `.npy`; every scoring function refuses to compare a query matrix and a gallery matrix whose
  metadata differ (the "never mix spaces" rule, eval plan 2.3).

### 4.3 Results and raw data

| What | Where |
|---|---|
| Raw run JSON, logs, telemetry, captions, engines | `artifacts_local/embedding_eval/` (gitignored; the eval plan's ledger layout, part 8) |
| Datasets and vectors | a scratch directory outside the repo, named in each run JSON |
| Small JSONs that back a published number | `docs/benchmarks/embedding_eval/` (new) with `SHA256SUMS`; never pruned |
| Write-up | `docs/research/embedding_eval_results.md` (new; generic file name, see J5) with registration first (section 9) |

### 4.4 Libraries before custom code

| Need | Library (docs) | Not hand-rolled because |
|---|---|---|
| Model loading, prefixes, Matryoshka | `sentence-transformers` and `transformers` (vendor-listed, eval plan 1.4) | the vendor's documented path; the card's prompts are part of the model |
| Exact kNN on up to 100k vectors | FAISS `IndexFlatIP` ([wiki](https://github.com/facebookresearch/faiss/wiki)) | exact, deterministic, already a repo dependency |
| BM25 lexical baseline | OpenSearch `match` query on a throwaway index ([OpenSearch full-text](https://docs.opensearch.org/latest/query-dsl/full-text/match/)) | the product's own engine and analyzer, so the baseline is what a user could have; the eval plan's `rank-bm25` package is dropped (a second, non-product BM25 implementation) |
| Today's OCR match | the product's token rule from `src/domain/item_text.py` called directly | the comparison must be against what ships |
| Bootstrap intervals | `retrieval_eval_metrics.bootstrap_ci`, `standard_benchmarks.metrics.paired_bootstrap` | tested in the repo |
| Cluster scores | `scripts/bench/cluster_eval.py` | tested in the repo |
| Text benchmark anchor | `mteb` | the reference implementation of the vendor's own numbers |
| GPU memory | `nvidia-ml-py` (NVML) plus `torch.cuda.max_memory_allocated` | standard |
| ONNX and TensorRT checks (stage D only) | Polygraphy, `trtexec`, ModelOpt ([TensorRT docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/)) | the repo's existing export path (`export/trt_utils.py`) |

---

## 5. Data (public only)

| Id | Set | Use | Source, licence, pin |
|---|---|---|---|
| D1 | The 2,000 COCO val2017 images of the retrieval baseline, 5 captions each | S1 text inputs, S3, S4 headline, A5 | `docs/benchmarks/retrieval_baseline/sample_ids.json`; captions CC BY 4.0, images Flickr terms; captions are re-derived, never committed |
| D1b | All 5,000 COCO val2017 images | S4 at a larger gallery, S6 | official annotations; record file sha256 in the ledger manifest |
| D2 | A seeded 5,000-image Open Images V7 validation sample that has Localized Narratives | S3 long captions, S4 generalization | Localized Narratives annotations CC BY 4.0; per-image licence recorded at download; a new pin `scripts/datasets/manifests/oi_ln_5000.json` |
| D3 | TextCaps captions with TextOCR words on Open Images images | S2 | annotations CC BY 4.0 (check each dataset page at download); images from Open Images train, disjoint from D2 by construction (D2 is validation) |
| D3b | 1,000 D3 images read by the repo's own OCR (PP-OCRv5) | S2 with real recognition noise | needs the stack in RW-2 (`-p op-rs-rw2`); the OCR strings stay in the scratch directory |
| D4 | Curation crops of `cluster_eval_fresh` (acceptance) and `cluster_eval_3000` (regression row; its held-out half is worn) | S5, the RQ-3b row | `scripts/datasets/manifests/cluster_eval_fresh.json`, `cluster_eval_3000.json`; crops and boxes from the throwaway-project export used by the cluster waves (`scripts/bench/cluster_eval_set.py`), so the crops are the product's |
| D5 | CUB-200-2011 and SOP test splits (local only) | secondary image-to-image rows comparable with WB2 | as WB2 (`wb2_standard_benchmarks.md` section 1); aggregates published only as WB2 did (J7) |
| D6 | XM3600 (only if J10 says yes) | S7 | licence checked at download |

Not used: CARS196 (at the ceiling, cannot rank), any private set (an owner-run repeat of a
winner is possible later, numbers only), any `baseline-*` project.

---

## 6. Stage B: text only (placements A and B)

### 6.1 S2: OCR and item text

| Sub-scenario | Query | Document | Relevant | Arms |
|---|---|---|---|---|
| S2a meaning search | a TextCaps caption of an image | the concatenated OCR strings of each image (TextOCR ground truth) | the same image | `emb_t` at 768/256/128, BM25 (OpenSearch), today's token match, PE text tower (a naive text-to-text arm) |
| S2b noisy string | a deterministic perturbation of one OCR line (case, one or two dropped or swapped characters, a word reorder; seeded) | all OCR lines of the pool | its source line | same arms |
| S2c real OCR noise | S2a with D3b strings (PP-OCRv5 output) instead of ground truth | | | `emb_t` at 256, BM25, token match |
| S2d union | per query, the union of the top 10 of `emb_t` and of the lexical arm | | | reports recall@10 of the union and how many hits only `emb_t` found |

Metric: recall@1/5/10, MRR@10 with bootstrap intervals (`with_ci`); also the share of queries for
which the token match returns nothing and `emb_t` finds the relevant item in the top 10.

### 6.2 S3: VLM captions against direct image search

- Captions for D1 (2,000 images) only: two passes, (a) one held-out human COCO caption per image,
  (b) a caption from the repo's catalog VLM with a fixed, recorded prompt (owner decision of
  2026-10-29, item 4). 2,000 VLM captions at about 1 to 2 s each is about 1 h of GPU; captioning
  D2 as well would be about 3 more hours and is not needed for the decision.
- Index A: `emb_t` vectors of the caption. Index B: image vectors of PE-Core-L, MobileCLIP2-S2
  and `emb_tv`. Query: a different human caption of the same image. Report recall@k for A and
  each B, and the overlap of their top-10 sets; union recall shows whether captions add hits.
- The VLM pass runs under the GPU lock, before any latency run, and the VLM is stopped after
  (eval plan section 6, rule 4).

### 6.3 S6 (text half): storage

Every Stage B metric re-scored at 768, 512, 256 and 128 with re-normalization after slicing,
queries and documents at the same cut; also int8 scalar-quantized 256-d vectors. Vector bytes per
item and the OpenSearch on-disk size of a throwaway index of D3 lines at each dimension (Lucene
engine, the product's mapping shape from `src/clients/curation_opensearch/base.py`).

### 6.4 CPU text latency (part of S1, needed for placement A)

fp32 on CPU (section 1, last bullet), 8 threads pinned with `taskset`, queries of 5 to 20 tokens
and documents up to 512 tokens, batch 1 and 8, against the PE text tower on the same cores.
p50/p95/p99 over at least 30 timed iterations after 3 warm-ups, 3 repetitions, load average
recorded.

### 6.5 Stage B decision

The eval plan's bars (`embedding_model_eval_plan.md` 3.6, rows A, B and Storage), registered
unchanged unless J9 says otherwise:

| Placement | Go | Conditional | No-go |
|---|---|---|---|
| A, OCR / item text | S2a recall@10 and S2b recall@1 clearly above BM25 and the token match (non-overlapping 95 % intervals), **or** the S2d union adds at least 5 recall points at 256-d; CPU p95 at most 50 ms for a 20-token query, or GPU with batching at most the PE text cost | better only at 768-d, or only for noisy strings | no gain over lexical, or CPU p95 above 150 ms |
| B, captions | S3 caption-index recall@10 at least equal to direct image search **and** union recall up by at least 5 points | equal but fully overlapping | worse and not complementary |
| Storage | 256-d loses at most 3 % relative on the Stage B metrics; 128-d at most 10 % for text-only | | 256-d loses more than 5 % |

If both A and B are no-go, run only the S4 screen of 7.1 on D1 before stopping (the stop rule).

---

## 7. Stage C: text plus vision (RW-2, GPU)

### 7.1 S4: cross-modal and image-to-image

- Text-to-image and image-to-text on D1 (2,000) and D1b (5,000) with the standard protocol (the 5
  captions of an image are its relevants). `single_relevant_metrics` scores the first relevant
  hit; for image-to-text with 5 relevants, report recall@k (any hit) and MRR, and say nDCG is the
  first-hit variant.
- Image-to-image on D1 with the category-Jaccard proxy (as `retrieval_baseline.md:27-29`), and on
  D5 (CUB and SOP Recall@1, as WB2, paired bootstrap against PE-Core-L).
- Arms: PE-Core-L whole image (centre square, the production rule), MobileCLIP2-S2, `emb_tv` at
  all four dimensions and two token budgets (140 and 280). The PE-Core-L whole-image text-to-image
  row is also the missing row of `retrieval_baseline.md:55-56` and part of E-12 of the research
  plan; share the vectors, do not encode twice.
- Fusion: reciprocal-rank fusion of PE-Core-L and `emb_tv` result lists after retrieval (never one
  shared space).
- **S4 screen** (used by the stop rule): D1 text-to-image recall@5 of `emb_tv` at 768-d against
  PE-Core-L and MobileCLIP2-S2, plus the fusion row.

### 7.2 S5: clustering view on real curation crops

- Encode the D4 crops with `emb_tv` (crops are images; no text). Partitions with the same methods
  and settings as the PE baseline of the cluster waves: IVF with K = min(512, n // 6) as
  production (`research_clustering_plan.md` 1.1), and average linkage at a cut calibrated on the
  dev half (WB2's operating rule, `wb2_standard_benchmarks.md` section 3).
- Score with `scripts/bench/cluster_eval.py` on the same labels and strata (low resolution,
  elongated, rare, similar classes, person); ARI between the `emb_tv` and PE partitions.
- This is also the RQ-3b row for this arm in E-4; report it there with the replacement bar marked
  "not applicable by owner decision" and the complement bar as the verdict.
- Bar (eval plan 3.6, row D): at least equal on the harness and complementary (wins at least one
  stratum PE loses, ARI below 0.8).

### 7.3 S1: speed and memory

- GPU: `emb_t` and `emb_tv` in bf16 (PyTorch), batch 1, 4, 8, 16, 32; text lengths 16, 32, 128,
  512; image budgets 70, 140, 280, 560. Baselines PE-Core-L and MobileCLIP2-S2 in the same
  harness and dtype rules (fp16 autocast, as WB2). Preprocessing time reported apart from model
  time.
- The production TensorRT numbers (PE image about 90 crops/s at batch 4,
  `docs/research/pe_encoder_batching_benchmark.md`; text encode about 32 ms on a cache miss) are
  context, not equal-footing rows.
- Peak VRAM per arm at batch 32; budget for an optional embedder: at most 2 GiB (text) and 4 GiB
  (vision) (eval plan 5.5, item 4; owner open question).

### 7.4 S6 (image half)

As 6.3 for S4 and S5 metrics, plus k-NN query p50/p95/p99 on a 100k-vector pool (D1b + D2 + COCO
train images, vectors only) in a throwaway OpenSearch under its own compose project
(`-p op-rs-emb`), one index per dimension, Lucene HNSW with the product's parameters. Never the
default project, never the live stack.

### 7.5 Reference arms (see J3)

Recommended scope: `ref_vl_2b` (Qwen3-VL-Embedding-2B, Apache-2.0) and `ref_nv_1b` (Nemotron 3.5
Embed 1B, licence stated as commercial-friendly on its card) on the S4 screen, S2a and S5 only,
served through `transformers` or vLLM (owner clarification of 2026-10-29: LLM-based embedders need
no TensorRT). `ref_vl_8b` and `ref_nv_omni` (research-only licence, measurement only) get one S4
row each, only if time remains in the window. One working day per arm, then recorded as
"not run" with the reason. They bound what a much larger model buys; they are not ship
candidates in this issue unless their licence and size pass the hard gates.

### 7.6 Stage C decision

Eval plan 3.6 rows C and D, and the hard gates (bf16 parity, no NaN in 100k encodes, licence,
a serving path on Triton or onnxruntime without an unmaintained fork, VRAM next to the current
models on one card):

| Placement | Go | Conditional | No-go |
|---|---|---|---|
| C, second image view | S4 text-to-image recall@5 not below PE-Core-L at 256-d or more, image-to-image not below PE-Core-L, and the extra ingest cost fits the measured headroom | wins only in fusion (at least 3 points recall@10) | below PE-Core-L and no fusion gain |
| D, clustering view | as 7.2 | equal everywhere | worse in every stratum |

---

## 8. Stage D: TensorRT and the shipping design (only after a pass)

### 8.1 TensorRT track

Run the eval plan's ladder (`embedding_model_eval_plan.md` 5.4, 5.5) only for the component a
passing placement needs: `emb_t` for A or B (text plan, and an onnxruntime CPU graph), `emb_tv`
vision for C or D. Precisions: fp32 reference and bf16; fp16 is built once to document the
expected failure; INT8 only if bf16 passes parity and speed matters. FP8 and NVFP4 are
`skipped: hardware` on sm_86 (eval plan 5.3). Build in the image the serving Triton uses and
record `trtexec --version`. One working day per tier, then `blocked` with the log.

The engine replaces PyTorch or onnxruntime only if it is at least 1.5x faster at the same batch
or halves VRAM (eval plan 5.5, item 3); otherwise the simpler runtime ships for an optional
model.

### 8.2 Shipping design (for the follow-up issue, see J6)

Only after a Go, an owner yes, and a design doc in `docs/design/` (#365's acceptance rule):

1. **Optional field, off by default.** A new knn_vector field per project at the chosen
   Matryoshka cut, generic name (for example `alt_text_embedding` for placement A or
   `alt_image_embedding` for C), created by a mapping migration in
   `src/clients/curation_opensearch/`. No change to `pe_embedding`.
2. **Per-field embedding state.** `embedding_state` gains a field dimension
   (`src/domain/embedding_state.py` is keyed to `ITEM_EMBEDDING_FIELD` today, `:17`, `:32`, `:84`),
   so "not computed for this field" never reads as "has a vector".
3. **Query path names its field.** A query encoded by this model is scored only against its own
   field (same dimension, cut, normalization); fusion with PE results happens after retrieval. A
   test fails if a vector from one model reaches another model's field.
4. **Text encoder, CPU or GPU, selectable** (owner decision of 2026-10-29): two Triton models with
   identical tensor names, one `tensorrt_plan` on `KIND_GPU`, one `onnxruntime_onnx` on one
   `KIND_CPU` instance, mirroring `pe_text_encoder` (`models/pe_text_encoder/config.pbtxt`). Never
   an in-process session per uvicorn worker (the 84 GiB lesson, `pe_encoder.py:29-36`). Not in
   Triton's default load list while the feature is off.
5. **Image path, one decode.** A model-specific resample from the GPU-resident frame beside
   `_embed` in `models/gpu_image_prep/1/model.py:286`, as the eval plan's part 9 proposes; when the
   feature is off nothing is constructed, loaded or reserved.
6. **Backfill and text embedding are jobs, not requests.** After B2, the backfill of the new field
   is part of the `op.jobs.reprocess` family (listed in `docs/design/program/03_b2_overview.md`,
   job-family table), under the project's lease, decoding each image once in the same pass that
   may refresh PE vectors. Embedding OCR lines or VLM captions for placement A or B happens where
   those texts are written (the ingest OCR stage, the labeling writer), never as a synchronous step
   in a search request. Query encoding in the request is fine (as PE text today).
7. **Settings and docs.** The setting that turns it on is a deployment or project setting, not a
   new `OP_*` variable unless the docs check and `env.template` are updated in the same change;
   ATTRIBUTION row for the model licence; CHANGELOG entry; contracts regenerated for any wire
   field.

Generic names everywhere in tracked code (no model or vendor name in plumbing), per the naming
scan and the eval plan's own rule.

---

## 9. Pre-registration and the write-up

- **R0 (docs only, before any score):** the registration section of
  `docs/research/embedding_eval_results.md`: arms with revisions, datasets with pins, scenarios,
  the bars of 6.5 and 7.6 and the gates of section 3, the stage order and the stop rule of
  section 10. Committed as `docs(research): register the embedding evaluation bars (#347)`. A bar
  changed after a read needs a new registration and a statement of what was read (the WB2 rule,
  `wb2_standard_benchmarks.md:3-6`).
- **Results** appended below, generated from the result JSONs by `report.py`.
- **Per placement**, one verdict line: Go, Conditional or No-go, with the deciding numbers.
- **Final recommendation** names exactly one of: replaces nothing (no-go); complements X (new
  field Y, query path Z); never "replaces PE-Core" (owner decision).
- **Pointers:** append the results summary to section 10 of `embedding_model_eval_plan.md`
  (the table there is the agreed place), a row in `docs/research/README.md` and the research table
  of `docs/design/README.md`, the RQ-3b/E-4 row of `research_clustering_plan.md`, and comments on
  #347 and #365 with the verdict. Append to `retrieval_baseline.md` the PE whole-image row
  (append, never rewrite, as the research plan says for E-12).

---

## 10. Stop rules

| Point | Stop when | Then |
|---|---|---|
| Stage A | licence incompatible (A0) | measure only if the owner still wants numbers; nothing ships; say so |
| Stage A | A2 fails for bf16, or A5 cannot be made to pass | fix or stop; no scenario is read on a harness that fails A5 |
| After Stage B | A and B both no-go **and** the S4 screen loses to PE-Core-L and MobileCLIP2-S2 with no fusion gain | no-go for #347; close with the evidence; no Stage C remainder, no reference arms, no TensorRT |
| After Stage C | no placement Go or Conditional | same |
| Conditional only | | report; one follow-up experiment only if the owner asks for it |
| Stage D | no TensorRT tier passes within its time box | serve the component through Triton's onnxruntime backend (GPU or CPU) if it meets the speed and VRAM budget of 7.3; otherwise the placement waits and the note says why |
| Reference arms | time box reached | "not run" with the reason; never blocks the verdict |

---

## 11. Judgment calls

| ID | Question | Options | Recommendation and trade-off |
|---|---|---|---|
| J1 | Stage gating | (a) staged A, B, C, D with the stop rule; (b) the full S1 to S7 matrix and all arms at once, as the eval plan lays out | **(a)**. The owner's rule is to cut unproven research; text is where the prior is highest and it is nearly free (CPU). (b) spends about a GPU day and several agent days before the first decision |
| J2 | TensorRT timing | (a) only for a component that passed (8.1); (b) TensorRT for every encoder-style arm up front (owner comment of 2026-10-29, item 6) | **(a)**. An engine for an arm that loses is wasted work; the parity reference is PyTorch either way. Ask the owner because (b) is his stated wish |
| J3 | Reference arms | (a) 2B and 1B on the S4 screen, S2a and S5, 8B and omni one row each if time remains; (b) all four arms on every scenario | **(a)**. They bound the gain from size; they cannot ship in this issue |
| J4 | GPU and window | RW-2 of the research plan, after the v0.7.0 tag, on GPU 0 or GPU 2 agreed with the peer sessions (`docs/design/program/11_risks_cleanup_host.md` section 1); the eval plan says GPU 0 only and never GPU 2; WB2 ran on GPU 2 | owner's call; never GPU 1; research windows are outside the program's three GPU windows (`docs/design/program/00_index.md` 8.1). Stage A and B need no window (CPU plus minutes of GPU for A2) and may run earlier if the owner agrees |
| J5 | Model name in new tracked files | (a) generic file and arm names, the model named only in the existing allowlisted eval-plan note and in issue comments; (b) a new narrow allowlist entry for the results note | **(a)**, with the results note saying "the model of `embedding_model_eval_plan.md`". (b) is fine if the owner prefers a readable note; one line in the allowlist with a reason |
| J6 | Where shipping lands | (a) a new feature issue for an optional second field (8.2); (b) under #365 | **(a)**. #365 is the swap of the production encoder with re-index, re-cluster and re-train; an optional additional field touches none of the PE-derived artefacts. Both need per-field embedding state, so whichever lands first builds it and the other reuses it |
| J7 | Local-only rows | publish CUB and SOP aggregates as WB2 did, or keep them out | **publish aggregates**, as precedent; no images or per-image data |
| J8 | Caption VLM for S3 | the catalog's default entry, or `qwen3-vl-8b` | the default entry (what users have); one model only, recorded with revision and prompt |
| J9 | Bars | (a) the eval plan's 3.6 bars as written; (b) amended | **(a)** plus two additions registered in R0: gate A5 (harness parity) and the S4 screen used by the stop rule |
| J10 | S7 multilingual | skip, or run XM3600 for English plus two languages | **skip** unless Stage C runs and an hour remains; the owner said "only if not hard" and the product has no multilingual need today |

Open questions from the eval plan still stand and should be answered with J1 to J4: the NVIDIA
arm identity (answered by the owner's later comment: the 3.5 Embed 1B as the arm that could ship,
the omni model as measurement only), the VRAM budget (2 GiB text, 4 GiB vision proposed), and
keeping noisy-host runs labelled rather than dropped (yes).

---

## 12. Corrections to premises

1. **The eval plan's harness location.** `$HOME/eval_work/scripts/eg2/` (eval plan 3.1, 3.5)
   conflicts with the research rule that every result reproduces from a clean checkout. The
   code goes in `scripts/research/embedding_eval/`; data and vectors stay outside.
2. **The retrieval pin.** `research_clustering_plan.md` 4.1 names `coco_bench_2000.json` as "the
   retrieval pin". That manifest holds 721 train2017 and 1,279 val2017 images
   (`scripts/datasets/manifests/coco_bench_2000.json`); the measured retrieval baseline used 2,000
   val2017 images pinned in `docs/benchmarks/retrieval_baseline/sample_ids.json` (seed 20261028).
   Use the latter, or the comparison with the measured numbers does not hold.
3. **RQ-3b's bar does not decide this arm.** The owner made it complement-only; the per-placement
   bars decide (section 1, second bullet). E-4 still records its row.
4. **"#365 is blocked on #347."** With complement-only, #347 cannot produce a swap candidate for
   #365. A passing placement creates an optional second field (J6); #365's swap of PE-Core stays
   blocked until some candidate passes WB2's B1 bar, which none has
   (`wb2_standard_benchmarks.md` 9.3).
5. **The WB2 encoder registry is not reusable for this arm as is** (fp16 autocast at
   `fine_grained/models.py:67`, square inputs at `:64-71`).
6. **The eval plan says the PE text tower runs "in-process on CPU by default"** (eval plan 2.1).
   The code's `auto` backend tries Triton's `pe_text_encoder` first and falls back to torch
   (`pe_encoder.py:39-42`); the module's own first paragraph (`pe_encoder.py:9-12`) still says
   in-process by default. Either way the production text encode is a CPU onnxruntime model in
   Triton, and that is the latency reference for placement A's CPU option.
7. **The model name cannot appear in new tracked code** (naming scan, section 1). The eval plan
   says the same for plumbing; it applies to research scripts too, because the scan covers the
   whole tracked tree except allowlisted lines.
8. **CPU bf16.** The eval plan's CPU latency step passes no dtype (eval plan 3.5, step 6). On this
   host's AVX2-only CPU, the CPU arm must be fp32 (section 1).
9. **BM25.** The eval plan's lexical baseline is the `rank-bm25` package; the product's engine is
   OpenSearch, whose BM25 is the realistic alternative a user could get, so it replaces the
   package.
10. **Benchmarks that cannot rank.** CARS196 is at the ceiling for CLIP-type encoders (WB2 9.1);
    it is dropped from this evaluation.

---

## 13. File-by-file change list

### 13.1 Research code (`scripts/research/`)

| File | Change |
|---|---|
| `embedding_eval/` (new package, section 4.1) | arms, data, encode, scorers, storage, latency, gates, report, `measured_run.sh` |
| `embedding_eval_requirements.txt` (new) | research venv dependencies (4.1) |
| `fine_grained/models.py` | unchanged (the baselines are loaded through it; its fp16 autocast is the documented WB2 setting) |
| `scripts/datasets/manifests/oi_ln_5000.json` (new) | D2 pin, written by a fetcher flag or a small selection script (extend `scripts/datasets/fetch_openimages_bench.py` or `proxy_openimages.py`, whichever already reads Open Images validation; do not write a third fetcher) |
| `scripts/datasets/manifests/textcaps_textocr_<n>.json` (new) | D3 pin |

### 13.2 Tests (`tests/research/test_embedding_eval.py`, new)

| Test | Red before the code because |
|---|---|
| an arm declared bf16-only raises on a fp16 request | no arm registry exists; the WB2 wrapper autocasts every arm to fp16 |
| the query prefix reaches the tokenizer for `role="query"` and not for `role="none"` (with a fake tokenizer) | no prefix handling exists |
| Matryoshka slicing re-normalizes (norm 1.0 after a cut) and refuses a cut above the native dim | |
| scoring refuses a query and a gallery with different arm, dim or cut metadata | |
| `encode.py` opens each image file once for three arms (a counting opener) | |
| `score_xmodal` on a constructed 3-image, 15-caption case gives hand-computed recall@1/5 and MRR (reusing `single_relevant_metrics`) | |
| no tracked file under `scripts/research/embedding_eval/` contains the model family's name (the naming scan run on the package) | the scan itself is the test; a literal id in a default would fail it |
| the lexical arm reproduces the product's token match on a fixed example (calls the domain function) | |

### 13.3 Docs and data

| File | Change |
|---|---|
| `docs/research/embedding_eval_results.md` (new) | R0 registration, results, verdicts (section 9) |
| `docs/benchmarks/embedding_eval/` (new) | backing JSONs and `SHA256SUMS` |
| `docs/research/embedding_model_eval_plan.md` | section 10 table filled from the results (append) |
| `docs/research/retrieval_baseline.md` | appended PE whole-image row |
| `docs/research/README.md`, `docs/design/README.md` | rows |
| `docs/design/research_clustering_plan.md` | one line in the RQ-3b / E-4 row with the verdict |

No product code in #347. Stage D's shipping work is a separate issue (J6).

---

## 14. How we prove it: verification plan

### 14.1 Static and unit (per lane: targeted tests only)

```bash
.venv/bin/python -m pytest tests/research/test_embedding_eval.py tests/test_retrieval_eval_metrics.py -q --no-cov
.venv/bin/pre-commit run --files <changed files>
```

Each test in 13.2 is written first and seen to fail.

### 14.2 Harness parity (gate A5, real data)

Red looks like: offline MobileCLIP recall@10 on D1 outside [0.872, 0.901]. Typical causes, in the
order to check: EXIF orientation not applied; resize or centre-crop rule differing from the
export; normalization constants (MobileCLIP2 uses mean 0 / std 1 in the repo's export,
`fine_grained/models.py:41-44`); the ranking including the query or ties broken differently. A5
must pass before any new-arm number is read.

### 14.3 Precision and loader sanity (gates A2 to A4)

Red looks like: NaN, cosine below 0.999 for bf16, or an MTEB task more than 1 point off the card.
The fp16 run is expected red; if it is green, the note says so and the vendor warning is reported
as not reproduced on our data (it does not change the bf16 choice).

### 14.4 Scenario results

Every result JSON carries: git sha, arm, model revision, dtype, device, dataset pin hash, seed,
`nvidia-smi` snapshot, load average at start and end, wall time, peak VRAM; latency rows also
the thread count and core list. A run starting at a 1-minute load above 16 is labelled noisy and
repeated before it can decide anything (eval plan section 6, rule 6).

### 14.5 Decode-once and space hygiene (stage D only, if shipped)

The checks of eval plan part 9: one nvJPEG decode per image with the embedder on, parity of the
frame-derived input against the vendor processor (cosine at least 0.99), marginal ingest cost.

### 14.6 One sweep

After the lanes merge on their integration branch: the offline suite with `-n 8`,
`pre-commit run --all-files`, the docs check. No program live run or program GPU window.

---

## 15. Edge cases and traps (the failure register)

1. **fp16 by accident.** The WB2 wrapper, a `torch.autocast` left on, ONNX export from an fp16
   model, or `trt_utils.bake_fp16_onnx` (the repo's TensorRT 11 helper converts to fp16 by design)
   all turn a bf16 arm into an fp16 arm without an error. The arm's dtype check and the A2 gate
   catch the first two; stage D must not call the fp16 baker for this model.
2. **Missing task prefixes.** Vectors look fine and quality drops a few points. The A4 anchor and
   the prefix test catch it. Images take no prefix.
3. **Different prefixes for query and document.** Using the document prefix for queries (or the
   same prefix for both) is a silent mismatch; the role argument is explicit everywhere.
4. **Matryoshka without re-normalization.** Cosine on a sliced, un-normalized vector is not
   cosine; FAISS inner product then ranks wrongly but plausibly.
5. **Query and gallery at different cuts.** A 256-d query against 768-d documents fails loudly in
   FAISS only if shapes differ; with one sliced and one padded, it fails silently. The metadata
   check refuses it.
6. **The 270M and 440M text vectors differ.** If A3 fails and the text-only vectors are used to
   query `emb_tv` image vectors, S4 is meaningless. Use the 440M text path for cross-modal.
7. **Image token budget changes the result and the cost.** Report both per budget; never compare
   `emb_tv` at 560 tokens with PE-Core at 336 px on quality alone.
8. **EXIF orientation.** WB2 decodes upright (`embed_run.py` uses `ImageOps`); a harness that
   does not will pass A5 on COCO (few rotated images) and fail on Open Images.
9. **Caption leakage into the repo.** Captions and OCR strings stay in the scratch directory; only
   ids and aggregates are committed. The retrieval note's rule (`retrieval_baseline.md:14-15`).
10. **Train/test overlap in S3.** The caption used as the index document and the caption used as
    the query must be different captions of the image, chosen by seed; using the same one gives
    recall near 1.
11. **TextCaps and TextOCR on Open Images train.** D3 is disjoint from D2 (validation) by
    construction; keep it so, or S2 and S3 share images.
12. **Pre-training contamination.** COCO and Open Images may be in every arm's training data;
    the comparison is relative between arms, not absolute. State it.
13. **BM25 analyzer mismatch.** OpenSearch's standard analyzer differs from the product's token
    normalization (`src/domain/item_text.py`); report both lexical arms, do not pick the weaker
    one as "the" baseline.
14. **Ties in exact kNN.** Equal scores (duplicate images) make rank order depend on index order;
    fix tie-breaking by id, as WB2 does (ties by index).
15. **CPU latency on a loaded host.** The host has 48 threads and peers; pin cores, record load,
    repeat 3 times, label noisy runs.
16. **vLLM on sm_86.** A vLLM build older than the fix noted in the eval plan (startup failure on
    GPUs with 99 KiB shared memory) fails on these cards; re-check the issue tracker and pin a
    version that includes the fix, or use `transformers`.
17. **Throwaway OpenSearch under the default project name** would act on the live stack. Always
    `-p op-rs-emb` (`docs/research/CLAUDE.md`).
18. **GPU 1.** Never. GPU 0 or 2 only as agreed (J4); serialize under the shared lock; refuse to
    start when a foreign process holds the card (eval plan section 6, rule 3).
19. **Raw data loss.** Do not bulk-delete `artifacts_local/embedding_eval/` or earlier bake-off and
    WB2 outputs; the backing JSONs in `docs/benchmarks/` are never pruned.
20. **A per-worker CPU model in production.** If shipped, a text model loaded once per uvicorn
    worker multiplies memory by the worker count (`pe_encoder.py:29-36`); it must be one Triton
    CPU instance.
21. **Optional means absent.** With the feature off, no plan is loaded, no VRAM reserved, no field
    created and no response field emitted; a test with the feature off asserts the response and
    the mapping are unchanged.
22. **Licence drift.** The card's licence line and the linked terms may change between the read
    and the release; the ATTRIBUTION row cites the revision read.

---

## 16. Suggested phasing

| Phase | What | Needs | Exit |
|---|---|---|---|
| P0 | R0 registration commit | docs | committed before any score |
| P1 | Harness code and tests (13.1, 13.2); D1 and D3 pins | CPU | lane tests green |
| P2 | Stage A gates A0 to A5 | owner reads the licence; minutes of GPU for A2/A3; CPU for A4/A5 | all gates recorded; A5 green |
| P3 | Stage B (S2, S3 human-caption pass, S6 text, CPU latency) | CPU; the S3 VLM pass waits for a window | Stage B verdicts |
| P4 | S4 screen on D1 | about 15 min of GPU | stop rule read |
| P5 | Stage C in RW-2 (S3 VLM captions, S4 full, S5, S1, S6 image, reference arms) | research window, agreed GPU | Stage C verdicts |
| P6 | Stage D for a passing component | TensorRT time boxes | runtime chosen |
| P7 | Write-up, pointers, comments on #347 and #365; close #347; open the follow-up issue if Go (J6) | | docs check green |

P0 to P4 can run before the research windows if the owner allows the few minutes of GPU for A2,
A3 and the S4 screen; P5 onward waits for RW-2.

---

## 17. Relation to other issues

- **#365** (encoder swap): unaffected by a #347 no-go; a #347 Go creates the optional-field path
  that #365 would also need (per-field embedding state), see J6.
- **#341 harness / #346 tracker**: S5 uses the cluster-eval harness; one row in #346.
- **#406** (judge calibration): any visual check of S4 or S5 results uses the judge only on kinds
  that passed there; faces are not part of #347.
- **#435 / #436 / RQ-10**: the PE whole-image text-to-image row of S4 is the measurement RQ-10
  asks for; share vectors with E-12.
- **#40, #259, #362** (one decode): the shipping design's image path follows them.
