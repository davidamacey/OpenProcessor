# EmbeddingGemma 2 evaluation plan (issue #347)

Status: research and plan, written 2026-10-28, extended 2026-10-29 with the owner decisions of that
date (optional off-by-default model, CPU or GPU text encoder, Qwen3-VL-Embedding and an NVIDIA
embedding model as reference arms, TensorRT for every arm, one decode per image, shared-GPU
scheduling, a metrics schema and a results ledger). **No measurement of any model was made for this
note** (no GPU, no Docker, no weights downloaded). Everything about the models is read from public
pages; each fact is tagged **[vendor]** (the maker's own model card, blog or guide, not independently
reproduced), **[community]** (a third party measured it), **[repo]** (read from this repository) or
**[inference]** (my reasoning, not a source). Part 3 is the quality benchmark an agent can execute
later; parts 5 to 9 are the TensorRT track, scheduling protocol, metrics schema, ledger and the
single-decode integration; part 10 is where results get recorded.

The Google model's public name appears in this file by design (it is a research note about that
model); generic plumbing must not hardcode it. The other arms are named only in this note too. See the entry in
`scripts/codegen/naming_leak_allowlist.txt`.

## 1. What the model is

### 1.1 Facts

| Item | Value | Source |
|---|---|---|
| Name, maker, date | EmbeddingGemma 2, Google DeepMind, announced 2026-10-06 | [vendor] [blog](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) |
| Built on | The Gemma 4 architecture; bidirectional text encoder with vision and audio towers projecting into it | [vendor] [transformers docs](https://huggingface.co/docs/transformers/en/model_doc/embedding_gemma2) |
| Total parameters | 740M | [vendor] [HF card](https://huggingface.co/google/embeddinggemma-2) |
| Text encoder | 270M (130M transformer + 140M embedder). 24 layers, model dim 512, hidden 2048, 4 heads, local:global attention 5:1, sliding window 1024, mean pooling, 512 to 768 projection, vocabulary 262,144 | [vendor] HF card |
| Vision encoder | 170M, optional | [vendor] |
| Audio encoder | 300M, optional (not relevant here) | [vendor] |
| Loadable subsets | text 270M; text+image 440M (`config_kwargs={"audio_config": None}`); text+audio 570M; full 740M | [vendor] HF card, [developer guide](https://developers.googleblog.com/en/embeddinggemma-2-the-developer-guide/) |
| Output | 768-d native, one shared space for all modalities. The guide says the 270M text-only and full-model setups share one vector space | [vendor] developer guide |
| Matryoshka | 768 / 512 / 256 / 128; re-normalize after slicing; same dimension for queries and documents | [vendor] HF card |
| Context | 8,192 tokens shared by all inputs. Image: 280 tokens default, configurable 70 to 1120 (`max_soft_tokens` in {70,140,280,560,1120}) | [vendor] HF card, transformers docs |
| Task prefixes (text only) | Query `task: search result \| query: {q}`; document `title: none \| text: {t}`; also clustering, classification and `sentence similarity` prefixes. Images get no prefix. Omitting prefixes lowers quality | [vendor] HF card |
| Precision | **bfloat16 or float32. Not float16**: activations exceed the fp16 range and can give NaN or silently degraded vectors | [vendor] HF card |
| License | Apache 2.0 on the card, linked to the Gemma 4 license text; the Gemma Prohibited Use Policy applies. Not gated per the card | [vendor] HF card. Read the license text before any shipping decision |
| Training data cutoff | January 2025; web (100+ languages), code, images, video, audio, paired data | [vendor] |
| Paper | arXiv 2610.37364, "EmbeddingGemma 2: Unified and Efficient Multimodal Representations", submitted 2026-10-27. The abstract gives no numbers and no named rivals; the full text was not read for this note | [vendor] [arXiv](https://arxiv.org/abs/2610.37364) |
| Weights | Safetensors, BF16, on Hugging Face and Kaggle | [vendor] |

### 1.2 Reported quality (all [vendor], 768-d, from the model card)

| Benchmark | Score | EmbeddingGemma 1 |
|---|---|---|
| MTEB multilingual v2 (mean task) | 61.36 | 61.15 |
| MTEB code v1 (nDCG@10) | 78.68 | 68.76 |
| MIEB lite | 65.67 (HF card); a secondary summary says 64.64 | n/a |
| MMEB v2 Image (Hit@1) | 57.28 | n/a |
| MMEB v2 VisDoc (nDCG@5) | 67.84 | n/a |
| MMEB v2 Video (Hit@1) | 50.67 | n/a |
| MSEB retrieval (MRR@10) / MAEB | 69.54 / 49.87 (HF card); the secondary summary says MAEB 49.39 | n/a |

Two of the numbers differ between the HF card and a secondary write-up
([unite.ai](https://www.unite.ai/deepmind-debuts-embeddinggemma-2-mapping-five-modalities-into-one-space/)).
The Google-hosted card carries a deprecation notice pointing to another site, so card versions may
differ. Treat the exact values as unsettled and re-read the card at execution time.

Matryoshka truncation, [vendor] (MTEB multilingual v2 / MIEB lite / MMEB v2 overall):

| Dim | MTEB multi v2 | MIEB lite | MMEB v2 overall |
|---|---|---|---|
| 768 | 61.36 | 65.67 | 59.01 |
| 512 | 61.17 | 65.34 | 58.38 |
| 256 | 60.41 | 64.16 | 56.24 |
| 128 | 57.89 | 59.97 | 45.65 |

Google says 256-d is close to lossless and 128-d is suitable for text only; the developer guide
says about 75 percent of full quality for image, video and speech retrieval at 128-d, and
recommends 768 or 512 for multimodal retrieval.

What is **not** published by the vendor: COCO or Flickr30k image-text recall, any head-to-head with
named models (CLIP, SigLIP, PE-Core, MobileCLIP), and any latency or throughput on a server GPU.
The only latency-like figure is on-device RAM on a phone (about 191 MB text-only, about 567 MB full,
quantized) [vendor, blog]. Google's claims of "leading sub-1B multimodal embedders" and beating
models "twice its size" name no model, so they cannot be checked.

### 1.3 Independent evidence found so far (thin)

* **[community]** [kotadymo/vision-language-benchmarks](https://github.com/kotadymo/vision-language-benchmarks):
  one author's repo, RTX 4070 Ti SUPER, FP32, batch 16. For this model it reports CIFAR-10 top-1 94.66
  (18 prompt templates) or 92.58 (native prompt), CIFAR-100 76.08 or 62.84, and XM3600 text-to-image
  R@1 of 58.61 (English) and 80.14 (Japanese) on 3,600 candidates, ahead of SigLIP 2 Base/16 224 in
  the same repo on the XM3600 rows (English 51.47) but behind it on CIFAR-100 under the native prompt
  (73.25). The repo states that COCO and Flickr30k are not run, that Japanese results are exploratory,
  that no BF16 run or confidence intervals are published, and that it reports no throughput or VRAM for
  this model. Useful as a sanity anchor only.
* No independent COCO/Flickr30k result, no MTEB/MIEB leaderboard reproduction and no latency/VRAM
  study was found as of 2026-10-28. The secondary press and blog pages found repeat Google's figures
  (one states plainly that it measured nothing itself).

### 1.4 Runtime and export feasibility

| Runtime | Status | Source |
|---|---|---|
| transformers | `EmbeddingGemma2Model`, `EmbeddingGemma2Processor` (transformers 5.x docs; version to be pinned at execution) | [vendor] |
| sentence-transformers | `SentenceTransformer("google/embeddinggemma-2")`; guide says v6.1.0 or newer; dict inputs `{"text"|"image"|"audio"|"video": ...}` | [vendor] |
| vLLM | Supported, with open or just-fixed problems: [#60847](https://github.com/vllm-project/vllm/issues/60847) startup failure on GPUs with 99 KiB shared memory (sm_86, which is the RTX A6000 class here, and sm_120); closed 2026-10-27 by PR #60942, so a vLLM build older than that fails on our GPUs. [#60977](https://github.com/vllm-project/vllm/issues/60977) rejects `dimensions=768` (reported, cause unconfirmed by maintainers). [#63945](https://github.com/vllm-project/vllm/issues/63945) fails with the FLEX_ATTENTION backend. Whether vLLM serves image input for this model was **not confirmed** in any page read | [community] issue trackers |
| llama.cpp, Ollama, MLX, SGLang, LM Studio | Listed by Google. Ollama on Linux failed to pull the model at the time of one report because it needs MLX ([issue #18825](https://github.com/ollama/ollama/issues/18825)). Not relevant to the Triton stack | [vendor] / [community] |
| ONNX | Community export exists: [onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX), three separate graphs (text, vision, audio). Published sizes: fp32 text 1,085 MB and vision 671 MB; fp16 543 and 336 MB; q8 314 and 195 MB; q4 175 and 109 MB. The page's worst-case cosine to fp32 is 0.9998 (fp16), 0.9997 (q8), 0.975 (q4; 0.988 text). The fp16 ONNX result conflicts in spirit with the vendor's "no float16" warning and must be re-checked on our data. [Synaptics](https://huggingface.co/Synaptics/Google-EmbeddingGemma-2) rebuilt NPU binaries from these ONNX graphs, which shows the graphs are exportable | [community] |
| TensorRT | **No source found** (Google's lists omit it; no TensorRT engine or export recipe found). Feasibility is plausible via the ONNX graphs but unproven: unusual parts are the local/global attention mix, per-layer embeddings, a 262k vocabulary table, and the fp16 range problem (a TensorRT engine would need to be bf16 or fp32, or keep the overflowing ops in fp32). Treat as an experiment, not an assumption | inference |
| LiteRT | [litert-community](https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm) on-device build; not relevant to the server | [vendor] |

Consequence for this repo: the production serving path is Triton (TensorRT plans, plus an
onnxruntime CPU model for the PE text tower [repo: `models/pe_text_encoder/config.pbtxt`]). For an
evaluation the simplest serving shape is PyTorch bf16 in a throwaway venv. If the model is adopted, the
likely first production shape is an ONNX text graph on onnxruntime (CPU or CUDA, like `pe_text_encoder`)
rather than a TensorRT plan; a TensorRT plan only after a measured need.

## 2. Where it could fit in OpenProcessor, and where it cannot

### 2.1 What the repo has today [repo]

* Image side: `pe_image_encoder` (PE-Core-L14-336, TensorRT, input 3x336x336, 1024-d, L2-normalized)
  and `mobileclip2_s2_image_encoder` (256x256, 512-d). Item vectors are stored as `pe_embedding`
  (1024-d k-NN field, `encoder_embedding_dim`); regions have per-box vectors in
  `region_box_embeddings`.
* Text side for semantic search: `src/clients/pe_encoder.py` encodes queries with the PE-Core text
  tower (context length 32, 1024-d), in-process on CPU by default or via the Triton onnxruntime model
  `pe_text_encoder`; search (`src/services/curation/semantic_search.py`) runs k-NN of that vector
  against `pe_embedding`. MobileCLIP2-S2 has a text encoder (77 tokens, 512-d, TensorRT) in the model
  repository. The PE text tower runs on CPU on purpose, so search keeps working with the GPU and Triton
  down.
* OCR text: OCR lines are stored as `item_text_lines` plus normalized keyword tokens
  `item_text_tokens`, matched by `GET .../crops?item_text=<q>` (`src/domain/item_text.py`). Matching is
  keyword-token based, not semantic.
* Measured search baseline [repo: `docs/research/v070_measurement_log.md`, section D]: text search on a
  91,184-item project, p50 about 35 to 69 ms over a persistent connection, p99 under 110 ms idle; the
  text encode through Triton costs about 32 ms on a cache miss; k-NN itself is under 13 ms.
* PE image batch throughput [repo: `docs/research/pe_encoder_batching_benchmark.md`]: about 30 crops/s
  at batch 1 and about 90 crops/s at batch 4 (TensorRT FP16, one instance); this is the production
  speed reference the candidate must be judged against, not a like-for-like number for PyTorch.
* Clustering: one graph per embedding model and task, never mixed spaces
  (`docs/research/cluster_types.md`, epic children #341 to #345). Cluster-quality harness:
  `scripts/bench/cluster_eval.py`, `cluster_eval_metrics.py`, `cluster_eval_set.py`
  (`docs/research/cluster_eval_baseline.md`).

### 2.2 Candidate placements

| # | Placement | Why it could help | What it needs |
|---|---|---|---|
| A | Text-only 270M for **item text**: embed OCR lines (`item_text_lines`) and VLM labels or notes; search by meaning next to the existing keyword match | Today OCR search is lexical. A 128 to 256-d vector is small. A new, separate field, no change to existing vectors | New k-NN field per project (new dimension), index mapping migration, a text-embedding worker, a query path. All additive |
| B | Text-only for **VLM caption or summary** embedding, searched by text query or by "similar caption" | Search by description without depending on the image tower; captions already exist after VLM labeling | Same as A, one more field |
| C | Text+vision 440M as a **second image view**: text-to-image and image-to-image beside PE-Core | One model gives text-to-image, image-to-image, image-to-caption in one space; Matryoshka sizes | A second image vector per item (extra GPU encode per image at ingest, extra storage), a second query encoder. Only worth it if it beats PE-Core on our public data or complements it in fusion |
| D | **Clustering view** (an alternative graph, experiment only) | A different inductive bias may separate classes PE-Core merges | Offline only, through the #341 harness; its own graph, never merged with PE-Core vectors |

### 2.3 Where it cannot replace anything

* **The image encoder is not swappable in place.** `pe_embedding` is the basis of item k-NN, IVF
  residual clustering, region-box clustering, and anything trained on those vectors. Replacing the image
  encoder means re-index, re-cluster and re-train (the dual-field migration rule; see #65 and the
  `embedding_state` handling). A candidate therefore enters as a **new field next to** `pe_embedding`,
  with its own index mapping, its own query encoder, and its own embedding-state tracking. Search by
  field is explicit; there is no silent fallback that mixes spaces.
* **Never mix spaces in one graph or one k-NN query.** A text query encoded by one model must only be
  scored against vectors written by the same model (same dimension, same Matryoshka cut, same
  normalization). A query from this model against `pe_embedding` is meaningless, and so is the reverse.
  Any fusion (reciprocal rank or score blending of two separate searches) happens after retrieval, not
  in a shared vector space.
* **Per-box region vectors** (`region_box_embeddings`) are PE-Core vectors of crops; a second model there
  would be a second per-box field and a second clustering run. Out of scope until the item-level result
  is positive.
* **MobileCLIP** stays a lighter option; this note does not propose to remove or replace it.
* The hardened or CPU-only deployment shape (search works with GPU and Triton down) is only preserved by
  a candidate that runs acceptably on CPU. This is checked in the benchmark (scenario S1).

## 3. Benchmark design

Goal: reproducible numbers per model with latency, VRAM and vector size, and one recommendation that
names the single place (if any) where the model replaces or complements a current encoder. The
owner's scenarios (issue comment) are S1 to S6.

### 3.1 Environment (throwaway, outside the repo)

* Host work only. No Docker, no production stack, **GPU 0 only, never GPU 1**: always
  `CUDA_VISIBLE_DEVICES=0`. Do not run while another run holds GPU 0; check `nvidia-smi` first and
  record GPU, driver, CUDA, torch, transformers, sentence-transformers versions and the model revision
  hash in the results file.
* Venv outside the repo, never on the host Python:

```bash
python3 -m venv $HOME/eval_work/envs/eg2/venv          # outside the checkout
$HOME/eval_work/envs/eg2/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
$HOME/eval_work/envs/eg2/venv/bin/pip install "transformers>=5" "sentence-transformers>=6.1" pillow numpy \
    pandas pycocotools faiss-cpu rank-bm25 onnxruntime-gpu pynvml datasets
# baseline model code: PE-Core via perception_models and MobileCLIP via open_clip / ml-mobileclip,
# pinned to the same checkpoints the repo exports from (see export/export_pe_text_encoder.py and
# the export scripts for the image encoders). Do not use a different checkpoint than production.
```

* Work dir `$HOME/eval_work/data/eg2/` (datasets, vectors); scripts in
  `$HOME/eval_work/scripts/eg2/`. Keep both out of git. **Per-run result JSON, logs and engines go to the
  ledger (part 8), not the work dir.** Results are summarized back into this file (part 10).
* Every GPU command in this plan is wrapped as described in part 6 (`flock` on the GPU 0 lock plus
  host and GPU telemetry). The `CUDA_VISIBLE_DEVICES=0` rule below stands.
  The model download here is the explicit, later execution step, not part of this note.
* Precision: candidate in **bfloat16** (fp32 as the reference), never fp16. Baselines in the precision
  production uses (PE image TensorRT FP16 is the production reference; the eager PyTorch bf16 or fp16 run
  is the like-for-like comparison). State which in each row.
* Determinism: fixed seeds, same image decode (Pillow, RGB) for all models, same image list order,
  3 warm-up and at least 30 timed iterations per latency cell, report p50/p95/p99.

### 3.2 Models (exact arms)

| Arm | Text side | Image side | Dim |
|---|---|---|---|
| PE-Core (production baseline) | PE-Core-L14-336 text tower (context 32) | PE-Core-L14-336 image tower, 336 px | 1024 |
| MobileCLIP | MobileCLIP2-S2 text (77 tokens) | MobileCLIP2-S2 image, 256 px | 512 |
| EG2-T | EmbeddingGemma 2 text only (270M), query/document prefixes per card | n/a | 768, 512, 256, 128 |
| EG2-TV | EmbeddingGemma 2 text+vision (440M, `audio_config=None`), default 280 image tokens; also 140 and 560 for the budget sweep | same | 768, 512, 256, 128 |
| Lexical baselines | BM25 (rank-bm25) and the repo's current `item_text_tokens` token match | n/a | n/a |
| QW-2B (reference) | Qwen3-VL-Embedding-2B, text and image through one decoder (section 5.1) | same model | 2048 (Matryoshka 64 to 2048) |
| QW-8B (reference) | Qwen3-VL-Embedding-8B | same model | 4096 (Matryoshka 64 to 4096) |
| NV-OMNI (reference) | NVIDIA `omni-embed-nemotron-3b` (section 5.2) | same model | 2048 |
| NV-35 (optional reference) | NVIDIA Nemotron 3.5 Embed 1B (and 8B if it fits), the license-compatible NVIDIA alternative (section 5.2) | same model | 2048 (1B; truncate to 1024/512), 4096 (8B) |

Reference arms are measured on the same data and protocol as the others but are **not candidates to ship
unless their license and size pass the hard gates in 3.6**: they bound what a much larger model buys.
Size honesty: report quality per GB of VRAM and per ms next to raw quality.

Verify before running that the text-only 270M and the 440M load produce the same text vectors (the
developer guide claims one shared space); if not, record the difference and use the 440M text path for
cross-modal scenarios.

### 3.3 Datasets (public only; never `baseline-*` projects)

| Id | Data | Use | Notes |
|---|---|---|---|
| D1 | COCO captions, 2017 val (5,000 images, 25,000 captions) and the Karpathy 5k test split if the split file is obtainable | S1, S3, S4, S6 | annotations CC BY 4.0, images per Flickr terms; record the exact files and checksums |
| D2 | Open Images V7 validation images, a fixed random 5,000-image sample with Localized Narratives captions where available | S3, S4 (longer captions), generalization check | confirm license of each file at download time |
| D3 | A public scene-text set: TextOCR (built on Open Images) with TextCaps captions, or COCO-Text annotations on COCO images | S2 | verify license and that images overlap with D2 or D1; use the dataset's own OCR ground truth strings as the "OCR text", and additionally run the repo's OCR pipeline (PP-OCRv5) on a 1,000-image subset to include real recognition noise |
| D4 | The repo's own cluster-eval sample (200 COCO val images, 10 classes) via `scripts/bench/cluster_eval_set.py` | S5 | needs a stack, so a separate later step with GPU 0 and the ingest that already produced PE vectors; the candidate vectors are computed offline and scored with `cluster_eval.py` on the same labels |

Pool sizes: report on 5k and, for latency of search itself, on a 100k pool built by encoding D1 + D2 +
more COCO train images (vectors only), so k-NN behaviour at realistic size is visible. Do not claim
1M results.

### 3.4 Scenarios, steps, metrics

Common retrieval metrics: recall@1/5/10, nDCG@10, MRR@10 (and mean rank for caption-to-image), with
bootstrap 95 percent confidence intervals (1,000 resamples over queries). Common systems metrics per
arm: model load time, peak VRAM (`torch.cuda.max_memory_allocated` plus `nvidia-smi` reserved),
vector bytes per item (dim x 4 for fp32, and the same at fp16/int8 storage), encode latency p50/p95/p99
and throughput at batch 1, 4, 8, 16, 32, with pre-processing time reported separately from model time.

| Id | Owner scenario | Task and pairing | Compared arms | Extra metrics |
|---|---|---|---|---|
| S1 | Speed | Encode latency and throughput. Text: 1,000 real queries (COCO captions, 5 to 20 tokens) and 1,000 long strings (D2 narratives, up to 512 tokens) on **GPU 0 and on CPU** (CPU threads fixed to 8, pinned; the PE text tower runs on CPU in production). Image: 1,000 D1 images, batch sweep. Also image budget sweep 70/140/280/560 tokens for EG2-TV | PE text/image, MobileCLIP, EG2-T, EG2-TV, QW-2B/8B, NV-OMNI, NV-35 (PyTorch bf16 here; TensorRT rows come from part 5) | per-component latency, GPU util, CPU % ; compare to production TensorRT references in sections 2.1 only as context, not as equal-footing rows |
| S2 | OCR text | (a) Meaning search over OCR strings: query = TextCaps caption (or a question) of an image, document = concatenated OCR strings of that image, relevant = same image. (b) Noisy-string match: query = deterministic perturbation (case, dropped characters, a word reordering; seed fixed) of an OCR line, relevant = its source line. (c) Exact-match comparison: the repo's token match and BM25 on the same pairs | EG2-T at 768/256/128, BM25, token match, PE text tower (as a naive text-to-text arm) | hit rate where exact match returns nothing; recall@k per query type |
| S3 | Image summaries | Caption-embedding search vs direct image search. Index A: EG2-T embeddings of one held-out caption per image. Index B: image embeddings (PE, MobileCLIP, EG2-TV). Query: a different human caption of the same image. Second pass with VLM-generated captions from the repo's catalog VLM on GPU 0 (owner-approved; run under the lock, before the latency runs, part 6); report which VLM, revision and prompt | EG2-T caption index, EG2-TV image index, PE image index, MobileCLIP image index, BM25 on captions | recall@k, and overlap of top-10 sets between A and B (does captioning add recall) |
| S4 | Cross-modal | Text to image, image to image, image to caption. Text-to-image and image-to-text use the standard protocol (the 5 captions of an image are its relevants). Image-to-image has no ground truth: use COCO instance-category set overlap (Jaccard at least 0.5) as relevance and state that it is a proxy in the one space | PE-Core, MobileCLIP, EG2-TV at all four dims | text-to-image recall@1/5/10, image-to-text recall@1/5/10, mean rank; fusion row: reciprocal-rank fusion of PE and EG2-TV result lists (post-retrieval) |
| S5 | Clustering | Cluster the D4 crops with the model's vectors using the same method and settings as the PE run (FAISS k-means IVF K and AHC distance 0.25 as in the current pipeline, via the #341 harness), score against the same ground-truth labels | PE-Core vectors vs EG2-TV vectors | the harness's own metrics and strata (low resolution, rare class, similar classes, mixed scene); agreement (adjusted Rand index) between the two partitions; which strata each wins |
| S6 | Storage | Matryoshka cut: every retrieval metric above re-scored at 768/512/256/128 with re-normalization, queries and documents at the same dim; also int8 scalar-quantized vectors at 256-d. Index size: vector bytes per 100k items and the OpenSearch on-disk size of a k-NN index for 100k x {1024, 768, 256, 128} (the repo's field mapping, Lucene engine) | EG2-T, EG2-TV; PE as the size reference | metric loss versus 768 in absolute points and percent, bytes per item, index size, k-NN query p50/p95/p99 at each dim on the 100k pool |
| S7 | Multilingual (optional, owner: only if not hard) | Text-to-image recall@k on public XM3600 captions per language (part 5.6) | EG2-TV, QW, NV-35 vs the PE text tower | per-language recall@k, languages supported |

### 3.5 Step-by-step commands (for the executing agent)

All paths are examples; the executing agent creates the scripts under `$HOME/eval_work/scripts/eg2/`.

```bash
# 0. preflight
nvidia-smi --query-gpu=index,name,memory.used --format=csv     # GPU 0 must be idle
export CUDA_VISIBLE_DEVICES=0 HF_HOME=$HOME/eval_work/data/eg2/hf
E=$HOME/eval_work/envs/eg2/venv/bin/python

# 1. data (record URLs, sizes, sha256 in $HOME/eval_work/data/eg2/MANIFEST.txt)
#    COCO 2017 val images + captions_val2017.json; Open Images V7 val sample + localized narratives;
#    TextOCR + TextCaps annotations. Public sources only.

# 2. model files (the first and only large download; check disk, then)
$E -c "from huggingface_hub import snapshot_download as s; \
  print(s('google/embeddinggemma-2'))"      # record the revision hash

# 3. precision/sanity gate before any benchmark
#    bf16 vs fp32 cosine on 200 texts and 200 images (must be >= 0.999), and a deliberate fp16 run
#    that is expected to show NaNs or drift (document it; confirms the card's warning)
$E $HOME/eval_work/scripts/eg2/sanity.py --out $D/sanity.json
#    text-only 270M vs 440M text vectors identical? (cosine on 1,000 texts)

# 4. encode once, store vectors (float32) per arm and dim: $D/vec/{arm}/{dataset}_{text|image}.npy
$E $HOME/eval_work/scripts/eg2/encode.py --arm eg2-tv --dataset coco_val --dtype bf16 --batch 16

# 5. retrieval metrics from stored vectors (brute-force cosine with numpy/faiss, exact)
$E $HOME/eval_work/scripts/eg2/score.py --scenario S4 --arms pe,mobileclip,eg2-tv --dims 768,512,256,128

# 6. latency/VRAM: separate process per arm and batch size, fresh CUDA context
$E $HOME/eval_work/scripts/eg2/latency.py --arm eg2-t --device cuda --batches 1,4,8,16,32 --iters 30 --warmup 3
$E $HOME/eval_work/scripts/eg2/latency.py --arm eg2-t --device cpu --threads 8

# 7. S5 clustering needs the repo harness with PE vectors already ingested for the D4 sample:
.venv/bin/python scripts/bench/cluster_eval_set.py build ...   # see docs/research/cluster_eval_baseline.md
#    then compute EG2-TV vectors for the same crops offline, cluster with the same method,
#    write a partition.jsonl, and run scripts/bench/cluster_eval.py score on both partitions.

# 8. 100k k-NN timing: load vectors into a throwaway OpenSearch (own compose project name with -p, own
#    ports, not the live stack), one index per dim, same mapping as src/clients/curation_opensearch/base.py
```

Write results to `$HOME/eval_work/data/eg2/results/*.json` and a summary table into part 5. Do not
bulk-delete any earlier outputs.

### 3.6 Pass/fail criteria and the go/no-go rubric

Rules before looking at numbers: a candidate must beat the baseline on the **same data, same
protocol, with non-overlapping 95 percent bootstrap intervals** to count as a win; a difference inside
the interval is a tie. Report PE-Core numbers measured here, not literature values.

| Placement | Go (adopt as a new field / feature) | Conditional (keep researching) | No-go |
|---|---|---|---|
| A, OCR/item text (EG2-T) | S2(a) recall@10 and S2(b) recall@1 clearly above BM25 + token match **or** the union with them adds at least 5 recall points at 256-d; text encode p95 on CPU at most 50 ms for a 20-token query, or on GPU with batching at most the PE text cost | Better only at 768-d, or only for noisy strings | No gain over lexical, or CPU p95 over 150 ms |
| B, caption/summary (EG2-T) | S3 caption index recall@10 at least equal to direct image search **and** top-10 overlap shows complementary hits (union recall up by at least 5 points) | Equal but fully overlapping | Captions index worse than image search and not complementary |
| C, text-to-image / image-to-image (EG2-TV) | S4 text-to-image recall@5 not below PE-Core (tie or win) at 256-d or larger **and** image-to-image not below PE-Core, **and** ingest cost (extra GPU ms per image at batch 4, plus VRAM) fits inside the measured ingest headroom (`v070_measurement_log.md` records the GPU 35 percent idle in one ingest run) | Win only in fusion with PE-Core (adds at least 3 points recall@10 via rank fusion) | Below PE-Core on text-to-image and no fusion gain |
| D, clustering view | S5 partition scores at least equal on the harness, and complementary (wins in at least one stratum PE loses, ARI below 0.8) | Equal everywhere | Worse in every stratum |
| Storage | 256-d loses at most 3 percent relative on S4/S2/S3 metrics; 128-d loses at most 10 percent for text-only scenarios (otherwise recommend 256-d minimum) | | 256-d loses over 5 percent |
| Hard gates (any failure = no-go for that placement) | bf16 vs fp32 cosine at least 0.999 on real data; no NaN in 100k encodes; license reviewed and compatible with the repo's AGPL licensing (the linked Gemma 4 license text and Prohibited Use Policy read in full); a serving path that runs on Triton or onnxruntime without an unmaintained fork; peak VRAM for EG2-TV at batch 16 fits next to the current Triton models on one card | | |

The final recommendation names exactly one of: replaces nothing (no-go), complements X (new field Y
with query path Z), or replaces a component only if the dual-field migration is separately approved.

## 4. Risks and open questions

Risks
1. **fp16 breaks the model** (vendor warning). Our production TensorRT builds are FP16; a TensorRT
   path needs bf16 or per-layer fp32 and may lose the speed advantage. The community fp16 ONNX similarity
   figure (0.9998) disagrees with the warning in spirit; test, do not assume either.
2. **No TensorRT precedent** and exotic blocks (per-layer embeddings, 5:1 local/global attention, 262k
   vocabulary table) make export risky; vLLM has had new startup bugs on sm_86 (our GPU class), fixed
   2026-10-27 only.
3. **Vendor-only quality numbers**; no COCO/Flickr30k results, no named rivals, small and discordant
   independent evidence (one community repo, with caveats of its own). Prefix handling, image token
   budget and resize behavior can swing results; the benchmark must pin them and log them.
4. **Image preprocessing mismatch**: PE-Core uses 336 px square; this model uses a variable token budget
   (default 280 tokens). Comparisons at different effective resolutions are not equal-cost; record both
   quality and cost per image.
5. **Ingest cost**: a second image tower per item adds GPU time and a second vector (for example 256 x 4
   bytes = 1 KB versus 4 KB for the 1024-d PE vector); at high ingest rates this competes with the
   detector and PE encoders.
6. **Space hygiene**: any adoption creates a second embedding space. Needs per-field embedding state,
   a query path that names its field, and tests that fail if a vector from one model reaches another
   model's field (class and field identity by name, not index).
7. **License**: Apache 2.0 headline but tied to the Gemma 4 license text and Prohibited Use Policy;
   confirm by reading the full text. It has no safety tuning (card), which matters little for retrieval
   but should be stated in user docs if shipped.
8. **Fast-moving ecosystem**: bug reports are days old. Pin versions and re-check issue status at
   execution.

Open questions for the owner

Decided on 2026-10-29 (issue #347 comment): complement only, optional and off by default, never
replacing PE-Core; text encoder CPU or GPU selectable; multilingual optional (S7); the local catalog VLM
may caption on GPU 0; Qwen3-VL-Embedding and an NVIDIA embedding model are reference arms; TensorRT for
all arms. Still open:
1. Which NVIDIA model is meant (section 5.2)? The plan assumes `omni-embed-nemotron-3b` as the literal
   "omni embedding" release, plus Nemotron 3.5 Embed as an optional arm because the first is research-only.
2. The `omni-embed-nemotron-3b` license is research and development only; it can be measured but never
   shipped in this AGPL repo. Is a measurement-only arm acceptable?
3. How much agent time may one TensorRT conversion take before it is recorded as blocked (proposal in 5.5:
   one working day per arm and tier)?
4. Is a VRAM budget for the optional embedder wanted (proposal in 5.5: at most 2 GiB text, 4 GiB vision)?
5. May `ledger` raw dmon and host files of runs made under a shared-host load be kept as-is (yes by default,
   labeled `noisy`)?

## 5. Reference arms and the TensorRT build-and-benchmark track

Owner decision: TensorRT engines for all arms as the production target, built so the models run on the
GPUs as fast as possible. The PyTorch bf16 runs of part 3 are the quality and parity reference; the
TensorRT track here produces the speed numbers that count. A TensorRT result never replaces the parity
gate: an engine is only benchmarked after it passes it.

### 5.1 Qwen3-VL-Embedding (reference arm)

| Item | Value | Source |
|---|---|---|
| Official sizes | **2B** (28 layers, 2048-d, Matryoshka 64 to 2048) and **8B** (36 layers, 4096-d, Matryoshka 64 to 4096); 32K context; paper also describes a reranker series (not used) | [vendor] [GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding), [2B card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), [paper arXiv 2601.04720](https://arxiv.org/abs/2601.04720) |
| License | Apache-2.0 (repo page, 2B card, 8B card metadata). The repo lists one license and no per-model text, so read each card's license file at download time | [vendor] |
| Inputs | Text, images, screenshots, video; instruction-aware (default instruction "Represent the user's input"); 30+ languages | [vendor] |
| Pooling | Last-token (EOS hidden state of the last layer) | [vendor] GitHub |
| Image token settings | `min_pixels=4096`, `max_pixels=1843200` (1280x1440), default `max_length=8192` | [vendor] GitHub |
| Reported quality | MMEB-V2 overall 77.8 (8B, the vendor's claim of first place at release) and 73.2 (2B); MMTEB mean (task) 63.87 (2B) | [vendor] |
| Runtimes named | transformers, sentence-transformers, vLLM 0.14.0 or newer, SGLang. **TensorRT is not mentioned.** FlashAttention-2 and bfloat16 recommended; the card examples also load float16 | [vendor] |
| In this repo | Not related to the catalog's generative `qwen3-vl-4b` (a VLM served by vLLM); the embedder is a different checkpoint | [repo] `examples/vlm/catalog.tsv` |
| Fit on GPU 0 (RTX A6000, 48 GB) | 2B bf16 about 4 GB of weights; 8B bf16 about 16 GB of weights; both fit alone. Estimates, to be measured | [inference] |

Quality numbers above are vendor claims. The benchmark measures its own.

### 5.2 NVIDIA embedding model: what is meant

"NVIDIA's omni embedding model" most plausibly means **`nvidia/omni-embed-nemotron-3b`** (Omni-Embed-Nemotron),
the only NVIDIA embedding release with "omni" in its name. No other release by that name was found.
Related NVIDIA multimodal embedders exist and are listed so the owner can redirect the arm.

| Model | Facts | Source |
|---|---|---|
| **`nvidia/omni-embed-nemotron-3b`** (the match) | Released 2025-10-01 (paper arXiv 2510.03458, 2025-10-03). Built on the Qwen2.5-Omni-3B "Thinker"; text, image, audio, video, alone or combined, one shared space; bi-encoder. The name says 3B; the card says **4.7B parameters** (HF metadata 5B, BF16). **2048-d**, 32,768-token context. Image token settings not stated on the card (video bounds are: `min_pixels` 32*14*14, `max_pixels` 64*28*28, 2 fps). Needs transformers 4.51.3 or the sentence-transformers loader. **License: research and development only.** The card names the "NVIDIA OneWay Noncommercial License" together with the Qwen Research License, and elsewhere cites other NVIDIA licenses, so the card is internally inconsistent. Runtimes listed on the card: "TensorRT, Triton" (software integration field), hardware A100 and H100 on Linux. **No engine, export recipe, NIM or Triton model was found.** | [vendor] [HF card](https://huggingface.co/nvidia/omni-embed-nemotron-3b), [paper](https://arxiv.org/abs/2510.03458) |
| `nvidia/llama-nemotron-embed-vl-1b-v2` | Released 2025-12-18. About 1.7B parameters (Llama 3.2 1B plus a SigLIP 2 400M image encoder), 2048-d, image or text or both; 256 tokens per image tile, up to 6 tiles plus a thumbnail (about 1,792 visual tokens), context 10,240. BF16. License: NVIDIA Open Model License plus the Llama 3.2 Community License. Runtimes: transformers 4.56 or newer, vLLM 0.17.0 or newer; "TensorRT, Triton" are listed as software integration; a NeMo Retriever NIM was listed as upcoming. Aimed at document-page retrieval | [vendor] [HF card](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) |
| `nvidia/Nemotron-3.5-Embed-1B-BF16` and `-8B-BF16` (plus a `1B-NVFP4` variant) | Released 2026-10-20. Multimodal (text query, document as text, image or both), 34 text languages and 18 image languages evaluated. 1B: about 1.55B parameters, 2048-d (truncate to 1024 or 512 and re-normalize), image longest edge 1,120 px default, up to 1,540; 8B: 8.38B parameters, 4096-d. Context 32,768. **License: OpenMDW-1.1**, stated usable commercially and non-commercially (Apache-2.0 Ministral base). Runtimes on the 1B card: transformers 5.20, sentence-transformers, vLLM 0.32; TensorRT is **not** mentioned | [vendor] [1B card](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16), [8B card](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16), [NVIDIA blog](https://huggingface.co/blog/nvidia/nemotron-retriever-world-class-agentic-retrieval) |
| NIM container `nemotron-embed` | NeMo Retriever multimodal embedding NIM for the 3.5 family: 1B in BF16 and NVFP4, 8B in BF16. NVIDIA proprietary license (NVIDIA Software License Agreement and product-specific terms). A closed container, so it is a service to call, not a TensorRT plan this repo can load into its own Triton | [vendor] [NGC](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/nemotron-embed/-) |

Answers to the owner's question: exact name `omni-embed-nemotron-3b`; one size (the card's 4.7B);
research-only license; **no TensorRT or NIM path is documented for it** (the card lists TensorRT and Triton as
integration targets only). A TensorRT path for it would be our own conversion of the Qwen2.5-Omni Thinker
trunk (a Qwen2.5-style decoder with a ViT image tower), so it shares the Qwen3-VL tier ladder in 5.4. Because
the license forbids shipping, NV-OMNI is a measurement arm only; NV-35 1B is the arm to carry forward if an
NVIDIA model is wanted for shipping. FP8 and NVFP4 engines of the 3.5 family are not testable on our
cards (5.3).

### 5.3 Hardware and software facts that constrain the track

| Fact | Value | Source |
|---|---|---|
| GPUs | GPU 0 and GPU 2: RTX A6000 (sm_86, 48 GB); GPU 1: RTX 3080 Ti (sm_86). Driver 615.71.10 (read on 2026-10-28) | [repo] host `nvidia-smi` |
| TensorRT version used in production | `tensorrt-cu13==11.1.0.106` pinned in `requirements.txt`, asserted against the Triton 26.06 image in `Dockerfile.triton` (the TensorRT version build argument). `dali_gpu_decode_lessons.md` describes the stock 26.08 image with TensorRT 11.2.1; engines must be built by the TensorRT that the serving Triton loads, so build in the same image the serving stack uses and record `trtexec --version` | [repo] |
| Strong typing | TensorRT 11 builds are strongly typed: no `--fp16`, `--bf16`, `--int8`, `--fp8` builder flags; the precision comes from the ONNX graph's tensor types. The repo's `export/trt_utils.bake_fp16_onnx` therefore converts the ONNX to FP16 before `trtexec` | [repo] `export/build_pe_trt.sh`, [vendor] [TensorRT 10 to 11 trtexec migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html) |
| BF16 | Supported on Ampere and newer in TensorRT; with strong typing, give an ONNX whose tensors are bfloat16 (exported from a bf16 PyTorch model, or converted with a ModelOpt precision-conversion tool) | [vendor] TensorRT docs, [inference] for the conversion route |
| FP8 | Needs Ada (sm_89) or newer. **Not testable on any card here (all sm_86).** Record FP8 cells as `skipped: hardware`, not failed | [vendor] [NVIDIA blog](https://developer.nvidia.com/blog/model-quantization-turn-fp8-checkpoints-into-high-performance-inference-engines-with-nvidia-tensorrt/) |
| NVFP4 | Blackwell only; same treatment as FP8 | [vendor] NVIDIA model cards |
| INT8 | Works on sm_86 with Q/DQ nodes (ModelOpt PTQ then ONNX, then `trtexec`) | [vendor] NVIDIA blog above |
| Why fp16 breaks Gemma-family models | The residual stream grows past the fp16 maximum of 65,504; reports show NaN from the last layers in fp16 ONNX exports, and that forcing only the layer norms to fp32 is probably not enough because the residual adds overflow too. bf16 or fp32 avoids it | [community] [ORT #26732](https://github.com/microsoft/onnxruntime/issues/26732), [transformers PR 37226](https://github.com/huggingface/transformers/pull/37226) |

### 5.4 Per-model TensorRT path (tiers, stop at the first that passes)

Every arm uses the same ladder. Record the tier reached and, for a failure, the exact error and the ONNX
op or layer in `notes` (a blocked tier is a result, not a skip).

* **Tier 1: ONNX then `trtexec`.** One ONNX per component (text, vision), then one plan each.
* **Tier 2: split the model.** Vision tower plan, decoder trunk plan, and a thin host or Triton BLS step
  for the glue (patch merge, position ids, pooling). Used for the Qwen-family arms, whose decoder takes a
  variable image-token count and 3D rotary positions.
* **Tier 3: a TensorRT-based LLM runtime.** TensorRT-LLM or NVIDIA's ModelOpt export path, only if its
  docs list the architecture. I did not verify that any of TensorRT-LLM's support matrices list these
  embedding checkpoints; check before spending time. If no tier passes within the time box (5.5) the arm
  is served for the benchmark with PyTorch bf16 or vLLM and flagged `no_trt`.

| Arm | Component engines | Precisions to build and test | Specific risks |
|---|---|---|---|
| PE-Core text/image, MobileCLIP | Existing builds (`export/build_pe_trt.sh`, `models/pe_image_encoder`): rebuild into the ledger with the same commands for a like-for-like baseline | FP16 (production), BF16, FP32, optional INT8 | None new; this is the control |
| EG2-T (text) | `eg2_text`: tokens in, 768-d out | FP32 (reference), **BF16**, FP16 (expected to fail, document), FP16 with FP32 islands for residual adds and norms (only if the tooling allows per-layer types), INT8 | Per-layer embeddings and a 262k vocabulary table (about 140M parameters): the embedding gather may dominate engine size; 5:1 local/global attention with a 1024 window; sequence-dependent masks. The community fp16 ONNX claims cosine 0.9998 against the vendor warning: test, do not assume |
| EG2-TV (vision) | `eg2_vision` for the 170M tower; the 440M configuration reuses `eg2_text` | same as above, per token budget 70/140/280/560/1120 | Patch-embedding input layout (read from the community vision ONNX: three separate graphs exist), variable token count |
| QW-2B, QW-8B | Tier 2: `qwen_vision` and `qwen_trunk` | BF16, FP16 (the card loads fp16), INT8 | Variable image resolution to variable token count, multimodal rotary positions, 8B engine memory (check against 48 GB); last-token pooling needs the sequence length per row |
| NV-OMNI | Tier 2 as Qwen (Qwen2.5-Omni Thinker) | BF16 | No vendor recipe; image path only (skip audio and video towers); research license, measurement only |
| NV-35 1B | Tier 2 | BF16; NVFP4 `skipped: hardware` | Vision encoder from a Ministral-based tower; 1,120 px longest edge means many tokens |

Build commands (a template; names of flags must be checked with `trtexec --help` in the pinned image
before the first run, since TensorRT 11 removed the precision flags and renamed some options). Run them
under the lock from part 6, in the Triton or SDK image of the serving stack:

```bash
# 0. versions and environment (record all in the run's metrics JSON)
trtexec --version; nvidia-smi --query-gpu=index,name,driver_version --format=csv

# 1. ONNX per component, one precision per file. Export from the PyTorch model loaded in that
#    dtype (bf16 or fp32), opset 18 or newer so layer norm maps to one fused layer.
$E $S/export_onnx.py --arm eg2-t --dtype bf16 --opset 18 --out $LEDGER/engines/eg2-t/bf16/model.onnx
sha256sum $LEDGER/engines/eg2-t/bf16/model.onnx > $LEDGER/engines/eg2-t/bf16/model.onnx.sha256

# 2. inspect and sanitize (constant folding) before building
polygraphy inspect model model.onnx --mode=basic
polygraphy surgeon sanitize model.onnx --fold-constants -o model.folded.onnx

# 3. parity of the ONNX against PyTorch fp32 first (ONNX Runtime on CPU, so GPU 0 is not needed)
polygraphy run model.folded.onnx --onnxrt --input-shapes input_ids:[8,32] attention_mask:[8,32]

# 4. build (dynamic shapes; precision comes from the ONNX types; the first build of a model uses
#    detailed profiling verbosity so layer precisions can be read back)
trtexec --onnx=model.folded.onnx --saveEngine=model.plan \
  --minShapes=input_ids:1x8,attention_mask:1x8 \
  --optShapes=input_ids:8x32,attention_mask:8x32 \
  --maxShapes=input_ids:32x512,attention_mask:32x512 \
  --memPoolSize=workspace:8192MiB --timingCacheFile=timing.cache \
  --profilingVerbosity=detailed --exportLayerInfo=layers.json \
  2>&1 | tee build.log                                    # build wall time = engine build time

# 5. engine parity: run the plan and the ONNX Runtime fp32 reference on the SAME real inputs
polygraphy run model.folded.onnx --trt --onnxrt --load-inputs real_inputs.json \
  --atol 1e-2 --rtol 1e-2   # plus the cosine gate in 5.5 on 1,000 real inputs, from our own script

# 6. speed: per batch, with the exact shapes the service sends
trtexec --loadEngine=model.plan --shapes=input_ids:8x32,attention_mask:8x32 \
  --warmUp=500 --iterations=200 --duration=10 --useCudaGraph --noDataTransfers \
  --dumpProfile --separateProfileRun --exportTimes=times.json
# repeat for batch 1,4,8,16,32 and for the text lengths 16 / 32 / 128 / 512 and image budgets

# 7. serving-level numbers through Triton (queueing, dynamic batching, HTTP/gRPC), with perf_analyzer
perf_analyzer -m eg2_text -b 1 --concurrency-range 1:32:4 --shape input_ids:32 --shape attention_mask:32
```

Dynamic shapes and batching:

* One optimization profile per workload, not one wide profile: text queries (sequence 8 to 64, opt 32 like the PE
  text tower) and text documents (up to 512, or 2048 for long captions if S3 needs it) are separate profiles of
  one plan, or separate plans when the fused kernels differ. Batch min 1, opt 8, max 32, matching
  `pe_image_encoder`.
* Image token budget is a shape: build one plan per budget in {70, 140, 280, 560, 1120} for EG2-TV (a
  fixed token count lets TensorRT pick better kernels; padding to the budget keeps the batcher simple), or
  one plan with a token-count axis if the profile is fast enough. Decide from the S1 sweep. For the Qwen
  and NVIDIA arms the equivalents are `max_pixels` and tile count; benchmark the vendor default and one
  smaller setting.
* Triton: `dynamic_batching` with `preferred_batch_size [4, 8, 16, 32]` and a short queue delay for text
  (2 ms like `pe_text_encoder`), 15 ms for images (like `pe_image_encoder`); `max_batch_size: 32`;
  `GPU_MIB_PER_INSTANCE` measured at max batch and recorded as in the existing configs.
* Integer inputs: build the plans with INT32 token ids (TensorRT narrows INT64 and warns) and have the
  client send INT32. The vector output is L2-normalized in the graph at full width; Matryoshka slicing and
  re-normalization stay on the client (one plan serves every dimension).

Triton repository layout, mirroring the existing models (`models/pe_image_encoder`, `models/pe_text_encoder`):

```
models/<embedder>_text_trt/   config.pbtxt  (platform: tensorrt_plan, KIND_GPU)    1/model.plan
models/<embedder>_text_ort/   config.pbtxt  (platform: onnxruntime_onnx, KIND_CPU) 1/model.onnx
models/<embedder>_image/      config.pbtxt  (platform: tensorrt_plan, KIND_GPU)    1/model.plan
```

The two text directories expose identical tensor names, dtypes and output dimension, so the client selects
CPU or GPU by model name (the owner's selectable text encoder). The CPU variant keeps search working with
the GPU or Triton down, as the PE text tower does. Both are not in Triton's default load list while the
feature is off. The tensor names are a contract with the client; write them into each config header as
the PE configs do. Naming in the real tree must be generic (no vendor name in plumbing); only this
note may carry model names.

### 5.5 Decision criteria for the TensorRT track

Evaluate in this order; the first failed gate stops that (arm, precision) pair.

1. **Builds and runs.** Engine builds without error; no NaN or Inf in 100,000 encodes of real inputs.
   Time box: one working day per arm and tier; after that, record `blocked` with the log and move down
   the ladder (the box is a proposal, open question 3).
2. **Parity with fp32 PyTorch.** On 1,000 real texts and 1,000 real images: mean cosine at least 0.999 and
   minimum at least 0.99. Then the retrieval metric (S4 text-to-image recall@5, S2 recall@10) within the
   95 percent bootstrap interval of the fp32 PyTorch value. Quantized (INT8) engines must additionally lose
   at most 1 point of recall@10 relative.
3. **Speed.** Among the precisions that pass 1 and 2, pick the one with the lowest p50 at opt batch.
   The engine is worth shipping over ONNX Runtime or PyTorch bf16 only if it is at least 1.5 times faster
   (p50, same batch, same host load class) or halves VRAM; otherwise the simpler runtime wins for an
   optional model. Production references for context: PE image TensorRT FP16 about 90 crops/s at batch 4;
   PE text encode through Triton about 32 ms on a cache miss (section 2.1).
4. **Memory.** Peak VRAM at max batch 32, proposed budget for an optional embedder: at most 2 GiB (text) and
   4 GiB (vision), so it fits beside the detector and PE encoders. Reference arms (QW-8B, NV-OMNI) are
   exempt and reported only.
5. **Build cost.** Engine build wall time and plan size are recorded; a build over 30 minutes or a plan over
   2 GiB for the small arms is flagged (not a fail).
6. **CPU/GPU text.** The `_ort` CPU variant must reach p95 at most 50 ms for a 20-token query at 8
   pinned threads (go) or at most 150 ms (conditional), matching 3.6; the GPU variant must be at least
   2 times faster at batch 32 to justify a TensorRT plan for text.
7. **License and ship rules** are unchanged from 3.6; they bound what ships, not what is measured.

Output of the track: for each (arm, component) one row in the speed table (part 9) per precision and batch,
and one line stating the shipped runtime (TensorRT plan, ONNX Runtime, or none).

### 5.6 Optional multilingual check (S7)

Not required. Cheap if the arm is already encoded: text-to-image recall@k on the public XM3600
multilingual captions for the languages the arm advertises (EG2: 100+, Qwen3-VL-Embedding: 30+, Nemotron
3.5: 34 text and 18 image), against the PE-Core text tower, which is, I believe, English-only
([inference], verify in the PE repo before claiming it). Confirm the dataset license at download. A
language that the production use does not need is dropped from the table, not argued.

## 6. GPU and host scheduling protocol

The host is shared: other agents and other users run jobs, GPU 2 hosts a live test stack, GPU 1 belongs to
other users, and the CPU load average was 8 to 33 during earlier measurement sessions
(`v070_measurement_log.md`). Rules:

1. **GPU 0 only.** Every GPU command runs with `CUDA_VISIBLE_DEVICES=0`. Never use GPU 1 (other users) or GPU
   2 (live test stack). Do not start or restart the live stack from this work.
2. **Every GPU-0 job runs under the lock**, so jobs from different agents never overlap:

```bash
flock /mnt/nvm/scratch/locks/gpu0.lock bash $S/measured_run.sh <run_id> <command...>
```

   The lock directory exists (`/mnt/nvm/scratch/locks`). Hold the lock per job, not per session: no
   interactive shells, no downloads, no waiting for input while holding it. Add `-w <seconds>` to give up
   cleanly rather than queue forever, and log the wait time in the run's `notes`. This covers engine builds
   (a build times kernels and is sensitive to a busy GPU), the VLM caption pass, latency runs and encodes.
   CPU-only runs (ONNX Runtime CPU latency, scoring, k-NN) need no GPU lock but still record host load.
3. **One job on GPU 0 at a time means VRAM is yours, so measure it.** Before each run assert no foreign
   process on GPU 0: `nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv` must be empty at
   start (the run fails closed otherwise).
4. **Caption pass before benchmarks, never concurrent.** The catalog VLM (`examples/vlm/catalog.tsv`) serves
   S3 captions on GPU 0 in its own compose project (`-p <name>`, VLM GPU set to 0), writes the captions to
   the ledger, and is stopped before any latency run. Record the VLM id, revision and prompt. This runs
   under the same lock.
5. **Telemetry next to every run**, written into the run directory by `measured_run.sh`:
   * `dmon.csv`: `nvidia-smi dmon -i 0 -s pucvmet -d 1 -o T` for the whole run (power, clocks, utilization,
     memory, throttling); start before the command, stop after.
   * `host_start.txt` and `host_end.txt`: `/proc/loadavg`, `uptime`, `nproc`, `free -m`, top 10 processes by CPU,
     `nvidia-smi -i 0` full output, `nvidia-smi --query-compute-apps` list, `numactl --hardware` summary if
     present.
   * `host_load.csv`: `vmstat 1` for the run, so CPU contention during the run is visible.
6. **Noisy host label.** If the 1-minute load average at start exceeds 16 (the host has 48 threads; earlier
   sessions ran at 8 to 16 as normal), the run is stored but marked `host_load.noisy = true`. Latency and
   throughput runs are repeated 3 times and reported as median with range, as the measurement log does; a
   noisy run is never silently dropped and never used for a go/no-go number unless a repeat at lower load
   agrees.
7. **CPU latency runs** pin threads (`taskset` to a fixed core set, 8 threads) and record which cores.
8. **Thermals/power:** read throttle reasons in `dmon.csv`; a run with throttling is flagged.
9. **Ordering:** quality scoring (CPU, vectors on disk) is scheduled during GPU runs of other arms; GPU time
   is spent only on encodes, builds and latency.

`measured_run.sh` (to be written in `$S/`, outside the repo) does, in order: create
`$LEDGER/runs/<run_id>/`; refuse if `nvidia-smi` shows a foreign process on GPU 0; snapshot host state; start
`dmon` and `vmstat` in the background; run the command with `CUDA_VISIBLE_DEVICES=0`, tee stdout and stderr
to `stdout.log`, save the exit code; stop the samplers; snapshot again; write the command line to
`cmd.txt`. The benchmark script itself writes `metrics.json` (part 7) and fills the host fields from those
snapshots.

## 7. Metrics schema (one JSON per run)

Every run, including failed and blocked ones, writes `metrics.json` into its run directory. One run is one
(arm, component, runtime, precision, batch, input shape) cell with its repetitions. Fields marked
nullable are `null` when not applicable; no field is omitted.

```json
{
  "schema_version": 1,
  "run_id": "20261030T0915_eg2-t_trt-bf16_b8_s32",
  "started_utc": "2026-10-30T09:15:02Z",
  "status": "ok",
  "model": {"arm": "eg2-t", "name": "google/embeddinggemma-2", "revision": "<hf commit sha>",
            "component": "text"},
  "runtime": {"kind": "tensorrt", "version": "11.1.0.106", "container": "<image digest>",
              "torch": null, "transformers": null},
  "precision": "bf16",
  "engine": {"build_time_s": 412.7, "plan_bytes": 0, "plan_sha256": "", "onnx_sha256": "",
             "profile": {"min": "1x8", "opt": "8x32", "max": "32x512"},
             "build_cmd": "", "timing_cache": true},
  "input": {"batch": 8, "seq_len": 32, "image_px": null, "image_tokens": null, "dataset": "coco_val"},
  "latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0,
                 "n": 200, "warmup": 3, "includes": "model_only"},
  "preprocess_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
  "throughput": {"items_per_s": 0.0, "batches_per_s": 0.0},
  "vram_mib": {"peak_nvidia_smi": 0, "peak_torch_allocated": null, "engine_device_memory": 0},
  "vector": {"dim": 768, "matryoshka_dim": 768, "dtype": "float32", "bytes_per_item": 3072},
  "quality": {"dataset": "coco_val", "n_queries": 5000, "protocol": "S4",
              "metrics": {"text_to_image_recall@5": {"value": 0.0, "ci95": [0.0, 0.0]}},
              "parity_vs_fp32": {"cosine_mean": 0.0, "cosine_min": 0.0, "n": 1000, "nan_count": 0}},
  "hardware": {"gpu_index": 0, "gpu_name": "NVIDIA RTX A6000", "sm": "8.6", "driver": "615.71.10",
               "cuda": "13.x", "cpu": "2x E5-2680 v3", "cpu_threads_used": 8, "cpu_affinity": ""},
  "host_load": {"load1_start": 0.0, "load5_start": 0.0, "load1_end": 0.0, "nproc": 48,
                "noisy": false, "foreign_gpu_procs_at_start": 0, "gpu_lock_wait_s": 0.0},
  "artifacts": {"dir": "runs/<run_id>", "dmon": "dmon.csv", "host": ["host_start.txt", "host_end.txt"],
                "vmstat": "host_load.csv", "log": "stdout.log", "cmd": "cmd.txt"},
  "commit": {"repo": "<git rev-parse HEAD>", "scripts": "<git rev-parse HEAD of eval scripts, or sha256>"},
  "notes": ""
}
```

Rules: `status` is one of `ok`, `failed`, `blocked`, `skipped` (with `notes` giving the reason, for
example `skipped: hardware (fp8 needs sm_89)`); `latency_ms.includes` states whether the number is model
only, with preprocessing, or end to end through Triton; times are wall-clock with CUDA synchronized;
`vram_mib.peak_nvidia_smi` is the maximum of `nvidia-smi` used memory sampled during the run minus the
pre-run baseline; vectors are stored float32 and `bytes_per_item` is for the stored dtype. The quality
fields use the metrics of 3.4 and the bootstrap intervals of 3.6. The schema is a single file in the ledger
(`SCHEMA.json`, JSON Schema form) and a small validator script rejects any run directory whose
`metrics.json` does not conform, so no row reaches the write-up unvalidated.

## 8. Results ledger

Gitignored directory `artifacts_local/embedding_eval/` in the main checkout
(`/mnt/nvm/repos/OpenProcessor/artifacts_local/embedding_eval/`, already covered by `.gitignore`). All
worktrees and agents write to this one path (`LEDGER` below) so results do not scatter. Append only; never
bulk-delete (the same rule as the bake-off outputs).

```
artifacts_local/embedding_eval/
  SCHEMA.json                      JSON Schema for metrics.json (part 7)
  ledger.csv                       one row per run: run_id, arm, component, runtime, precision, batch, shape,
                                   status, p50, p95, p99, items_per_s, vram_mib, dim, headline_quality,
                                   noisy, commit  (generated from the metrics.json files by a script)
  runs/<run_id>/                   metrics.json, cmd.txt, stdout.log, dmon.csv, host_start.txt,
                                   host_end.txt, host_load.csv
  engines/<arm>/<precision>/       model.onnx(.sha256), model.plan, build.log, layers.json, timing.cache
  captions/<vlm_id>/               VLM captions for S3 (with the prompt and revision)
  manifests/                       dataset file lists with sha256 (public data only)
```

Large dataset downloads and encoded vectors stay in `$HOME/eval_work/data/eg2/` (not in the ledger);
`manifests/` points at them. `<run_id>` is `YYYYMMDDTHHMM_<arm>_<runtime>-<precision>_b<batch>_<what>`.

Write-up tables are generated from `ledger.csv`, never typed by hand. Templates (fill from the ledger,
one table per section 10 scenario; `n` = repetitions, a row marked `noisy` carries a flag):

Speed and cost, per (arm, component, runtime, precision):

| Arm | Component | Runtime | Precision | Batch | Shape | p50 ms | p95 ms | p99 ms | Items/s | VRAM MiB | Dim | Bytes/item | Engine build s | Plan MiB | Load1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PE-Core | image | tensorrt | fp16 | 4 | 336 px | | | | | | 1024 | 4096 | | | | not run |

Quality, per (scenario, arm, dimension):

| Scenario | Dataset, n | Arm | Runtime, precision | Dim | Metric | Value | 95% CI | Parity cosine (mean / min) | Status |
|---|---|---|---|---|---|---|---|---|---|
| S4 text-to-image | COCO val, 5,000 | PE-Core | tensorrt, fp16 | 1024 | recall@5 | | | | not run |

TensorRT track outcome, per arm:

| Arm | Tier reached | Precisions passing parity | Fastest passing | Speedup vs PyTorch bf16 | Peak VRAM MiB | Ship runtime | Blocker (if any) |
|---|---|---|---|---|---|---|---|
| EG2-T | | | | | | | not run |

## 9. One decode per image: plugging an embedder into `gpu_image_prep`

Rule (owner, #40/#259/#362): an image is decoded once per request and its GPU-resident frame is reused.
Facts from the repo: `models/gpu_image_prep/1/model.py` decodes the upload with nvJPEG into a CuPy frame
(`im.frame`), runs the detector on a letterbox resampled from that frame, builds the PE squares with
`kernels.resample_into(im.frame, ...)` (`_embed`), and calls `pe_image_encoder` through BLS with the
CuPy tensor, so the pixels never go back to the host. The stored-box mode (`_embed_stored`, the reprocess
pass) decodes once and embeds the whole frame and each stored box.

Proposal for an optional second embedding (design only, to build after a go decision):

1. **Add a model-specific resample step beside `_embed`, reading the same `im.frame`.** A function
   `_embed_alt(im)` that, for the configured embedder only, derives its input geometry from the frame with
   the existing `resample_into` kernel (bilinear, GPU): for EG2-TV an aspect-preserving resize to the token
   budget and patchify; for the Qwen and NVIDIA arms the vendor's resize rule (a multiple-of-28 resize
   within `max_pixels`, or tiles). Patchify, mean/std and dtype conversion happen in a GPU kernel or
   inside the engine (UINT8 in, the trick `pe_image_encoder` uses via `export/uint8_input.py`), so the
   input to the plan is a GPU tensor. No JPEG decode, no host round trip, no PIL.
2. **Call the embedder plan through the same `_bls` helper**, with the frame-derived tensor, and return a new
   optional output tensor (for example an alternate whole-frame embedding and, when requested, per-box
   rows) in the model config. The tensors are named generically (not by vendor), declared `optional`,
   and absent from the response when the feature is off.
3. **Off by default means no cost**: when no embedder is configured, `_embed_alt` is not constructed, no
   plan is loaded, no VRAM is reserved, and the response carries nothing new. A misconfigured or missing
   embedder fails closed for that field only (the PE path is untouched, and the field's embedding state
   reads as not computed rather than silently falling back to another model).
4. **Fusion at the plan level**: the new vector is a new field next to `pe_embedding` (section 2.3). The
   same frame-reuse applies to the reprocess embed pass (#259) so a backfill of the new field decodes each
   image once, in the same pass that may refresh PE vectors, never in a second pass over the library.
5. **Text queries** (the CPU or GPU text encoder of 5.4) need no image and are unrelated to the decode rule.

Checks that prove the rule, to run as part of the track (after a prototype exists):

* A counter of nvJPEG decodes per request; with the embedder on and a mixed set of 200 images, the count
  equals the number of images, and the host-to-device copy count for pixels equals one per image.
* Parity: embeddings from the frame-derived tensor versus the vendor reference processor (PIL path) on
  the same 1,000 images; mean cosine at least 0.99 (resize differences are expected; DALI and cv2 versus PIL
  differ the same way, see `dali_gpu_decode_lessons.md` section 3). Document which reference is chosen.
* Marginal cost: ingest images/s and p99 with the embedder on versus off at the same concurrency
  (`gpu_image_prep` two-instance configuration), plus the VRAM added per instance
  (`GPU_MIB_PER_INSTANCE`).

In the benchmark itself, quality numbers use the vendor processor (reference), the engine-only speed rows
use pre-prepared tensors, and one end-to-end row per shipped candidate goes through the prototype above, so
the decode-once path is measured and not assumed.

## 10. Results (to be filled in at execution)

One row per scenario S1 to S6 and per arm, with: date, hardware, library versions, model revision,
dataset ids, n queries, metric values with 95 percent intervals, latency p50/p95/p99, VRAM, vector
bytes. Generate the rows from the ledger (part 8). Then the go/no-go per placement from section 3.6 and the single recommendation.

| Scenario | Arm | Result | Status |
|---|---|---|---|
| S1 speed | | | not run |
| S2 OCR text | | | not run |
| S3 image summaries | | | not run |
| S4 cross-modal | | | not run |
| S5 clustering | | | not run |
| S6 storage | | | not run |
| S7 multilingual (optional) | | | not run |
| TensorRT track (part 5) | | | not run |

## Sources

* Google blog (2026-10-06): https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/
* Model card (HF): https://huggingface.co/google/embeddinggemma-2
* Model card (Google, deprecation notice): https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2
* Developer guide: https://developers.googleblog.com/en/embeddinggemma-2-the-developer-guide/
* transformers docs: https://huggingface.co/docs/transformers/en/model_doc/embedding_gemma2
* Paper: https://arxiv.org/abs/2610.37364
* Community ONNX: https://huggingface.co/onnx-community/embeddinggemma-2-ONNX
* NPU build from the ONNX graphs: https://huggingface.co/Synaptics/Google-EmbeddingGemma-2
* Community benchmark: https://github.com/kotadymo/vision-language-benchmarks
* vLLM issues: https://github.com/vllm-project/vllm/issues/60847, /60977, /63945
* Ollama issue: https://github.com/ollama/ollama/issues/18825
* Secondary summary (no measurements of its own): https://www.unite.ai/deepmind-debuts-embeddinggemma-2-mapping-five-modalities-into-one-space/
* Qwen3-VL-Embedding (GitHub, 2B and 8B cards, paper): https://github.com/QwenLM/Qwen3-VL-Embedding, https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B, https://arxiv.org/abs/2601.04720
* NVIDIA Omni-Embed-Nemotron-3B (card and paper): https://huggingface.co/nvidia/omni-embed-nemotron-3b, https://arxiv.org/abs/2510.03458
* NVIDIA llama-nemotron-embed-vl-1b-v2: https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2
* NVIDIA Nemotron 3.5 Embed: https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16, https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16, https://huggingface.co/blog/nvidia/nemotron-retriever-world-class-agentic-retrieval, NIM: https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/nemotron-embed/-
* TensorRT 10 to 11 trtexec migration (no precision flags, strong typing): https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html
* FP8 needs Ada or newer: https://developer.nvidia.com/blog/model-quantization-turn-fp8-checkpoints-into-high-performance-inference-engines-with-nvidia-tensorrt/
* Gemma-family fp16 overflow: https://github.com/microsoft/onnxruntime/issues/26732, https://github.com/huggingface/transformers/pull/37226
* Issue: https://github.com/davidamacey/OpenProcessor/issues/347
