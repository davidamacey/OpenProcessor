# EmbeddingGemma 2 evaluation plan (issue #347)

Status: research and plan, written 2026-10-28. **No measurement of the model was made for this note**
(no GPU, no Docker, no weights downloaded). Everything about the model is read from public pages;
each fact is tagged **[vendor]** (Google's own model card, blog or guide, not independently
reproduced), **[community]** (a third party measured it), or **[repo]** (read from this
repository). Part 3 is a plan an agent can execute later; part 5 is where results get recorded.

The model's public name appears in this file by design (it is a research note about that model);
generic plumbing must not hardcode it. See the entry in
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

* Work dir `$HOME/eval_work/data/eg2/` (datasets, vectors, results); scripts in
  `$HOME/eval_work/scripts/eg2/`. Keep both out of git. Results summarized back into this file (part 5).
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
| Optional reference | Qwen3-VL-Embedding-2B, only if time allows, because the community repo above also lists it; not a required baseline | | |

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
| S1 | Speed | Encode latency and throughput. Text: 1,000 real queries (COCO captions, 5 to 20 tokens) and 1,000 long strings (D2 narratives, up to 512 tokens) on **GPU 0 and on CPU** (CPU threads fixed to 8, pinned; the PE text tower runs on CPU in production). Image: 1,000 D1 images, batch sweep. Also image budget sweep 70/140/280/560 tokens for EG2-TV | PE text/image, MobileCLIP, EG2-T, EG2-TV | per-component latency, GPU util, CPU % ; compare to production TensorRT references in sections 2.1 only as context, not as equal-footing rows |
| S2 | OCR text | (a) Meaning search over OCR strings: query = TextCaps caption (or a question) of an image, document = concatenated OCR strings of that image, relevant = same image. (b) Noisy-string match: query = deterministic perturbation (case, dropped characters, a word reordering; seed fixed) of an OCR line, relevant = its source line. (c) Exact-match comparison: the repo's token match and BM25 on the same pairs | EG2-T at 768/256/128, BM25, token match, PE text tower (as a naive text-to-text arm) | hit rate where exact match returns nothing; recall@k per query type |
| S3 | Image summaries | Caption-embedding search vs direct image search. Index A: EG2-T embeddings of one held-out caption per image. Index B: image embeddings (PE, MobileCLIP, EG2-TV). Query: a different human caption of the same image. Optional second pass with VLM-generated captions from the repo's own VLM on GPU 0, only if that VLM fits; report which VLM and prompt | EG2-T caption index, EG2-TV image index, PE image index, MobileCLIP image index, BM25 on captions | recall@k, and overlap of top-10 sets between A and B (does captioning add recall) |
| S4 | Cross-modal | Text to image, image to image, image to caption. Text-to-image and image-to-text use the standard protocol (the 5 captions of an image are its relevants). Image-to-image has no ground truth: use COCO instance-category set overlap (Jaccard at least 0.5) as relevance and state that it is a proxy in the one space | PE-Core, MobileCLIP, EG2-TV at all four dims | text-to-image recall@1/5/10, image-to-text recall@1/5/10, mean rank; fusion row: reciprocal-rank fusion of PE and EG2-TV result lists (post-retrieval) |
| S5 | Clustering | Cluster the D4 crops with the model's vectors using the same method and settings as the PE run (FAISS k-means IVF K and AHC distance 0.25 as in the current pipeline, via the #341 harness), score against the same ground-truth labels | PE-Core vectors vs EG2-TV vectors | the harness's own metrics and strata (low resolution, rare class, similar classes, mixed scene); agreement (adjusted Rand index) between the two partitions; which strata each wins |
| S6 | Storage | Matryoshka cut: every retrieval metric above re-scored at 768/512/256/128 with re-normalization, queries and documents at the same dim; also int8 scalar-quantized vectors at 256-d. Index size: vector bytes per 100k items and the OpenSearch on-disk size of a k-NN index for 100k x {1024, 768, 256, 128} (the repo's field mapping, Lucene engine) | EG2-T, EG2-TV; PE as the size reference | metric loss versus 768 in absolute points and percent, bytes per item, index size, k-NN query p50/p95/p99 at each dim on the 100k pool |

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
1. Is the target a replacement or a complement? This plan assumes complement only (new field).
2. Is CPU-only text encoding a hard requirement for any new text encoder (the PE text tower is CPU by
   design)? It decides whether a 270M model is acceptable at all.
3. Which languages matter? The model advertises 100+; the public benchmark above is English COCO and
   Open Images, so multilingual value is not tested unless added (for example XM3600).
4. May the benchmark use the repo's local VLM (GPU 0) to produce captions for S3, or are human captions
   enough?
5. Should Qwen3-VL-Embedding-2B or SigLIP 2 be added as extra reference arms?
6. Is a production TensorRT/ONNX export of this model wanted if it wins, or is PyTorch/onnxruntime
   acceptable?

## 5. Results (to be filled in at execution)

One row per scenario S1 to S6 and per arm, with: date, hardware, library versions, model revision,
dataset ids, n queries, metric values with 95 percent intervals, latency p50/p95/p99, VRAM, vector
bytes. Then the go/no-go per placement from section 3.6 and the single recommendation.

| Scenario | Arm | Result | Status |
|---|---|---|---|
| S1 speed | | | not run |
| S2 OCR text | | | not run |
| S3 image summaries | | | not run |
| S4 cross-modal | | | not run |
| S5 clustering | | | not run |
| S6 storage | | | not run |

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
* Issue: https://github.com/davidamacey/OpenProcessor/issues/347
