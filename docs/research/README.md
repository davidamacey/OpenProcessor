# Research notes

Measured snapshots, literature reviews and evaluation plans behind OpenProcessor's design and
performance work. Design decisions live in [`docs/design/`](../design/README.md); this folder holds
the evidence. Conventions for adding or changing a note are in [`CLAUDE.md`](CLAUDE.md).

Notes are dated snapshots. Numbers drift as the code changes, so each note states where, when and
on what it was measured, and every result that matters has a reproduce command.

## Measurement logs (the source for any write-up)

| Note | What it holds |
|---|---|
| [v070_measurement_log.md](v070_measurement_log.md) | The running 0.7.0 log: cluster pass size ladder to 1M crops, ingest and decode-instance sweeps, NVMe versus HDD, contention, search latency, live checks of the embedding fixes. Each section has setup, method, results and reproduce commands. |
| [cluster_eval_baseline.md](cluster_eval_baseline.md) | First cluster-quality baseline from the evaluation harness on public COCO crops. |
| [cluster_proxy_datasets.md](cluster_proxy_datasets.md) | Public face and vehicle make/model proxy sets for cluster evaluation (LFW, Open Images, Commons set C): licences, counts, sizes, what was skipped and why, reproduce commands. |
| [labelled_benchmarks.md](labelled_benchmarks.md) | Established, author-labelled benchmarks to replace the weakly labelled Commons set C for judging embeddings and clustering: protocols, metrics, published numbers and their limits, licences and access (CARS196, VMMRdb, CUB-200, Stanford Online Products, LFW/CALFW/CPLFW), what was downloaded and what was skipped and why, and the benchmark to re-run first. |
| [installer_drive_log.md](installer_drive_log.md) | Installer drives (fresh, re-run, interrupt, control-plane-only, branch mode, upgrade and rollback, edge cases): results and the fixes they led to. |
| [cropwright_health_2026-10-29.md](cropwright_health_2026-10-29.md) | Cropwright frontend health pass against a live stack: checks, defects fixed, screenshots index. |
| [ingest_concurrency_tuning.md](ingest_concurrency_tuning.md) | Batch and concurrency sweep for bulk ingest. |
| [pe_encoder_batching_benchmark.md](pe_encoder_batching_benchmark.md) | Batched TensorRT PE-Core-L14-336: parity and throughput. |
| [detector_quantization_results.md](detector_quantization_results.md) | Detector precision, runtime and hardware matrix. |
| [triton_perf_crossvalidation.md](triton_perf_crossvalidation.md) | Batch size versus output payload on a live Triton. |
| [curation_scores.md](curation_scores.md) | Validation of the uniqueness, mistakenness and diversity scores. |
| [retrieval_baseline.md](retrieval_baseline.md) | First search-quality baseline on a test stack with public COCO images, captions and instance annotations. |
| [near_duplicate_detection.md](near_duplicate_detection.md) | Near-duplicate prototypes nd:P0..P2: registration, product-engine encoder parity, copy-level cosine cuts per space and linkage on 26,000 constructed copies, Copydays, exact and perceptual hash tiers (SHA-256 and PDQ cost). |
| [ingest_write_path_investigation.md](ingest_write_path_investigation.md) | Why bulk ingest looked write- and queue-bound: OpenSearch write-path measurements on throwaway containers, the fixes that followed and the remaining follow-ups. |
| [storage_sizing_and_ingest_baselines.md](storage_sizing_and_ingest_baselines.md) | Index size per item, memory and ingest baselines (faiss figures for old projects, Lucene for new ones), re-measured on the public set. |
| [v060_vector_db_evaluation_plan.md](v060_vector_db_evaluation_plan.md) | Tuned OpenSearch against Milvus, Qdrant, pgvector and LanceDB at 2k, 10k and 100k real vectors; results in `docs/PERFORMANCE.md`. |
| [v060_serving_backend_trade_study.md](v060_serving_backend_trade_study.md) | Serving backend study and experiment plan: rooflines, decision rules and the alternative-runtime head-to-head (#231) and MPS (#403) that still read them. |
| [audits/phase_a1_audit.md](audits/phase_a1_audit.md) | Independent audit of PR #361 (release/0.7.0) and the #357 and #362 fix branches. |

## Clustering

| Note | What it holds |
|---|---|
| [cluster_types.md](cluster_types.md) | The three kinds of clustering (whole image, detection crops, faces), their embedding spaces and code paths. |
| [cluster_config_baseline.md](cluster_config_baseline.md) | Configuration baseline on stored COCO vectors: raw-vector arms (IVF now, 2b k-means, average and complete AHC, Leiden CPM, ECG) on a held-out half, the detector-class prior, the row-order fix, and the diagnostics scripts that reproduce the consolidated note. |
| [gpu_pipeline_prototypes.md](gpu_pipeline_prototypes.md) | Registration and results of the GPU pipeline prototypes PR-0 (harness and the honest GPU number), PR-1 (decoder matrix) and PR-2 (resize and crop on CV-CUDA), short tiered runs on one RTX A6000. |
| [cluster_config_decision.md](cluster_config_decision.md) | Decision after the 3,000-item baseline and the W0c and W0d experiments (acceptance failed on A4 and A6 on a fresh set; a use-case bar for the auto-label loop registered before its data): average linkage inside the detector class with a per-key cut and per-key refusal, no global cut, no novelty state or cross-class suggestions; arm-by-arm kill rules, corrected acceptance bars fixed before a fresh public set, the next experiments (set P size and judge checks) and what WC can build now. |
| [cluster_config_experiments.md](cluster_config_experiments.md) | Results of the CPU experiments the decision note asks for (X0 to X7, X9) on the 3,000-item set: product bars for within-class average linkage, representation arms, novelty, cut transfer, late arrivals, scale and split spread, each as pass or fail with its limits. |
| [cluster_config_fresh_set.md](cluster_config_fresh_set.md) | Results of the W0d experiments: the corrected bars A1 to A7 applied once to a fresh public set, the label-free per-key cut rule, chaining by size on set P and split by descent, each as pass or fail with its limits. |
| [cluster_config_loop_cost.md](cluster_config_loop_cost.md) | Results of wave W0e: the auto-label loop's gate simulated on IVF, one cluster per detector key and within-key average linkage (Z1 on the two sets already read, Z2 on a fresh 8,000-image Open Images vehicle-heavy set with the registered bars U-a to U-d beside A1 to A7, Z3 judge-gated merge), with the registration committed before scoring. |
| [cluster_config_z4.md](cluster_config_z4.md) | Results of Z4 (wave W0f): detector-key roots with and without a label-free hold-back guard, within-key roots and IVF, each run through the auto-label loop's recursion on a fresh 8,000-image Open Images test-split set with the registered bars B-a to B-c. Key roots cost 0.29 decisions per item against IVF's 0.83 at 0.98 precision and the guard removes the minority exposure (9.05 to 0.25 wrong open-set labels per run) for 0.08 more; the open-set bar is unreadable because the test split holds only 758 open-set crops, so the registered rule leaves IVF in place. Registration committed before scoring. |
| [fine_grained_benchmark_plan.md](fine_grained_benchmark_plan.md) | Plan for wave WB, registered before any data: vehicle make and model grouping inside the car key. Covers the data (Commons set C, set C-large, distractors, set P sizes), the embeddings and preprocessing, the within-key methods, metrics including review and loop cost, pre-registered bars, and judge use. |
| [fine_grained_benchmark_results.md](fine_grained_benchmark_results.md) | Results of wave WB against the registered bars FG-0 to FG-7: five encoders and four preprocessing variants on Commons set C plus 10,358 car-key distractors, average linkage, k-reciprocal, DBA, Leiden and ECG inside the key, novelty queue, judge agreement at make and model level, encode cost. |
| [wb2_standard_benchmarks.md](wb2_standard_benchmarks.md) | Embedding and clustering re-run on author-labelled benchmarks (CARS196 zero-shot, CUB-200-2011, Stanford Online Products) in place of Commons set C: registered protocol, sanity check and bars, then results: no challenger beats PE-Core-L, the set C ranking does not reproduce, the registered sanity check was not met (CARS196 near ceiling). |
| [clustering_methods.md](clustering_methods.md) | UMAP, AHC, HDBSCAN and IVF for the residual pool; the IVF-then-refine design. |
| [gpu_clustering.md](gpu_clustering.md) | cuML UMAP backend detection and VRAM budget. |
| [clustering_research_consolidated.md](clustering_research_consolidated.md) | Consolidated clustering reference for whole images, crops, faces and fine-grained groups: internal notes, apps (Immich, PhotoPrism), literature, real-data geometry and whitening diagnostics, recommended stack per kind, real-data experiment plan. |
| [cross_domain_clustering_lessons.md](cross_domain_clustering_lessons.md) | What a sibling speaker-linking system learned and which techniques transfer to multi-view object and face clustering, with an ordered experiment plan for #342 to #345, #354, #372 and #286. |
| [leiden_refinement_prototype.md](leiden_refinement_prototype.md) | Graph-based global refinement (kNN graph, Leiden, split/merge validation) against the current pipeline; CPU and GPU (added when the prototype lands, epic #346). |
| [cluster_research_9oct.md](cluster_research_9oct.md) | Early survey of continuous, recursive, self-refining clustering research. |
| [spatial_datasets.md](spatial_datasets.md) | Public multi-view datasets for the spatial-memory work (#348), research snapshot of 2026-10-29. |
| [cluster_program_outcome.md](cluster_program_outcome.md) | Closing summary of the 0.7.0 cluster research program: what was tried, the decision per arm with numbers, what stays in production, the class-root lead (Z5), parked issues #342 to #346, data licences, reproduce pointers. |

## Serving, decode and models

| Note | What it holds |
|---|---|
| [triton_deep_research_2026-09.md](triton_deep_research_2026-09.md) | External references, production lessons and a performance architecture review for Triton-based vision serving (basis for #40). |
| [dali_gpu_decode_lessons.md](dali_gpu_decode_lessons.md) | GPU decode inside Triton: traps and lessons. |
| [vlm_serving_sizing.md](vlm_serving_sizing.md) | vLLM sizing, flag recon, profile sweep and VLM-versus-VLM comparison. |
| [visual_judge_e0.md](visual_judge_e0.md) | Contact-sheet visual judge harness, the sheet-file backend for hosted judges and the first agreement run (Haiku subagents). |
| [embedding_model_eval_plan.md](embedding_model_eval_plan.md) | Evaluation design for an optional multimodal embedding model and reference arms (LLM-based arms served, encoders on TensorRT), scheduling protocol and metrics schema (#347). |

## Long-term

| Note | What it holds |
|---|---|
| [OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md](OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md) | Discovering same-place photo groups, photogrammetry and a dated spatial timeline (issue #348, no milestone; the plan is `docs/design/v070_rebuild/09f_research_spatial_attributes.md`). Not implemented. |

## Related material elsewhere

- A domain-specific bake-off paper (curated, cluster-stratified training data and a detector comparison) is maintained privately outside this repository; its data-engine section corresponds to the ingest, clustering and split work measured in these notes. The public write-up of the systems work is planned from the measurement logs (see "Writing up" in [CLAUDE.md](CLAUDE.md)).

- Raw benchmark JSON used by the docs: [`docs/benchmarks/`](../benchmarks/).
- Performance numbers and acceptance bars: [`docs/PERFORMANCE.md`](../PERFORMANCE.md).
- Market and competitor analysis: [`docs/market/`](../market/).
- Bench and evaluation tooling: `scripts/bench/` (`cluster_eval*.py`, `cluster_ladder.py`, `scale_run.py`).
