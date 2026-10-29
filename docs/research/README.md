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
| [installer_drive_log.md](installer_drive_log.md) | Installer drives (fresh, re-run, interrupt, control-plane-only, branch mode, upgrade and rollback, edge cases): results and the fixes they led to. |
| [ingest_concurrency_tuning.md](ingest_concurrency_tuning.md) | Batch and concurrency sweep for bulk ingest. |
| [pe_encoder_batching_benchmark.md](pe_encoder_batching_benchmark.md) | Batched TensorRT PE-Core-L14-336: parity and throughput. |
| [detector_quantization_results.md](detector_quantization_results.md) | Detector precision, runtime and hardware matrix. |
| [triton_perf_crossvalidation.md](triton_perf_crossvalidation.md) | Batch size versus output payload on a live Triton. |
| [curation_scores.md](curation_scores.md) | Validation of the uniqueness, mistakenness and diversity scores. |

## Clustering

| Note | What it holds |
|---|---|
| [cluster_types.md](cluster_types.md) | The three kinds of clustering (whole image, detection crops, faces), their embedding spaces and code paths. |
| [cluster_config_baseline.md](cluster_config_baseline.md) | Configuration baseline on stored COCO vectors: raw-vector arms (IVF now, 2b k-means, average and complete AHC, Leiden CPM, ECG) on a held-out half, the detector-class prior, the row-order fix, and the diagnostics scripts that reproduce the consolidated note. |
| [cluster_config_decision.md](cluster_config_decision.md) | Decision after the 3,000-item baseline and the W0c and W0d experiments (acceptance failed on A4 and A6 on a fresh set; a use-case bar for the auto-label loop registered before its data): average linkage inside the detector class with a per-key cut and per-key refusal, no global cut, no novelty state or cross-class suggestions; arm-by-arm kill rules, corrected acceptance bars fixed before a fresh public set, the next experiments (set P size and judge checks) and what WC can build now. |
| [cluster_config_experiments.md](cluster_config_experiments.md) | Results of the CPU experiments the decision note asks for (X0 to X7, X9) on the 3,000-item set: product bars for within-class average linkage, representation arms, novelty, cut transfer, late arrivals, scale and split spread, each as pass or fail with its limits. |
| [cluster_config_fresh_set.md](cluster_config_fresh_set.md) | Results of the W0d experiments: the corrected bars A1 to A7 applied once to a fresh public set, the label-free per-key cut rule, chaining by size on set P and split by descent, each as pass or fail with its limits. |
| [cluster_config_loop_cost.md](cluster_config_loop_cost.md) | Results of wave W0e: the auto-label loop's gate simulated on IVF, one cluster per detector key and within-key average linkage (Z1 on the two sets already read, Z2 on a fresh 8,000-image Open Images vehicle-heavy set with the registered bars U-a to U-d beside A1 to A7, Z3 judge-gated merge), with the registration committed before scoring. |
| [fine_grained_benchmark_plan.md](fine_grained_benchmark_plan.md) | Plan for wave WB, registered before any data: vehicle make and model grouping inside the car key. Covers the data (Commons set C, set C-large, distractors, set P sizes), the embeddings and preprocessing, the within-key methods, metrics including review and loop cost, pre-registered bars, and judge use. |
| [clustering_methods.md](clustering_methods.md) | UMAP, AHC, HDBSCAN and IVF for the residual pool; the IVF-then-refine design. |
| [gpu_clustering.md](gpu_clustering.md) | cuML UMAP backend detection and VRAM budget. |
| `leiden_refinement_prototype.md` | Graph-based global refinement (kNN graph, Leiden, split/merge validation) against the current pipeline; CPU and GPU (added when the prototype lands, epic #346). |

## Serving, decode and models

| Note | What it holds |
|---|---|
| [triton_deep_research_2026-09.md](triton_deep_research_2026-09.md) | External references, production lessons and a performance architecture review for Triton-based vision serving (basis for #40). |
| [dali_gpu_decode_lessons.md](dali_gpu_decode_lessons.md) | GPU decode inside Triton: traps and lessons. |
| [vlm_serving_sizing.md](vlm_serving_sizing.md) | vLLM sizing, flag recon, profile sweep and VLM-versus-VLM comparison. |
| [embedding_model_eval_plan.md](embedding_model_eval_plan.md) | Evaluation design for an optional multimodal embedding model and reference arms (LLM-based arms served, encoders on TensorRT), scheduling protocol and metrics schema (#347). |

## Long-term

| Note | What it holds |
|---|---|
| [OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md](OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md) | Discovering same-place photo groups, photogrammetry and a dated spatial timeline (issue #348, milestone v0.9.0). Not implemented. |

## Related material elsewhere

- A domain-specific bake-off paper (curated, cluster-stratified training data and a detector comparison) is maintained privately outside this repository; its data-engine section corresponds to the ingest, clustering and split work measured in these notes. The public write-up of the systems work is planned from the measurement logs (see "Writing up" in [CLAUDE.md](CLAUDE.md)).

- Raw benchmark JSON used by the docs: [`docs/benchmarks/`](../benchmarks/).
- Performance numbers and acceptance bars: [`docs/PERFORMANCE.md`](../PERFORMANCE.md).
- Market and competitor analysis: [`docs/market/`](../market/).
- Bench and evaluation tooling: `scripts/bench/` (`cluster_eval*.py`, `cluster_ladder.py`, `scale_run.py`).
