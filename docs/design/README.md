# Design documents

Index of the design docs and plans in `docs/design/`, plus the research notes in
`docs/research/`. Status values: **implemented** (reference doc, code matches), **open** (not
started), **partially done** (evidence in the doc), **in progress**. Work is announced and tracked
in GitHub issues; the repository file is the canonical, versioned copy of each plan.

## Reference docs

| Doc | Status | Issue | What it is |
|---|---|---|---|
| [curation_api_contract.md](curation_api_contract.md) | implemented | n/a | Semantics of the `/curation` HTTP API: ordering, locking, error codes, concurrency tokens, which route for which job |
| [curation_design_rationale.md](curation_design_rationale.md) | implemented | n/a | Why the curation subsystem is built the way it is, and the known tracked gaps |

## Plans

| Doc | Status | Issue | What it is |
|---|---|---|---|
| [generic_detector_and_selective_embedding_plan.md](generic_detector_and_selective_embedding_plan.md) | implemented (W0 to W9; deferred: arbitrary-filter clustering, model-delete guard for a project's own detector, optional `backfill_embedding_state.py`) | #52 | Default detector, store every detection, per-project embedding policy |
| [auto_label_loop_plan.md](auto_label_loop_plan.md) | open (plan; nothing implemented) | #286 (W3 of `v070_wave_plan.md`; seam for #343) | Cluster-sampled VLM auto-labeling loop over assist runs: pinned partition, exact sampled-agreement gate, propagate or split and recurse, budgets, resume, second-opinion audit, rollback, routes, Cropwright panel, staged build order |
| [cluster_search_makemodel_plan.md](cluster_search_makemodel_plan.md) | open (plan; nothing implemented) | #341 to #346, #353, #354, #356, #372, #379, #286, #347 | Clustering and semantic search for faces, objects, whole images and items; Immich comparison; crop search; fine-grained make/model workflow for set P; public proxies and identity ground truth; attribute signals and plate identity text; embedding and processing benchmark; measurement standard; LLM visual verification and a human review gate per wave |
| [sam3_full_image_detection_plan.md](sam3_full_image_detection_plan.md) | implemented (waves 1 to 8; as-built deviations in section 16) | #30 (shares the gating layer with #46, region-stage gating #46 also shipped) | Open-vocabulary full-image detection with SAM 3 |
| [triton_pipeline_optimization_plan.md](triton_pipeline_optimization_plan.md) | partially done (shipped in 0.6.0: the baseline, decode once, uint8 inputs, FP16 PE engine, GPU decode, multi-GPU PE encoding, coalesced writes; the fully GPU-resident pipeline and the 1M scale step are v0.7.0, issue #40) | #40 | Decode once, stay on GPU, Triton pipeline optimization with before/after numbers; design detail and accuracy gates |
| [v070_wave_plan.md](v070_wave_plan.md) | open (plan; waves not started) | #341, #337, #259, #219, #268, #286, #84, #45, #41, #335 (umbrella #40) | v0.7.0 sequencing: cluster evaluation harness first, then cluster and embed pass, high-res path, assist loop, rehearsal and ladders, slim images |
| [program_plan.md](program_plan.md) | open (plan) | every open issue (#8 to #363) and the research items | Ordered program through v0.7.0, v0.8.0 and v0.9.0: completion matrix with the definition of done, Opus design reviews and audit gates, phases A (winter pipeline) to G (white paper), data and storage budget, public multi-view datasets, GPU schedule |
| [v060_wave_plan.md](v060_wave_plan.md) | implemented in 0.6.0 (waves executed; measured results in `docs/PERFORMANCE.md`; the unfinished items moved to milestone v0.7.0) | #208 to #221, #207, #153 (umbrella #40) | v0.6.0 sequencing: FP16 PE and engine build stamp, API and idle CPU, cluster scheduling, INT8 evaluation, uint8 inputs, shared memory, GPU-resident ingest, clustering acceleration, downstream parity chain |
| [v060_serving_backend_trade_study.md](v060_serving_backend_trade_study.md) | open (study and experiment plan; nothing implemented) | WP-S1 to WP-S7 issues (umbrella #40, #67) | Roofline ceilings per stage on one and two A6000s, trade matrix of serving architectures (Triton hardened, bulk ingest runner, ortloom-serve, custom offline), recommendation, gated experiments toward a 1M-image batch |
| [gpu_decode_plan.md](gpu_decode_plan.md) | implemented (built and on by default above 3 MP; build and A/B in section 13 of the doc) | #217 (WP-2.4; related #67, #219, #220) | GPU decode once and a GPU-resident ingest GPU decode: dataflow map, decode and resize costs per image size, contention with PE, decoder coverage, parity and AP effect, size gate, placement, batching, implementation items |
| [v060_vector_db_evaluation_plan.md](v060_vector_db_evaluation_plan.md) | implemented (tuned OpenSearch compared with Milvus, Qdrant, pgvector and LanceDB; results in `docs/PERFORMANCE.md`; Lucene engine adopted for new projects) | #221, #56 | Fair benchmark of OpenSearch k-NN (shipped and tuned) against dedicated vector stores, datastore needs beyond vectors, split-architecture costs, decision criteria |
| [ingest_surfaces_plan.md](ingest_surfaces_plan.md) | implemented in 0.6.0 (S3 and registered sources, bulk ingest runner, poll, cron and webhook triggers, Cropwright sources panel; transports measured and gRPC skipped) | #264, #263 | S3 and S3-compatible bulk source, webhook, schedule and poll-with-checkpoint triggers on one queue, Cropwright sources panel, and the measured decision on a gRPC ingest channel |
| [release_acceptance_plan.md](release_acceptance_plan.md) | open (release scripts, `make release-verify` and API-level live tests exist; the full clean-install run is not scripted) | #54 (installer live test #44) | The scripted clean-install-to-uninstall run that gates each release |
| [combine_preview_warnings_plan.md](combine_preview_warnings_plan.md) | implemented | #55 | `embedding_model_mismatch` and `region_profiles_differ` combine preview warnings |
| [storage_sizing_and_ingest_baselines.md](storage_sizing_and_ingest_baselines.md) | partially done (measurements done; re-measured on the public set in PR #206; quantization benchmark open, planned in `v060_vector_db_evaluation_plan.md`) | #45, #53, #56 | Measured OpenSearch storage per vector, item and image; sizing formula; ingest and pipeline baseline to repeat |
| [label_confirmation_plan.md](label_confirmation_plan.md) | partially done (`vlm_scope` is implemented; #192 fixed the `representatives` bound) | #192 | Detector against VLM against human ground truth: whether the VLM runs on every crop, and how labels are confirmed |
| [v060_execution_path.md](v060_execution_path.md) | in progress (status table, updated as packages merge) | umbrella #40 | Status of the v0.6.0 packages and the order of the remaining work; companion to `v060_wave_plan.md` |
| [docs_overhaul_plan.md](docs_overhaul_plan.md) | in progress | n/a | Documentation overhaul: audit, task guides, pipeline modes, screenshots, and the work breakdown |
| [repo_cleanup_plan.md](repo_cleanup_plan.md) | in progress (PR 1 of 9, the guards) | n/a | Repo cleanup: effective guards, dead code, naming, fail-open handlers, duplicates, and splits of oversize files, in nine PRs |
| [engineering_standards.md](engineering_standards.md) | implemented (reference) | n/a | Portable standards and lessons from the repo cleanup: config that worked, every guard with its baseline and proof recipe, the ordered port checklist for a sister repo, pitfalls hit, before/after numbers |

## Issues without a plan doc

The issue body carries the full description (context, evidence, proposed fix, acceptance).
Rows marked implemented are merged for v0.4.0; the rest are open.

| Issue | Topic |
|---|---|
| #57, #58, #59, #60, #64, #66 | Implemented in the v0.4.0 hardening merge (regions re-verify safety, config-store hardening, test fakes and dev tooling, silent failures, docs, GPU clustering overlay); the issues stay open until the owner closes them |
| #53, #55 | Implemented (shard budget `capacity` block, combine preview warnings) |
| #46 | Implemented (region-stage hit-rate gate, pause and resume) |
| #61 | VLM proposal noise, optional detector hint, label agreement vs cluster purity |
| #62 | Split oversize files, unify job state files, rename the `api` compose service |
| #63 | Release decisions (`:latest`, control-plane-only) and remaining local VLM catalog verification |
| #65 | Roadmap: deferred any-domain features |
| #38, #39, #41, #42, #43 | Existing follow-ups: image slimming and sizes (#41), 12 GB engine builds (#42), docs screenshots (#43) |

## Research notes (`docs/research/`)

Measured snapshots and external-reference research. Numbers drift; each note states where and
how it was measured and whether the design is implemented.

| Doc | What it is |
|---|---|
| [triton_deep_research_2026-09.md](../research/triton_deep_research_2026-09.md) | External references, production lessons and a performance architecture review for Triton-based vision serving (basis for #40) |
| [triton_perf_crossvalidation.md](../research/triton_perf_crossvalidation.md) | Batch size versus output payload findings on a live Triton |
| [dali_gpu_decode_lessons.md](../research/dali_gpu_decode_lessons.md) | GPU decode with DALI inside Triton: traps, ensemble configuration, throughput |
| [vlm_serving_sizing.md](../research/vlm_serving_sizing.md) | vLLM sizing, flag recon, profile sweep and the VLM-versus-segmenter bottleneck finding |
| [pe_encoder_batching_benchmark.md](../research/pe_encoder_batching_benchmark.md) | Perception encoder batched TensorRT parity and throughput |
| [ingest_concurrency_tuning.md](../research/ingest_concurrency_tuning.md) | Batch and concurrency sweep for bulk ingest |
| [embedding_model_eval_plan.md](../research/embedding_model_eval_plan.md) | Plan (not yet measured) to evaluate Google's newest open multimodal embedding model for text and image search, with benchmark design and go/no-go rubric (#347) |
| [cluster_eval_baseline.md](../research/cluster_eval_baseline.md) | Baseline of the cluster-quality harness (false merges and false splits scored separately) on the current IVF + AHC partition of a public COCO project (#341) |
| [v070_measurement_log.md](../research/v070_measurement_log.md) | Paper-ready log of the IVF cluster pass size ladder (10k to 1M crops): stage split, memory, index size, k-NN latency, and the super-linear k-means fit (#337, #356, #45) |
| [clustering_methods.md](../research/clustering_methods.md) | UMAP, AHC, HDBSCAN and IVF methods, the IVF-then-refine hierarchy and the ablation |
| [cluster_types.md](../research/cluster_types.md) | The three kinds of clustering (visual-search IVF, curation item and region-box clustering, face clustering): embedding spaces, algorithms, triggers, storage, routes, limits, and how the graph refinement epic (#346) applies to each |
| [OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md](../research/OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md) | Long-term research: discovering same-place photo groups, COLMAP and Gaussian-splat reconstruction, a dated spatial timeline. Not implemented; scheduled after 0.7.0 |
| [installer_drive_log.md](../research/installer_drive_log.md) | Hand-driven runs of the one-line installer (control-plane-only, no-start, branch mode, upgrade and rollback, edge cases) with the defects they found |
| [cross_domain_clustering_lessons.md](../research/cross_domain_clustering_lessons.md) | What a sibling speaker-linking system does and learned, which techniques transfer to multi-view object and face clustering (multi-prototype clusters, anti-chaining, constraints carried by data, calibrated thresholds with abstention, view-stratified evaluation), and an ordered experiment plan for #342 to #345, #354, #372 and #286 |
| [gpu_clustering.md](../research/gpu_clustering.md) | cuML UMAP backend detection and VRAM budget (#66) |
| [clustering_research_consolidated.md](../research/clustering_research_consolidated.md) | Consolidated clustering research for the four kinds (whole image, crops, faces, fine-grained): sources, real-data diagnostics, recommended method stack and real-data experiments (#346) |
| [curation_scores.md](../research/curation_scores.md) | Uniqueness, mistakenness, near-duplicate and diversity score validation |
| [detector_quantization_results.md](../research/detector_quantization_results.md) | Detector precision, runtime and hardware matrix |
