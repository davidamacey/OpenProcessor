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
| [sam3_full_image_detection_plan.md](sam3_full_image_detection_plan.md) | implemented (waves 1 to 8; as-built deviations in section 16) | #30 (shares the gating layer with #46, region-stage gating #46 also shipped) | Open-vocabulary full-image detection with SAM 3 |
| [triton_pipeline_optimization_plan.md](triton_pipeline_optimization_plan.md) | open (Wave 0 tooling and the v0.5.0 set A baseline in PR #206; optimization waves not started; sequenced for v0.6.0 by `v060_wave_plan.md`) | #40 | Decode once, stay on GPU, Triton pipeline optimization with before/after numbers; design detail and accuracy gates |
| [v060_wave_plan.md](v060_wave_plan.md) | open (plan only) | #208 to #221, #207, #153 (umbrella #40) | v0.6.0 sequencing: FP16 PE and engine build stamp, API and idle CPU, cluster scheduling, INT8 evaluation, uint8 inputs, shared memory, GPU-resident ingest, clustering acceleration, downstream parity chain |
| [v060_vector_db_evaluation_plan.md](v060_vector_db_evaluation_plan.md) | open (plan only) | #221, #56 | Fair benchmark of OpenSearch k-NN (shipped and tuned) against dedicated vector stores, datastore needs beyond vectors, split-architecture costs, decision criteria |
| [release_acceptance_plan.md](release_acceptance_plan.md) | open (release scripts, `make release-verify` and API-level live tests exist; the full clean-install run is not scripted) | #54 (installer live test #44) | The scripted clean-install-to-uninstall run that gates each release |
| [combine_preview_warnings_plan.md](combine_preview_warnings_plan.md) | implemented | #55 | `embedding_model_mismatch` and `region_profiles_differ` combine preview warnings |
| [storage_sizing_and_ingest_baselines.md](storage_sizing_and_ingest_baselines.md) | partially done (measurements done; re-measured on the public set in PR #206; quantization benchmark open, planned in `v060_vector_db_evaluation_plan.md`) | #45, #53, #56 | Measured OpenSearch storage per vector, item and image; sizing formula; ingest and pipeline baseline to repeat |

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
| [clustering_methods.md](../research/clustering_methods.md) | UMAP, AHC, HDBSCAN and IVF methods, the IVF-then-refine hierarchy and the ablation |
| [gpu_clustering.md](../research/gpu_clustering.md) | cuML UMAP backend detection and VRAM budget (#66) |
| [curation_scores.md](../research/curation_scores.md) | Uniqueness, mistakenness, near-duplicate and diversity score validation |
| [detector_quantization_results.md](../research/detector_quantization_results.md) | Detector precision, runtime and hardware matrix |
