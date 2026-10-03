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
| [generic_detector_and_selective_embedding_plan.md](generic_detector_and_selective_embedding_plan.md) | partially done (W0 to W3 merged) | #52 | Default detector, store every detection, per-project embedding policy |
| [sam3_full_image_detection_plan.md](sam3_full_image_detection_plan.md) | open | #30 (shares the gating layer with #46) | Open-vocabulary full-image detection with SAM 3 |
| [triton_pipeline_optimization_plan.md](triton_pipeline_optimization_plan.md) | open (plan on main; baseline-first measurement) | #40 | Decode once, stay on GPU, Triton pipeline optimization with before/after numbers. |
| [release_acceptance_plan.md](release_acceptance_plan.md) | open (release scripts and API-level live tests exist) | #54 (installer live test #44) | The scripted clean-install-to-uninstall run that gates each release |
| [combine_preview_warnings_plan.md](combine_preview_warnings_plan.md) | open | #55 | `embedding_model_mismatch` and `region_profiles_differ` combine preview warnings |
| [storage_sizing_and_ingest_baselines.md](storage_sizing_and_ingest_baselines.md) | partially done (measurements done; publish and re-measure open) | #45, #53, #56 | Measured OpenSearch storage per vector, item and image; sizing formula; ingest and pipeline baseline to repeat |

## Open work tracked as issues without a plan doc

The issue body carries the full description (context, evidence, proposed fix, acceptance).

| Issue | Topic |
|---|---|
| #57 | Region re-verify and requeue paths can overwrite stored human data or select the wrong items |
| #58 | Config store, project clone, keymap and activation hardening |
| #59 | Test fakes, dead helpers, destructive developer tooling |
| #60 | Failures that look like empty results or flood logs (OCR, worker tracebacks, promote warm-up, health, PE FP16) |
| #61 | VLM proposal noise, optional registry prior, label agreement vs cluster purity |
| #62 | Split oversize files, unify job state files, rename the `yolo-api` compose service |
| #63 | Release decisions (`:latest`, control-plane-only) and remaining local VLM catalog verification |
| #64 | Docs: `include_classes` location and "Importing other formats" |
| #65 | Roadmap: deferred any-domain features |
| #66 | Compose overlay and make targets for GPU clustering |
| #38, #39, #41, #42, #43, #46, #51 | Existing follow-ups, image slimming, 12 GB engines, docs screenshots, region-stage gating, regions route |

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
