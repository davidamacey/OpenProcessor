"""Curation detection worker package.

Run it with ``python -m scripts.curation.worker``.

Sub-modules:
    state        — constants, ``_ItemTask`` dataclass, crop IO helpers
    cascade      — pending fetch, ``SegmenterClient`` re-export, geometry helpers
    verify       — VLM verify + region doc builders + auto-confirm + verdicts_to_boxes
    bulk_writer  — ``_bulk_update`` + ``_publish_region_events``
    client       — SAM3 HTTP client with circuit breaker
    runner       — the long-running ``run()`` entry point (the live
                   multi-box pipeline; the old single-candidate
                   ``_process_crop``/``combined.py`` cohort path was
                   deleted once this became the only production cascade)
    pipeline     — ``PipelineContext`` (shared queues/counters) + stage helpers
    flow         — producer, writer and metrics reporter tasks
    stage_a      — Stage A.primary (detector routing) + A.vlm_visible
    stage_sam    — Stage A.secondary (segmenter + text-hint fallback)
    stage_b      — Stage B (batched combined VLM call)
    __main__     — ``parse_args`` + ``main`` for ``python -m`` invocation
"""
