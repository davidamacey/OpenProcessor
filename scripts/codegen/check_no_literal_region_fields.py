#!/usr/bin/env python3
"""Pre-commit regression guard: no `'plate_...'` literals in ported files.

``RegionFields`` (``src/config/region_fields.py``) is the single source
of truth for OpenSearch region field names — see
``docs/design/curation_design_rationale.md`` §4. On the working
branch a ``'plate_...'``/``"plate_..."`` string literal is *always* a
mistake: it means a file was not fully genericized to read fields via
``RegionFields``.

Run by the ``check-no-literal-region-fields`` pre-commit hook, which
passes every changed ``*.py`` file under ``src/``, ``scripts/`` and
``tests/`` as a positional argument (same wiring style as
``check_file_size.py``). Of those, this script only actually checks
files that fall under ``PORTED_PATHS`` — a growing allowlist of
already-ported paths. It starts empty; each newly-ported path is appended
in the same commit that ports it. This gives a
ratchet: once a module is ported, it can never regress to hardcoding a
`plate_*` literal again.

Two line-level skips for the frozen HTTP wire contract of the
generic curation API (see ``docs/design/curation_api_contract.md``) —
never an OpenSearch field reference, and explicitly out of
``RegionFields``' scope:

- Pydantic attribute declarations of the shape ``plate_foo: ...``
  (matching ``^\\s*plate_[a-z_]+\\s*:``).
- Wire-response dict-literal keys whose *value* is visibly routed
  through a ``RegionFields`` instance -- conventionally bound to ``F``,
  ``_F``, or ``fields`` across this codebase (``F.foo`` / ``doc[F.foo]``
  / ``fields.foo``), a Pydantic model attribute (``payload.foo``), or a
  URL path literal (``f'/...'``) — e.g. ``'plate_status':
  src.get(fields.status)`` in a router's OpenSearch-doc -> wire-JSON
  serializer, or ``'plate_text' in fields_set`` checking membership
  against a wire model's own frozen field-set. The **left-hand** key is
  the wire contract (frozen); the right-hand side is what this guard
  actually polices, and it's already clean by construction here.
- ``wire_fields.append('plate_foo')`` bookkeeping -- a router recording
  which frozen wire-contract field *names* it just applied (for an
  ``updated_fields`` response), never an OpenSearch document key.

Run manually: `python3 scripts/codegen/check_no_literal_region_fields.py <files...>`
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Growing allowlist of paths (files or directory prefixes, POSIX,
# relative to the repo root) that have been ported to the `curation`
# namespace and are expected to be free of `plate_*` OpenSearch-field
# literals. Starts empty (scaffolding only, nothing ported
# yet). Each newly-ported path is appended here in the same
# commit that ports it.
#
# Foundations note: `test_region_fields_mapping_
# coverage.py` and `test_region_fields.py` construct a `roi_*`-named
# RegionFields instance to prove overridability — not `plate_*`,
# so they never needed a guard exemption in the first place.
PORTED_PATHS: tuple[str, ...] = (
    # commit (a) — OpenSearch client
    'src/clients/curation_opensearch.py',
    'tests/curation/test_curation_opensearch.py',
    # commit (b) — router `_common` foundations
    'src/routers/curation/_common.py',
    'src/routers/curation/__init__.py',
    'tests/curation/test_ensure_indexes.py',
    # Image serving, history, source-image cache
    'src/services/curation/image_serving.py',
    'src/services/curation/history.py',
    'src/services/curation/source_image_cache.py',
    'src/routers/curation_images.py',
    'tests/curation/test_curation_images.py',
    'tests/curation/test_history.py',
    'tests/curation/test_source_image_cache.py',
    # Clustering methods + backend primitives
    'src/services/curation/clustering/backend.py',
    'src/services/curation/clustering/id_normalize.py',
    'src/services/curation/clustering/outliers.py',
    'src/services/curation/clustering/methods/',
    'tests/curation/test_cluster_id_normalize.py',
    # Clustering orchestrator + cluster/umap/viz routers
    'src/services/curation/clustering/orchestrator.py',
    'src/services/curation/clustering/auto_promote.py',
    'src/services/curation/clustering/embedding_reduce.py',
    'src/services/curation/embedding_viz.py',
    'src/routers/curation/clusters.py',
    'src/routers/curation_umap.py',
    'src/routers/curation/viz.py',
    'tests/curation/test_clustering_orchestrator.py',
    'tests/curation/test_clustering_orchestrator_extra.py',
    'tests/curation/test_cluster_backend_nonblocking.py',
    'tests/curation/test_ivf_idspace.py',
    'tests/curation/test_embedding_viz.py',
    'tests/curation/test_curation_viz_router.py',
    'tests/curation/test_cluster_representatives_router.py',
    # A companion regression-guard test file from an earlier version of
    # this codebase is deliberately NOT ported — it exercised four
    # legacy-only pre-commit guard scripts (label-validated, prototype,
    # legacy-search, mobileclip guards) that don't apply on this
    # codebase.
    # src.clients.occ ported early — a genuine hard dependency of
    # auto_promote.py that would otherwise leave `import src.main` clean
    # but the function itself uncallable.
    'src/clients/occ.py',
    'tests/curation/occ_fakes.py',
    # Scoring, selection and review services + routers
    'src/services/curation/item_scores/',
    'src/services/curation/selection/',
    'src/services/curation/review_queries.py',
    'src/services/curation/review_sorts.py',
    'src/services/curation/holdout.py',
    'src/services/curation/strategy_registry.py',
    'src/routers/curation/review.py',
    'src/routers/curation/scores.py',
    'src/routers/curation/select.py',
    'src/routers/curation/methods.py',
    'scripts/curation/backfill_scores.py',
    'tests/curation/test_crop_scores.py',
    'tests/curation/test_mistakenness.py',
    'tests/curation/test_kcenter_greedy.py',
    'tests/curation/test_select_router.py',
    'tests/curation/test_review_router.py',
    'tests/curation/test_review_sorts.py',
    'tests/curation/test_review_disagreements.py',
    'tests/curation/test_scores_router.py',
    'tests/curation/test_methods_router.py',
    'tests/curation/test_test_holdout_freeze.py',
    'tests/curation/test_backfill_scores_cli.py',
    # Training pipeline (services, router, bakeoff harness)
    'src/services/training/jobs.py',
    'src/services/training/preflight_scan.py',
    'src/services/training/profiles.py',
    'src/services/training/triton_promote.py',
    'src/services/training/yolo_triton_config.py',
    'src/services/training/gpu_arbiter.py',
    'src/config/gpu_arbiter.py',
    'src/routers/curation_train.py',
    'tests/curation/test_train_jobs.py',
    'tests/curation/test_train_preflight_scan.py',
    'tests/curation/test_yolo26_triton_config.py',
    'tests/curation/test_triton_promote.py',
    'tests/curation/test_promote_registry_pin.py',
    'tests/curation/test_train_router.py',
    'tests/curation/test_gpu_arbiter.py',
    'tests/curation/test_gpu_arbiter_config.py',
    'src/routers/curation/bakeoff.py',
    'scripts/curation/bakeoff/',
    'tests/curation/test_bakeoff_router.py',
    # VLM client (transport) + PromptPack (prompt
    # data). `vlm_prompts.py` carries no `RegionFields`-governed literals
    # of its own (its neutral pack's wire-key strings match
    # `RegionFields` defaults already, e.g. `region_visible`) but is
    # listed for completeness since it's part of the same port.
    'src/services/labeling/vlm_client.py',
    'src/services/labeling/vlm_prompts.py',
    'tests/curation/test_prompt_pack.py',
    # VLM labeler (orchestration) + router.
    'src/services/labeling/vlm_labeler.py',
    'src/routers/curation/vlm.py',
    'tests/curation/test_vlm_labeler.py',
    'tests/curation/test_vlm_combined.py',
    'tests/curation/test_class_synonyms.py',
    # Detection cascade, parameterized by
    # DetectionProfile and renamed to region terms.
    'src/services/detection/cascade_detect.py',
    'tests/curation/test_cascade_detect.py',
    'tests/curation/test_region_sanity.py',
    'tests/curation/test_detection_profile_second_profile.py',
    # Region + region-fp routers.
    'src/routers/curation/regions.py',
    'src/routers/curation/regions_fp.py',
    # Curation detection worker package.
    'scripts/curation/worker/',
    'scripts/curation/region_worker_main.py',
    'tests/curation/test_region_worker.py',
    'tests/curation/test_label_combined_wireup.py',
    'tests/curation/test_segmenter_telemetry.py',
    'tests/integration/test_segmenter_circuit_breaker.py',
    # Remaining services.
    'src/services/curation/semantic_search.py',
    'src/services/curation/event_hub.py',
    'src/services/curation/probe_predictions.py',
    'src/services/curation/export.py',
    'src/services/curation/autolabel/job.py',
    'src/services/curation/autolabel/cli.py',
    'tests/curation/test_semantic_search.py',
    'tests/curation/test_probe_predictions.py',
    'tests/curation/test_export_service.py',
    # Leaf routers.
    'src/clients/pe_encoder.py',
    'src/routers/curation/classes.py',
    'src/routers/curation/crops.py',
    'src/routers/curation/events.py',
    'src/routers/curation/export.py',
    'src/routers/curation/ingest.py',
    'src/routers/curation/models.py',
    'src/routers/curation/search.py',
    'src/routers/curation/stats.py',
    'src/routers/curation/pipeline_control.py',
    'src/routers/curation/pipeline_events.py',
    'src/routers/curation/pipeline_health.py',
    'tests/curation/test_pe_encoder.py',
    'tests/curation/test_export_router.py',
    'tests/curation/test_search_router.py',
    'tests/curation/test_search_no_legacy_tokens.py',
    'tests/curation/test_classes_router.py',
    'tests/curation/test_stats_router.py',
    'tests/integration/test_stats_router.py',
    'tests/curation/test_router_wireup.py',
    'tests/curation/test_models_unload.py',
    'tests/curation/test_write_guards.py',
    'tests/curation/test_occ.py',
    'tests/integration/test_ingest_occ.py',
    'tests/integration/test_auto_label_class_name_fix.py',
    'tests/integration/test_request_id_propagation.py',
    # Pipeline router.
    'src/routers/curation/pipeline.py',
    'tests/curation/test_pipeline.py',
    # Generic curation ingest path.
    'src/services/curation/label_import.py',
    'src/services/detection/geometry.py',
    'src/services/curation/ingest.py',
    'src/services/curation/item_doc.py',
    'src/services/curation/clustering/ivf_ingest.py',
    'tests/curation/test_label_import.py',
    'tests/curation/test_geometry.py',
    'tests/curation/test_ingest_service.py',
    'tests/curation/test_ensemble_nms.py',
    'tests/curation/test_pe_preprocess.py',
    'tests/integration/test_ingest_roundtrip.py',
    # Generic single-class / class-subset dataset export.
    'src/services/curation/export_single_class.py',
    'src/services/curation/export_single_class_rows.py',
    'src/services/curation/export_support.py',
    'src/routers/curation/export_single_class.py',
    'tests/curation/test_export_single_class.py',
    # Operator tooling — probe-inference backfill driver.
    'scripts/curation/run_probe.py',
    'tests/curation/test_run_probe_cli.py',
    # Operator tooling — registry-growth reclassification.
    'src/services/curation/registry_reclassify.py',
    'scripts/curation/reclassify_after_registry_growth.py',
    'tests/curation/query_fakes.py',
    'tests/curation/test_registry_reclassify.py',
    # Operator tooling — terminal-status region requeue.
    'src/services/curation/region_requeue.py',
    'scripts/curation/requeue_regions.py',
    'tests/curation/test_region_requeue.py',
    'tests/curation/test_labels_export_roundtrip.py',
)

_LITERAL_RE = re.compile(r"""['"](plate_[a-z_]+)['"]""")
_PYDANTIC_ATTR_RE = re.compile(r'^\s*plate_[a-z_]+\s*:')
# Wire-response dict key whose value is visibly RegionFields-routed, a
# Pydantic wire-model attribute, or a URL path literal — see the module
# docstring's "two line-level skips" note. The URL-literal form covers
# both a literal leading slash (``f'/...'``) and a leading f-string
# interpolation (``f'{config.api_prefix}/...'``) — both are still just
# URL construction, never an OpenSearch field reference.
_WIRE_KEY_RE = re.compile(
    r"""^\s*['"]plate_[a-z_]+['"]\s*:\s*(bool\()?"""
    r"""(src\.get\(_?F\.|src\.get\(fields\.|payload\.|\w+\[_?F\.|\w+\[fields\.|f['"](/|\{))"""
)
# Membership check against a wire model's own `model_fields_set`.
_FIELDS_SET_RE = re.compile(r"""['"]plate_[a-z_]+['"]\s+in\s+fields_set""")
# Bookkeeping: a router recording which frozen wire-contract field name it
# just applied (e.g. into an `updated_fields` response), never an
# OpenSearch document key.
_WIRE_FIELDS_APPEND_RE = re.compile(r"""wire_fields\.append\(['"]plate_[a-z_]+['"]\)""")


def _is_ported(rel_posix: str) -> bool:
    return any(
        rel_posix == prefix or rel_posix.startswith(prefix.rstrip('/') + '/')
        for prefix in PORTED_PATHS
    )


def _is_exempt(rel_posix: str) -> bool:
    return rel_posix.startswith('docs/')


def _scan_file(path: Path) -> list[tuple[int, str]]:
    violations: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return violations
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _PYDANTIC_ATTR_RE.match(line):
            continue
        if (
            _WIRE_KEY_RE.match(line)
            or _FIELDS_SET_RE.search(line)
            or _WIRE_FIELDS_APPEND_RE.search(line)
        ):
            continue
        if _LITERAL_RE.search(line):
            violations.append((lineno, line.strip()))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    exit_code = 0

    for path in args.files:
        try:
            rel_posix = path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            rel_posix = path.as_posix()

        if not _is_ported(rel_posix) or _is_exempt(rel_posix):
            continue

        violations = _scan_file(path)
        if violations:
            exit_code = 1
            sys.stderr.write(f'{rel_posix}:\n')
            for lineno, line in violations:
                sys.stderr.write(f'  {lineno}: {line}\n')

    if exit_code:
        sys.stderr.write(
            "\nERROR: 'plate_...' literal(s) found in ported curation "
            'code. Route field access through a RegionFields instance '
            'instead (src/config/region_fields.py). See '
            'docs/design/curation_design_rationale.md §4.\n'
        )
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
