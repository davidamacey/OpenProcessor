#!/usr/bin/env python3
"""Pre-commit regression guard: no `'plate_...'` literals in ported files.

``RegionFields`` (``src/config/region_fields.py``) is the single source
of truth for OpenSearch region field names — see
``docs/design/oss_genericization_phase2_plan.md`` §3.2. On the working
branch a ``'plate_...'``/``"plate_..."`` string literal is *always* a
mistake: it means a file was copied across from the reference tree
without being genericized to read fields via ``RegionFields``.

Run by the ``check-no-literal-region-fields`` pre-commit hook, which
passes every changed ``*.py`` file under ``src/``, ``scripts/`` and
``tests/`` as a positional argument (same wiring style as
``check_file_size.py``). Of those, this script only actually checks
files that fall under ``PORTED_PATHS`` — a growing allowlist of
already-ported paths (§3.2 "Per-chunk enforcement guard"). It starts
empty in Chunk 0; each later wave appends its newly-ported paths in the
same commit that ports them. This gives a ratchet: once a module is
ported, it can never regress to hardcoding a `plate_*` literal again.

Two hardcoded exemptions (never driven by ``PORTED_PATHS``):
- ``src/config/region_fields.py`` — its docstrings legitimately name
  `plate_*` as the illustrative override example.
- ``tests/curation/test_region_fields.py`` — the overridability fixture
  legitimately constructs a `plate_*`-named instance.

Plus a line-level skip for Pydantic attribute declarations of the shape
``plate_foo: ...`` (matching ``^\\s*plate_[a-z_]+\\s*:``) — those are
the frozen HTTP wire contract with the labeler frontend (see
``docs/design/labeler_api_contract.md``), not an OpenSearch field
reference, and are explicitly out of ``RegionFields``' scope.

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
# literals. Starts empty in Chunk 0 (scaffolding only, nothing ported
# yet). Each later wave appends its newly-ported paths here in the same
# commit that ports them — see §3.2 "Per-chunk enforcement guard".
#
# Chunk 1 (foundations). Note: `test_region_fields_mapping_
# coverage.py` is deliberately NOT listed here — like
# `test_region_fields.py`, it legitimately constructs a `plate_*`-named
# RegionFields instance to prove overridability (§3.2).
PORTED_PATHS: tuple[str, ...] = (
    # commit (a) — OpenSearch client
    'src/clients/curation_opensearch.py',
    'tests/curation/test_curation_opensearch.py',
    # commit (b) — router `_common` foundations
    'src/routers/curation/_common.py',
    'src/routers/curation/__init__.py',
    'tests/curation/test_ensure_indexes.py',
    # Chunk 2 — image serving, history, source-image cache
    'src/services/curation/image_serving.py',
    'src/services/curation/history.py',
    'src/services/curation/source_image_cache.py',
    'src/routers/curation_images.py',
    'tests/curation/test_curation_images.py',
    'tests/curation/test_history.py',
    'tests/curation/test_source_image_cache.py',
    # Chunk 4 commit (a) — clustering methods + backend primitives
    'src/services/curation/clustering/backend.py',
    'src/services/curation/clustering/id_normalize.py',
    'src/services/curation/clustering/outliers.py',
    'src/services/curation/clustering/methods/',
    'tests/curation/test_cluster_id_normalize.py',
    # Chunk 4 commit (b) — clustering orchestrator + cluster/umap/viz routers
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
    # The reference line's clustering-wave regression-guards test file
    # is deliberately NOT ported — it exercises four reference-line-only
    # pre-commit guard scripts (label-validated, prototype, legacy-search,
    # mobileclip guards) that plan section 0.6 already resolved as "not
    # applicable" / "not inherited" on the working branch, independent
    # of this chunk.
    # Chunk 4 commit (b), continued — src.clients.occ ported ahead of its
    # originally-scheduled wave (see the port's commit message / plan
    # deviation note): a genuine, plan-missed hard dependency of
    # auto_promote.py that would otherwise leave `import src.main` clean
    # but the function itself uncallable.
    'src/clients/occ.py',
    'tests/curation/occ_fakes.py',
    # Chunk 5 — scoring, selection and review services + routers
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
    # Chunk 6 — training pipeline (services, router, bakeoff harness)
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
    'scripts/curation/bakeoff/dedup_sweep.py',
    'scripts/curation/bakeoff/lean_candidates.py',
    'scripts/curation/bakeoff/deskew_prototype.py',
    'tests/curation/test_bakeoff_router.py',
    # Chunk 7 commit (a) — VLM client (transport) + PromptPack (prompt
    # data). `vlm_prompts.py` carries no `RegionFields`-governed literals
    # of its own (its neutral pack's wire-key strings match
    # `RegionFields` defaults already, e.g. `region_visible`) but is
    # listed for completeness since it's part of the same port.
    'src/services/labeling/vlm_client.py',
    'src/services/labeling/vlm_prompts.py',
    'tests/curation/test_prompt_pack.py',
    # Chunk 7 commit (b) — VLM labeler (orchestration) + router.
    'src/services/labeling/vlm_labeler.py',
    'src/routers/curation/vlm.py',
    'tests/curation/test_vlm_labeler.py',
    'tests/curation/test_vlm_combined.py',
    'tests/curation/test_class_synonyms.py',
)

# Hardcoded exemptions — never touched by PORTED_PATHS growth.
_FULLY_EXEMPT_FILES = frozenset(
    {
        'src/config/region_fields.py',
        'tests/curation/test_region_fields.py',
    }
)

_LITERAL_RE = re.compile(r"""['"](plate_[a-z_]+)['"]""")
_PYDANTIC_ATTR_RE = re.compile(r'^\s*plate_[a-z_]+\s*:')


def _is_ported(rel_posix: str) -> bool:
    return any(
        rel_posix == prefix or rel_posix.startswith(prefix.rstrip('/') + '/')
        for prefix in PORTED_PATHS
    )


def _is_exempt(rel_posix: str) -> bool:
    if rel_posix in _FULLY_EXEMPT_FILES:
        return True
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
            'docs/design/oss_genericization_phase2_plan.md §3.2.\n'
        )
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
