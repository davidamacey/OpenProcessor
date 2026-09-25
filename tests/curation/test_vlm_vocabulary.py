"""The curation API speaks vendor-neutral VLM / classifier vocabulary.

Pins the gemma_* -> vlm_* and v6_* -> classifier_* renames on every
wire-visible surface: route params, review tab ids, stats keys, the
health envelope, and the class_source values queries filter on.
"""

from __future__ import annotations

import re

import pytest
from fastapi.routing import APIRoute

from src.routers.curation.stats import _count_by_proposal, _rollup_class_sources
from src.services.curation import ingest_class_sources
from src.services.curation.review_queries import KNOWN_TABS


_VENDOR_RE = re.compile(r'gemma|(^|_)v6(_|$)', re.IGNORECASE)


def test_non_ingest_class_sources_are_vendor_neutral() -> None:
    for value in (
        ingest_class_sources.VLM_CLASS_SOURCE,
        ingest_class_sources.CLUSTER_MAJORITY_CLASS_SOURCE,
        ingest_class_sources.CLASSIFIER_VLM_AGREEMENT_CLASS_SOURCE,
    ):
        assert not _VENDOR_RE.search(value)
    assert ingest_class_sources.VLM_CLASS_SOURCE == 'vlm'


def test_review_tab_is_vlm_low_conf() -> None:
    assert 'vlm_low_conf' in KNOWN_TABS
    assert not any(_VENDOR_RE.search(t) for t in KNOWN_TABS)


def test_class_source_rollup_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Buckets follow the configured ingest profile names, not literals."""
    monkeypatch.setenv('OP_INGEST_PRIMARY_NAME', 'proposer')
    monkeypatch.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'classifier')
    monkeypatch.setenv('OP_INGEST_SECONDARY_NAME', 'clf')
    buckets = [
        {'key': 'human', 'doc_count': 1},
        {'key': 'vlm', 'doc_count': 2},
        {'key': 'vlm_unmatched', 'doc_count': 3},
        {'key': 'clf_model', 'doc_count': 4},
        {'key': 'cluster_majority_agreement', 'doc_count': 6},
        {'key': 'classifier_vlm_agreement', 'doc_count': 7},
        {'key': 'proposer_proposal', 'doc_count': 5},
        {'key': 'proposer_low_conf', 'doc_count': 10},
        {'key': 'unlabeled_proposal', 'doc_count': 8},
        {'key': 'something_else', 'doc_count': 9},
    ]
    # F-23: proposal sources never carry a class_id (they're the
    # detector's "no class assigned yet" marker), so this function --
    # which callers only ever feed class_id-scoped buckets -- no longer
    # special-cases them; they fall into 'other' here. The real count is
    # `_count_by_proposal`, fed the class-less buckets instead (below).
    assert _rollup_class_sources(buckets) == {
        'by_human': 1,
        'by_vlm': 5,
        'by_classifier': 17,
        'other': 32,
    }


def test_count_by_proposal_sums_proposal_and_low_conf_and_default_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-23: the fixed accounting for 'detector proposed it, nothing has
    classified it yet' -- fed class-less (no class_id) buckets."""
    monkeypatch.setenv('OP_INGEST_PRIMARY_NAME', 'proposer')
    no_class_buckets = [
        {'key': 'proposer_proposal', 'doc_count': 5},
        {'key': 'proposer_low_conf', 'doc_count': 10},
        {'key': 'unlabeled_proposal', 'doc_count': 8},
        {'key': 'vlm_unmatched', 'doc_count': 3},
        {'key': '__none__', 'doc_count': 2},
    ]
    assert _count_by_proposal(no_class_buckets) == 23


@pytest.fixture(scope='module')
def curation_routes() -> list[APIRoute]:
    from src.main import app
    from src.routers.curation._common import config

    return [
        r for r in app.routes if isinstance(r, APIRoute) and r.path.startswith(config.api_prefix)
    ]


def test_no_vendor_names_in_curation_params(curation_routes: list[APIRoute]) -> None:
    names: list[str] = []
    for route in curation_routes:
        names.append(route.path)
        params = route.dependant.query_params + route.dependant.path_params
        names.extend(f'{route.path}?{p.alias}' for p in params)
        for b in route.dependant.body_params:
            model_fields = getattr(b.field_info.annotation, 'model_fields', {})
            names.extend(f'{route.path} body.{name}' for name in model_fields)
    offenders = [n for n in names if _VENDOR_RE.search(n.rsplit('/', 1)[-1])]
    assert offenders == []


def test_auto_label_params_renamed(curation_routes: list[APIRoute]) -> None:
    start = next(r for r in curation_routes if r.path.endswith('/pipeline/auto_label/start'))
    names = {p.alias for p in start.dependant.query_params}
    assert {
        'vlm_batch_size',
        'vlm_concurrency',
        'max_vlm_crops',
        'classifier_confidence_skip_vlm',
        'run_vlm',
    } <= names
