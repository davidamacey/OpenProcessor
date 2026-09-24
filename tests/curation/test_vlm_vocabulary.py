"""The curation API speaks vendor-neutral VLM / classifier vocabulary.

Pins the gemma_* -> vlm_* and v6_* -> classifier_* renames on every
wire-visible surface: route params, review tab ids, stats keys, the
health envelope, and the class_source values queries filter on.
"""

from __future__ import annotations

import re

import pytest
from fastapi.routing import APIRoute

from src.routers.curation.ingest import _get_detection_profile
from src.routers.curation.stats import _rollup_class_sources
from src.services.curation import class_sources
from src.services.curation.review_queries import KNOWN_TABS


_VENDOR_RE = re.compile(r'gemma|(^|_)v6(_|$)', re.IGNORECASE)


def test_classifier_class_source_matches_ingest_profile() -> None:
    """Queries filter on CLASSIFIER_CLASS_SOURCE; ingest writes
    f'{profile.name}_model'. They must agree or the filters match nothing."""
    name = _get_detection_profile().name
    assert f'{name}_model' == class_sources.CLASSIFIER_CLASS_SOURCE
    assert f'{name}_low_conf' == class_sources.CLASSIFIER_LOW_CONF_CLASS_SOURCE


def test_review_tab_is_vlm_low_conf() -> None:
    assert 'vlm_low_conf' in KNOWN_TABS
    assert not any(_VENDOR_RE.search(t) for t in KNOWN_TABS)


def test_class_source_rollup_buckets() -> None:
    buckets = [
        {'key': 'human', 'doc_count': 1},
        {'key': 'vlm', 'doc_count': 2},
        {'key': 'vlm_unmatched', 'doc_count': 3},
        {'key': 'item_model', 'doc_count': 4},
        {'key': 'item_low_conf', 'doc_count': 5},
        {'key': 'cluster_majority_agreement', 'doc_count': 6},
        {'key': 'classifier_vlm_agreement', 'doc_count': 7},
        {'key': 'unlabeled_proposal', 'doc_count': 8},
        {'key': 'something_else', 'doc_count': 9},
    ]
    assert _rollup_class_sources(buckets) == {
        'by_human': 1,
        'by_vlm': 5,
        'by_classifier': 22,
        'by_proposal': 8,
        'other': 9,
    }


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
