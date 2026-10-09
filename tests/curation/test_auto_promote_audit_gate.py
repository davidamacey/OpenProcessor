"""``POST /clusters/auto_promote`` is gated on the accuracy audit: a class is
promoted only when enough audited crops show the detector is right about it.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import order matters (see auto_promote.py's module docstring).
import src.services.curation.clustering.orchestrator as _orchestrator  # noqa: F401
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation import base_curation_config
from src.services.curation.audit_math import (
    AUDIT_HUMAN,
    AUDIT_LABEL_NAME,
    AUDIT_LABEL_SOURCE,
    AUDIT_OUTCOME,
    AUDIT_SAMPLE,
)
from src.services.curation.ingest_class_sources import classifier_class_sources


ITEMS = base_curation_config().items_index
URL = '/curation/projects/default/clusters/auto_promote'
CLASSIFIER = sorted(classifier_class_sources())[0]


def _member(crop_id: str, cluster: int, name: str, **kw: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'cluster_id': cluster,
        'class_id': 1,
        'class_name': name,
        'class_source': CLASSIFIER,
        'class_validated': False,
        'test_holdout': False,
        'detector_class_name': name,
        **kw,
    }


def _audited(crop_id: str, detector: str, human: str) -> dict[str, Any]:
    """A crop a human labelled in the audit (so it is validated and not promotable)."""
    return {
        'crop_id': crop_id,
        'cluster_id': 3,
        'class_id': 2,
        'class_name': human,
        'class_source': 'human',
        'class_validated': True,
        'detector_class_name': detector,
        AUDIT_SAMPLE: True,
        AUDIT_LABEL_NAME: detector,
        AUDIT_LABEL_SOURCE: 'vlm',
        AUDIT_HUMAN: human,
        AUDIT_OUTCOME: 'agree' if detector == human else 'detector_wrong',
    }


def _population(*, van_right: int, van_wrong: int) -> dict[str, dict[str, Any]]:
    docs = [_member(f'van{i}', 10001, 'van') for i in range(6)]
    # a second promotable cluster of a class nothing is classifier-labelled in
    docs += [
        _member(f'bus{i}', 10002, 'bus', class_source='human', class_validated=True)
        for i in range(5)
    ]
    docs += [_audited(f'a_ok{i}', 'van', 'van') for i in range(van_right)]
    docs += [_audited(f'a_bad{i}', 'van', 'truck') for i in range(van_wrong)]
    return {d['crop_id']: d for d in docs}


def _client(fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from unittest.mock import AsyncMock

    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _validated(fake: QueryFakeOpenSearch) -> set[str]:
    return {
        k
        for k, d in fake.docs(ITEMS).items()
        if k.startswith('van') and d.get('class_validated') is True
    }


def test_refused_with_no_audit_and_nothing_is_written(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = QueryFakeOpenSearch({ITEMS: _population(van_right=0, van_wrong=0)})
    r = _client(fake, monkeypatch).post(URL)
    assert r.status_code == 409, r.text
    detail = r.json()['detail']
    assert detail['error'] == 'audit_required'
    assert detail['classes'] == [{'name': 'van', 'audited': 0, 'precision': None}]
    assert _validated(fake) == set()
    assert fake.write_calls == 0


def test_refused_when_the_audit_sample_is_below_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = QueryFakeOpenSearch({ITEMS: _population(van_right=29, van_wrong=0)})
    r = _client(fake, monkeypatch).post(URL)
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'audit_required'
    assert r.json()['detail']['classes'][0]['audited'] == 29


def test_refused_when_the_audited_precision_is_too_low(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = QueryFakeOpenSearch({ITEMS: _population(van_right=28, van_wrong=2)})  # 28/30 = 0.933
    r = _client(fake, monkeypatch).post(URL)
    assert r.status_code == 409
    detail = r.json()['detail']
    assert detail['error'] == 'audit_precision_low'
    assert detail['classes'] == [
        {'name': 'van', 'audited': 30, 'precision': pytest.approx(0.9333, abs=1e-3)}
    ]
    assert _validated(fake) == set()


def test_allowed_when_audited_precision_clears_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = QueryFakeOpenSearch({ITEMS: _population(van_right=29, van_wrong=1)})  # 0.9667
    r = _client(fake, monkeypatch).post(URL)
    assert r.status_code == 200, r.text
    assert r.json()['promoted'] == 6
    assert _validated(fake) >= {f'van{i}' for i in range(6)}


def test_the_threshold_is_a_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = QueryFakeOpenSearch({ITEMS: _population(van_right=28, van_wrong=2)})
    client = _client(fake, monkeypatch)
    assert client.post(URL, params={'promote_min_precision': 0.95}).status_code == 409
    assert client.post(URL, params={'promote_min_precision': 0.9}).status_code == 200


def test_force_bypasses_the_gate_and_logs_it_distinctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.curation.clustering import auto_promote

    events: list[tuple[str, dict[str, Any]]] = []

    class _Spy:
        def warning(self, event: str, **fields: Any) -> None:
            events.append((event, fields))

        def info(self, *_a: Any, **_k: Any) -> None:
            return None

    monkeypatch.setattr(auto_promote, 'logger', _Spy())
    fake = QueryFakeOpenSearch({ITEMS: _population(van_right=0, van_wrong=0)})
    r = _client(fake, monkeypatch).post(URL, params={'force': 'true'})
    assert r.status_code == 200, r.text
    assert r.json()['promoted'] == 6
    assert ('auto_promote_audit_gate_bypassed', 'audit_required') in [
        (event, fields['code']) for event, fields in events if 'code' in fields
    ]


def test_a_dry_run_is_never_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = QueryFakeOpenSearch({ITEMS: _population(van_right=0, van_wrong=0)})
    r = _client(fake, monkeypatch).post(URL, params={'dry_run': 'true'})
    assert r.status_code == 200
    assert r.json()['promoted'] == 6  # the count it would promote
    assert fake.write_calls == 0


def test_a_cluster_with_nothing_to_promote_does_not_demand_an_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = {k: d for k, d in _population(van_right=0, van_wrong=0).items() if k.startswith('bus')}
    fake = QueryFakeOpenSearch({ITEMS: docs})
    r = _client(fake, monkeypatch).post(URL)
    assert r.status_code == 200, r.text  # the only promotable cluster has no classifier members
    assert r.json()['promoted'] == 0


@pytest.mark.asyncio
async def test_the_auto_label_pipeline_stage_goes_through_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    from src.routers.curation import pipeline

    source = inspect.getsource(pipeline._run_auto_label)
    assert 'gated_auto_promote(' in source
    assert 'auto_promote_clusters(' not in source
