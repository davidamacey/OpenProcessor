"""The accuracy audit end to end over an in-memory items index: draw a sample,
queue it, record the human verdicts, report.

Verdicts are written the way the label routes write them, through
``class_label_update`` (the single human class writer), so the outcome stamp is
tested where it really happens.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation import base_curation_config
from src.services.curation.audit_math import AUDIT_OUTCOME, AUDIT_SAMPLE
from src.services.curation.class_label import ItemLabel, class_label_update, human_move_class_update
from src.services.curation.holdout import build_cohort_query


ITEMS = base_curation_config().items_index
BASE = '/curation/projects/default/audit'


def _doc(crop_id: str, detector: str | None, label: str | None, **kw: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_id': 1,
        'class_name': label,
        'class_source': 'vlm',
        'class_validated': False,
        'detector_class_name': detector,
        'detector_confidence': 0.8,
        'confidence': 0.8,
        'test_holdout': False,
        **kw,
    }


def _population() -> dict[str, dict[str, Any]]:
    docs = [_doc(f'car{i}', 'car', 'car' if i % 2 else 'truck') for i in range(6)]
    docs += [_doc(f'bus{i}', 'bus', 'bus') for i in range(3)]
    docs += [
        # never drawn
        _doc('human', 'car', 'car', class_validated=True, class_source='human'),
        _doc('holdout', 'car', 'car', test_holdout=True),
        _doc('excluded', 'car', 'car', class_excluded=True),
        _doc('imported', 'car', 'car', class_source='external_label'),
        _doc('no_detector', None, 'car'),
        _doc('no_class', 'car', None, class_id=None),
    ]
    return {d['crop_id']: d for d in docs}


@pytest.fixture
def fake() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch({ITEMS: _population()})


@pytest.fixture
def client(fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _sampled(fake: QueryFakeOpenSearch) -> set[str]:
    return {k for k, d in fake.docs(ITEMS).items() if d.get(AUDIT_SAMPLE)}


def _start(client: TestClient, **body: Any) -> Any:
    return client.post(f'{BASE}/start', json={'min_per_class': 2, 'sample_size': 5, **body})


def test_start_draws_a_stratified_sample_from_eligible_crops_only(
    client: TestClient, fake: QueryFakeOpenSearch
) -> None:
    r = _start(client)
    assert r.status_code == 200, r.text
    body = r.json()
    # 6 car + 3 bus eligible. Floors of 2 each (4), the last of the budget of 5 goes to car.
    strata = {s['detector_class']: s for s in body['strata']}
    assert (strata['car']['available'], strata['car']['sampled']) == (6, 3)
    assert (strata['bus']['available'], strata['bus']['sampled']) == (3, 2)
    assert body['sampled'] == 5
    drawn = _sampled(fake)
    assert len(drawn) == 5
    assert not drawn & {'human', 'holdout', 'excluded', 'imported', 'no_detector', 'no_class'}
    for crop_id in drawn:
        doc = fake.docs(ITEMS)[crop_id]
        assert doc['audit_batch_id'] == body['batch_id']
        assert doc['audit_label_name'] == doc['class_name']
        assert doc['audit_label_source'] == 'vlm'
        assert doc['class_validated'] is False  # sampling never validates anything


def test_the_sample_is_deterministic_and_never_redraws_a_sampled_crop(
    client: TestClient, fake: QueryFakeOpenSearch
) -> None:
    first = _start(client).json()
    drawn = _sampled(fake)
    second = _start(client)
    assert second.status_code == 200
    again = _sampled(fake) - drawn
    assert not again & drawn
    assert first['batch_id'] != second.json()['batch_id']
    # the same population and parameters pick the same crops
    fresh = QueryFakeOpenSearch({ITEMS: _population()})
    fresh_drawn = {k for k, d in _run_start(fresh).items() if d.get(AUDIT_SAMPLE)}
    assert fresh_drawn == drawn


def _run_start(fake: QueryFakeOpenSearch) -> dict[str, dict[str, Any]]:
    import asyncio

    from src.services.curation.audit import start_audit

    asyncio.run(start_audit(fake, min_per_class=2, sample_size=5))
    return fake.docs(ITEMS)


def test_start_with_nothing_eligible_is_a_409() -> None:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    empty = QueryFakeOpenSearch({ITEMS: {}})
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: empty
    r = TestClient(app).post(f'{BASE}/start', json={})
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'audit_no_candidates'


def test_start_skips_a_crop_that_became_human_validated_after_the_scan(
    fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from src.services.curation import audit

    real = audit.scan_items

    async def racing_scan(*args: Any, **kwargs: Any) -> Any:
        found = await real(*args, **kwargs)
        # a human validates one of the scanned crops between the scan and the write
        fake.docs(ITEMS)['bus0'].update({'class_validated': True, 'class_source': 'human'})
        return found

    monkeypatch.setattr(audit, 'scan_items', racing_scan)
    asyncio.run(audit.start_audit(fake, min_per_class=3, sample_size=100))
    assert 'bus0' not in _sampled(fake)
    assert fake.docs(ITEMS)['bus0']['class_source'] == 'human'


def test_queue_lists_the_sampled_crops_still_waiting_for_a_human(
    client: TestClient, fake: QueryFakeOpenSearch
) -> None:
    batch = _start(client).json()['batch_id']
    ids = sorted(_sampled(fake))
    waiting = client.get(f'{BASE}/queue', params={'page_size': 100}).json()
    assert {i['crop_id'] for i in waiting['items']} == set(ids)
    assert waiting['total'] == 5
    fake.docs(ITEMS)[ids[0]].update(
        class_label_update(
            fake.docs(ITEMS)[ids[0]],
            ItemLabel.human(class_id=1, class_name='car', label_source='human', writer='test'),
        )
    )
    after = client.get(f'{BASE}/queue', params={'batch_id': batch}).json()
    assert ids[0] not in {i['crop_id'] for i in after['items']}
    assert after['total'] == 4


def _label(fake: QueryFakeOpenSearch, crop_id: str, name: str) -> None:
    doc = fake.docs(ITEMS)[crop_id]
    doc.update(
        class_label_update(
            doc, ItemLabel.human(class_id=1, class_name=name, label_source='human', writer='test')
        )
    )


def test_a_human_verdict_stamps_the_outcome_and_the_report_counts_it(
    client: TestClient, fake: QueryFakeOpenSearch
) -> None:
    _start(client, min_per_class=1, sample_size=20)
    docs = fake.docs(ITEMS)
    # car1 and car3 carry label 'car', car0/2/4 carry 'truck' (detector said car for all).
    _label(fake, 'car1', 'car')  # detector right, label right -> agree
    _label(fake, 'car0', 'car')  # detector right, label (truck) wrong -> vlm_wrong
    _label(fake, 'car2', 'truck')  # detector wrong, label (truck) right -> detector_wrong
    _label(fake, 'bus0', 'person')  # both wrong
    assert docs['car1'][AUDIT_OUTCOME] == 'agree'
    assert docs['car0'][AUDIT_OUTCOME] == 'vlm_wrong'
    assert docs['car2'][AUDIT_OUTCOME] == 'detector_wrong'
    assert docs['bus0'][AUDIT_OUTCOME] == 'both_wrong'

    report = client.get(f'{BASE}/report', params={'min_per_class': 3}).json()
    assert report['audited'] == 4
    assert report['pending'] == len(_sampled(fake)) - 4
    detector = {c['name']: c for c in report['detector']}
    assert (detector['car']['n'], detector['car']['correct']) == (3, 2)
    assert detector['car']['insufficient_sample'] is False
    assert (detector['bus']['n'], detector['bus']['correct']) == (1, 0)
    assert detector['bus']['insufficient_sample'] is True
    assert report['confusion'] == {'car': {'car': 2, 'truck': 1}, 'bus': {'person': 1}}
    assert report['outcomes'] == {
        'agree': 1,
        'detector_wrong': 1,
        'vlm_wrong': 1,
        'both_wrong': 1,
    }


def test_an_undone_verdict_drops_out_of_the_report(
    client: TestClient, fake: QueryFakeOpenSearch
) -> None:
    _start(client, min_per_class=1, sample_size=20)
    _label(fake, 'car1', 'car')
    assert client.get(f'{BASE}/report').json()['audited'] == 1
    fake.docs(ITEMS)['car1']['class_validated'] = False  # the label was undone
    assert client.get(f'{BASE}/report').json()['audited'] == 0


def test_a_move_into_a_class_is_a_human_verdict_too(fake: QueryFakeOpenSearch) -> None:
    _run_start(fake)
    crop_id = sorted(_sampled(fake))[0]
    doc = fake.docs(ITEMS)[crop_id]
    doc.update(human_move_class_update(doc, class_id=1, class_name='person', now='2026-10-09'))
    assert doc[AUDIT_OUTCOME] in {'both_wrong', 'detector_wrong', 'vlm_wrong', 'agree'}
    assert doc['audit_human_name'] == 'person'


def test_only_a_human_verdict_on_a_sampled_crop_is_stamped() -> None:
    plain = _doc('c', 'car', 'car')
    human = ItemLabel.human(class_id=1, class_name='car', label_source='human', writer='t')
    assert AUDIT_OUTCOME not in class_label_update(plain, human)  # not sampled
    sampled = {**plain, 'audit_sample': True, 'audit_label_name': 'car'}
    imported = ItemLabel.imported(import_id='i1', class_id=1, class_name='car')
    assert AUDIT_OUTCOME not in class_label_update(sampled, imported)  # not a human
    suggestion = ItemLabel.imported(
        import_id='i1', class_id=1, class_name='car', trust='suggestion'
    )
    assert AUDIT_OUTCOME not in class_label_update(sampled, suggestion)
    assert class_label_update(sampled, human)[AUDIT_OUTCOME] == 'agree'


@pytest.mark.asyncio
async def test_audited_human_labels_are_holdout_eligible_and_unaudited_vlm_labels_are_not(
    client: TestClient, fake: QueryFakeOpenSearch
) -> None:
    _start(client, min_per_class=1, sample_size=20)
    _label(fake, 'car1', 'car')
    resp = await fake.search(index=ITEMS, body={'size': 100, 'query': build_cohort_query()})
    matched = {h['_id'] for h in resp['hits']['hits']}  # the real freeze cohort query
    assert 'car1' in matched  # audited, then human-labelled
    unaudited_vlm = {
        k
        for k, d in fake.docs(ITEMS).items()
        if d['class_source'] == 'vlm' and not d.get(AUDIT_SAMPLE)
    }
    assert unaudited_vlm
    assert not unaudited_vlm & matched
