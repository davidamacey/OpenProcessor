"""#193: optional per-item detector-name hint in the open-vocabulary VLM prompt."""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.test_proposal_denylist import _labeler
from src.services.labeling.detector_hint import detector_hint_for
from src.services.labeling.vlm_models import ItemCrop
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack


PREFIX = '/curation/projects/default/prompt_packs'


@pytest.mark.parametrize(
    ('src', 'min_pct', 'expected'),
    [
        ({'detector_class_name': 'car', 'detector_confidence': 0.91}, 50, ('car', 0.91)),
        ({'detector_class_name': 'car', 'detector_confidence': 0.91}, 0, ('', None)),
        ({'detector_class_name': 'car', 'detector_confidence': 0.4}, 50, ('', None)),
        ({'detector_class_name': 'car', 'detector_confidence': 0.5}, 50, ('car', 0.5)),
        ({'detector_class_name': '', 'detector_confidence': 0.9}, 1, ('', None)),
        ({'detector_confidence': 0.9}, 1, ('', None)),
        ({'detector_class_name': 'car'}, 1, ('', None)),
        ({'detector_class_name': 'car', 'detector_confidence': 'high'}, 1, ('', None)),
    ],
)
def test_detector_hint_for(src: dict[str, Any], min_pct: int, expected: tuple[str, Any]) -> None:
    assert detector_hint_for(src, min_pct) == expected


def test_default_pack_has_the_hint_off_and_round_trips() -> None:
    assert GENERIC_ITEM_PACK.detector_hint_min_confidence_pct == 0
    pack = dataclasses.replace(GENERIC_ITEM_PACK, detector_hint_min_confidence_pct=40)
    assert PromptPack.from_dict(pack.to_dict()).detector_hint_min_confidence_pct == 40


def _sent_text(labeler: Any) -> str:
    payload = labeler._client.post.await_args.kwargs['json']
    return payload['messages'][1]['content'][0]['text']


_ENTRY = {'img': 1, 'class': 'car', 'confidence': 'high', 'proposed_class': ''}


@pytest.mark.asyncio
async def test_prompt_names_the_detector_class_per_image() -> None:
    pack = dataclasses.replace(GENERIC_ITEM_PACK, detector_hint_min_confidence_pct=1)
    labeler = _labeler([_ENTRY], pack)
    crops = [
        ItemCrop(img_id='c1', jpeg_bytes=b'j', detector_class='car', detector_confidence=0.91),
        ItemCrop(img_id='c2', jpeg_bytes=b'j'),
        ItemCrop(img_id='c3', jpeg_bytes=b'j', detector_class='dog', detector_confidence=0.5),
    ]
    await labeler.label_or_propose_batch(crops, ['car', 'dog'])
    text = _sent_text(labeler)
    assert 'image 1: car (detector confidence 0.91)' in text
    assert 'image 3: dog (detector confidence 0.50)' in text
    assert 'image 2' not in text.split('Label each')[0]
    assert 'hint, not a constraint' in text


@pytest.mark.asyncio
async def test_prompt_has_no_hint_when_the_pack_has_it_off_or_no_crop_has_one() -> None:
    crop = ItemCrop(img_id='c1', jpeg_bytes=b'j', detector_class='car', detector_confidence=0.9)
    off = _labeler([_ENTRY])
    await off.label_or_propose_batch([crop], ['car'])
    assert 'detector' not in _sent_text(off).lower()

    on = _labeler(
        [_ENTRY], dataclasses.replace(GENERIC_ITEM_PACK, detector_hint_min_confidence_pct=1)
    )
    await on.label_or_propose_batch([ItemCrop(img_id='c1', jpeg_bytes=b'j')], ['car'])
    assert 'detector' not in _sent_text(on).lower()


# --- pack REST ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    fake_os = FakeConfigOpenSearch()
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


def _body(**over: object) -> dict[str, object]:
    body = GENERIC_ITEM_PACK.to_dict()
    body.pop('name')
    body.update(over)
    return body


def test_schema_serves_the_setting_as_an_int_row(app_client: TestClient) -> None:
    fields = {f['field']: f for f in app_client.get(f'{PREFIX}/schema').json()['fields']}
    row = fields['detector_hint_min_confidence_pct']
    assert row['kind'] == 'int'
    assert row['group'] == 'open_classify'
    assert '100' in row['help']


@pytest.mark.parametrize('bad', [-1, 101, True, 'x', 2.5, None])
def test_bad_value_is_a_validation_error(app_client: TestClient, bad: object) -> None:
    body = _body(detector_hint_min_confidence_pct=bad)
    report = app_client.post(f'{PREFIX}/validate', json={'body': body}).json()
    assert report['ok'] is False, bad
    assert any(e['field'] == 'detector_hint_min_confidence_pct' for e in report['errors'])


@pytest.mark.parametrize('good', [0, 1, 100])
def test_value_in_range_round_trips(app_client: TestClient, good: int) -> None:
    body = _body(detector_hint_min_confidence_pct=good)
    r = app_client.post(PREFIX, json={'name': 'hint_pack', 'description': '', 'body': body})
    assert r.status_code == 201, r.text
    got = app_client.get(f'{PREFIX}/hint_pack').json()['body']
    assert got['detector_hint_min_confidence_pct'] == good


def test_missing_value_defaults_off(app_client: TestClient) -> None:
    body = _body()
    body.pop('detector_hint_min_confidence_pct')
    assert app_client.post(f'{PREFIX}/validate', json={'body': body}).json()['ok'] is True


# --- wiring: POST /vlm/label_batch -------------------------------------------


def _wire_label_batch(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    pack: PromptPack,
    seen: list[Any],
    doc_extra: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    import io
    from types import SimpleNamespace

    from PIL import Image

    import src.routers.curation.vlm as vlm_mod
    from curation.query_fakes import QueryFakeOpenSearch
    from src.clients.curation_opensearch.registry import ClassRegistry
    from src.config.curation import base_curation_config

    items = base_curation_config().items_index
    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (1, 2, 3)).save(buf, format='JPEG')
    (tmp_path / 'c1.jpg').write_bytes(buf.getvalue())
    monkeypatch.setattr(
        vlm_mod, 'get_curation_config', lambda: SimpleNamespace(crop_cache_dir=tmp_path)
    )
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('widget')
    monkeypatch.setattr(vlm_mod, 'get_class_registry', lambda: reg)

    async def _no_pack(_os: Any) -> None:
        return None

    monkeypatch.setattr(vlm_mod, '_default_pack_name', _no_pack)
    fake = QueryFakeOpenSearch(
        {
            items: {
                'c1': {
                    'crop_id': 'c1',
                    'image_path': '/data/c1.jpg',
                    'bbox_norm': [0.1, 0.1, 0.5, 0.5],
                    'class_source': 'item_proposal',
                    'class_validated': False,
                    **(doc_extra or {}),
                }
            }
        }
    )

    return vlm_mod, fake


class _RecordingLabeler:
    def __init__(self, pack: PromptPack, crops: list[Any]) -> None:
        from src.services.labeling.vlm_client import VlmIdentity

        self.identity = VlmIdentity('env@None', 'test-vlm')
        self._pack = pack
        self.crops = crops

    async def label_or_propose_batch(
        self, crops: list[Any], _names: list[str], **_k: Any
    ) -> list[Any]:
        self.crops.extend(crops)
        return []


@pytest.mark.usefixtures('vlm_env')
@pytest.mark.asyncio
@pytest.mark.parametrize(('pct', 'hinted'), [(0, False), (50, True), (95, False)])
async def test_label_batch_attaches_the_stored_detector_name_only_when_enabled(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, pct: int, hinted: bool
) -> None:
    from src.routers.curation.vlm import VlmLabelBatchRequest

    pack = dataclasses.replace(GENERIC_ITEM_PACK, detector_hint_min_confidence_pct=pct)
    vlm_mod, fake = _wire_label_batch(
        tmp_path,
        monkeypatch,
        pack,
        [],
        doc_extra={'detector_class_name': 'car', 'detector_confidence': 0.9},
    )
    crops: list[Any] = []
    monkeypatch.setattr(
        vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: _RecordingLabeler(pack, crops)
    )
    await vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=['c1']), fake)
    assert len(crops) == 1
    assert crops[0].detector_class == ('car' if hinted else '')
