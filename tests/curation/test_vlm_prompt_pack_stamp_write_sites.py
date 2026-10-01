"""``vlm_prompt_pack`` (any_domain_plan.md §3.7/§9 W2) is stamped onto the
item at every VLM write site: ``/vlm/label_batch``, ``/vlm/verify_regions``,
the pipeline's inline VLM classification stage, and (the 4th site) the
detection worker's bulk writer (``scripts/curation/worker/bulk_writer.py``),
alongside ``RegionFields.profile``/``profile_revision``."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import get_region_fields
from src.config.curation import base_curation_config
from src.services.curation import image_serving
from src.services.curation.region_boxes import RegionBox, boxes_write_fields
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_labeler import VlmClassPrediction
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, prompt_pack_stamp


if TYPE_CHECKING:
    from pathlib import Path

ITEMS = base_curation_config().items_index
F = get_region_fields()


pytestmark = pytest.mark.usefixtures('vlm_env')


def _item(crop_id: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_path': f'/data/{crop_id}.jpg',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        **boxes_write_fields(
            [RegionBox(box_id='b1', bbox_norm=(0.1, 0.1, 0.5, 0.5), state='proposed')],
            current_src={},
        ),
        'class_source': 'item_proposal',
        'class_validated': False,
        **extra,
    }


class _Labeler:
    identity = VlmIdentity('env@None', 'test-vlm')
    _pack = GENERIC_ITEM_PACK

    async def label_or_propose_batch(self, crops: list[Any], _names: list[str]) -> list[Any]:
        return [
            VlmClassPrediction(img_id=c.img_id, class_name='widget', confidence='high')
            for c in crops
        ]

    async def verify_region(self, _crop: Any) -> Any:
        return SimpleNamespace(is_region=True, reason='looks real', confidence='high')


@pytest.mark.asyncio
async def test_label_batch_stamps_vlm_prompt_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmLabelBatchRequest

    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (1, 2, 3)).save(buf, format='JPEG')
    (tmp_path / 'cand.jpg').write_bytes(buf.getvalue())
    monkeypatch.setattr(
        vlm_mod, 'get_curation_config', lambda: SimpleNamespace(crop_cache_dir=tmp_path)
    )
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('widget')
    monkeypatch.setattr(vlm_mod, 'get_class_registry', lambda: reg)

    async def _no_pack(_os: Any) -> None:
        return None

    monkeypatch.setattr(vlm_mod, '_default_pack_name', _no_pack)
    monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: _Labeler())
    fake = QueryFakeOpenSearch({ITEMS: {'cand': _item('cand')}})

    await vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=['cand']), fake)

    doc = fake.docs(ITEMS)['cand']
    assert doc['vlm_prompt_pack'] == prompt_pack_stamp(GENERIC_ITEM_PACK)


@pytest.mark.asyncio
async def test_verify_regions_stamps_vlm_prompt_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmVerifyRegionsRequest

    async def _no_pack(_os: Any) -> None:
        return None

    monkeypatch.setattr(vlm_mod, '_default_pack_name', _no_pack)
    monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: _Labeler())
    monkeypatch.setattr(image_serving, 'resolve_crop_root', lambda _p: '/data')
    monkeypatch.setattr(image_serving, 'resolve_safe_path', lambda p, _r: p)
    monkeypatch.setattr(
        image_serving, 'THUMBNAIL_CACHE', SimpleNamespace(get_or_compute=lambda *_a, **_k: b'jpeg')
    )
    fake = QueryFakeOpenSearch({ITEMS: {'cand': _item('cand')}})

    await vlm_mod.vlm_verify_regions(VlmVerifyRegionsRequest(crop_ids=['cand']), fake, object())

    doc = fake.docs(ITEMS)['cand']
    assert doc['vlm_prompt_pack'] == prompt_pack_stamp(GENERIC_ITEM_PACK)
