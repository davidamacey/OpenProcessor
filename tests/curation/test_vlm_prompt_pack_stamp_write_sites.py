"""``vlm_prompt_pack`` is stamped onto the
item at every VLM write site: ``/vlm/label_batch``, the assist run's verify action,
the pipeline's inline VLM classification stage, and (the 4th site) the
detection worker's bulk writer (``scripts/curation/worker/bulk_writer.py``),
alongside ``RegionFields.profile``/``profile_revision``."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from _class_registry_fake import InMemoryRegistry
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_region_fields
from src.config.curation_accessors import base_curation_config
from src.domain.region_boxes import RegionBox, boxes_write_fields
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_models import VlmClassPrediction
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, prompt_pack_stamp


if TYPE_CHECKING:
    from pathlib import Path

ITEMS = base_curation_config().items_index
F = get_region_fields()


pytestmark = pytest.mark.usefixtures('vlm_env_active')


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

    async def label_or_propose_batch(
        self, crops: list[Any], _names: list[str], **_k: Any
    ) -> list[Any]:
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
    import src.services.curation.vlm_batch as vlm_mod
    from src.services.curation.vlm_batch import run_label_batch

    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (1, 2, 3)).save(buf, format='JPEG')
    (tmp_path / 'default').mkdir(exist_ok=True)
    (tmp_path / 'default' / 'cand.jpg').write_bytes(buf.getvalue())
    monkeypatch.setattr(
        vlm_mod, 'get_curation_config', lambda: SimpleNamespace(crop_cache_dir=tmp_path)
    )
    reg = InMemoryRegistry()
    reg.add('widget')
    monkeypatch.setattr(vlm_mod, 'get_class_registry', lambda: reg)

    labeler = _Labeler()
    fake = QueryFakeOpenSearch({ITEMS: {'cand': _item('cand')}})

    await run_label_batch(fake, labeler, ['cand'])

    doc = fake.docs(ITEMS)['cand']
    assert doc['vlm_prompt_pack'] == prompt_pack_stamp(GENERIC_ITEM_PACK)
