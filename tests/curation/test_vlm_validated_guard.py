"""VLM class writes must never land on a class-validated item.

A ``vlm_unmatched`` write sets ``class_source`` but not ``class_id``, so
letting it through on a validated item (cluster auto-promote, label
import, ...) produced ``class_source='vlm_unmatched'`` with a validated
class_id that a different writer set.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import get_curation_config
from src.services.labeling.vlm_labeler import VlmClassPrediction


if TYPE_CHECKING:
    from pathlib import Path


ITEMS = get_curation_config().items_index


def _item(crop_id: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_path': f'/data/{crop_id}.jpg',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_id': 0,
        'class_name': 'widget',
        'class_source': 'item_model',
        'label_source': 'item_model',
        'class_validated': False,
        **extra,
    }


@pytest.mark.asyncio
async def test_label_batch_never_overwrites_validated_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmLabelBatchRequest

    ids = ['promoted', 'raced', 'plain']
    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (1, 2, 3)).save(buf, format='JPEG')
    for cid in ids:
        (tmp_path / f'{cid}.jpg').write_bytes(buf.getvalue())
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
            ITEMS: {
                'promoted': _item(
                    'promoted',
                    class_source='cluster_majority_agreement',
                    label_source='cluster_majority_agreement',
                    class_validated=True,
                ),
                'raced': _item('raced'),
                'plain': _item('plain'),
            }
        }
    )
    sent: list[str] = []

    class _Labeler:
        async def label_or_propose_batch(self, crops: list[Any], _names: list[str]) -> list[Any]:
            sent.extend(c.img_id for c in crops)
            # A concurrent auto-promote validates 'raced' after it was
            # fetched but before the VLM result is written.
            fake.docs(ITEMS)['raced'].update(
                class_validated=True, class_source='cluster_majority_agreement'
            )
            return [
                VlmClassPrediction(img_id=c.img_id, class_name='zeppelin', confidence='high')
                for c in crops
            ]

    monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: _Labeler())

    await vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=ids), fake)

    docs = fake.docs(ITEMS)
    assert 'promoted' not in sent
    for cid in ('promoted', 'raced'):
        assert docs[cid]['class_source'] == 'cluster_majority_agreement'
        assert docs[cid]['class_validated'] is True
    # Non-regression: an unvalidated item still records the VLM's miss.
    assert docs['plain']['class_source'] == 'vlm_unmatched'
    assert docs['plain']['class_validated'] is False
