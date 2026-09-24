"""DQ-m3: a VLM label from ``POST /vlm/label_batch`` places the item in its
class cluster, the same as the worker's combined VLM call and the
pipeline's ``cluster_id == class_id`` normalize.

Before, the batch path wrote the class but left ``cluster_id`` alone, so a
labelled item stayed in its candidate cluster (inflating that cluster's
labels) or in another class's cluster until the next clustering run.
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
from src.services.curation.clustering.id_normalize import class_cluster_placement
from src.services.labeling.vlm_labeler import VlmClassPrediction


if TYPE_CHECKING:
    from pathlib import Path


ITEMS = get_curation_config().items_index


def _item(crop_id: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_path': f'/data/{crop_id}.jpg',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_source': 'item_proposal',
        'class_validated': False,
        **extra,
    }


@pytest.mark.asyncio
async def test_label_batch_moves_labelled_items_into_their_class_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmLabelBatchRequest

    answers = {'cand': 'widget', 'other_class': 'widget', 'unmatched': 'zeppelin'}
    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (1, 2, 3)).save(buf, format='JPEG')
    for cid in answers:
        (tmp_path / f'{cid}.jpg').write_bytes(buf.getvalue())
    monkeypatch.setattr(
        vlm_mod, 'get_curation_config', lambda: SimpleNamespace(crop_cache_dir=tmp_path)
    )
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('gadget')
    widget_id = reg.add_class('widget')
    monkeypatch.setattr(vlm_mod, 'get_class_registry', lambda: reg)

    async def _no_pack(_os: Any) -> None:
        return None

    monkeypatch.setattr(vlm_mod, '_default_pack_name', _no_pack)
    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'cand': _item('cand', cluster_id=10004, cluster_subid='10004a'),
                'other_class': _item('other_class', cluster_id=0, class_id=0),
                'unmatched': _item('unmatched', cluster_id=10004),
            }
        }
    )

    class _Labeler:
        async def label_or_propose_batch(self, crops: list[Any], _names: list[str]) -> list[Any]:
            return [
                VlmClassPrediction(img_id=c.img_id, class_name=answers[c.img_id], confidence='high')
                for c in crops
            ]

    monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: _Labeler())
    await vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=list(answers)), fake)

    docs = fake.docs(ITEMS)
    for cid in ('cand', 'other_class'):
        assert docs[cid]['class_id'] == widget_id
        assert docs[cid]['cluster_id'] == widget_id
    assert docs['cand'].get('cluster_subid') is None
    # No registry class: the item stays where clustering put it.
    assert docs['unmatched']['class_source'] == 'vlm_unmatched'
    assert docs['unmatched']['cluster_id'] == 10004


def test_placement_rules() -> None:
    update = {'class_id': 3, 'class_source': 'vlm'}
    assert class_cluster_placement(update, {'cluster_id': 10001}) == {
        'cluster_id': 3,
        'cluster_subid': None,
    }
    assert class_cluster_placement(update, {'cluster_id': 3}) == {}
    # Excluded items keep their exclusion bucket.
    assert class_cluster_placement(update, {'cluster_id': -2, 'class_excluded': True}) == {}
    # Only writes that set a class move anything.
    assert class_cluster_placement({'class_source': 'vlm_unmatched'}, {'cluster_id': 1}) == {}
