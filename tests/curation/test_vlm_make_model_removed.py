"""``make`` and ``model`` are no longer hard-coded VLM outputs: a pack declares
the attributes it wants (``attributes``) and they live in the item's ``assist``
object, not in ``vlm_item_make`` / ``vlm_item_model``."""

from __future__ import annotations

import pytest

from curation.test_proposal_denylist import _labeler
from src.clients.curation_opensearch.bodies_core import _items_body
from src.services.curation.class_write_guard import ClassWriteGuard
from src.services.curation.vlm_class_batch import VlmContext, updates_from_predictions
from src.services.labeling.vlm_models import ItemCrop, VlmClassPrediction, VlmCombinedReply
from src.services.labeling.vlm_prompts import REPLY_KEY_CONTRACT


def test_the_classify_contract_names_no_make_or_model() -> None:
    assert REPLY_KEY_CONTRACT['classify']['optional'] == []


def test_the_models_carry_no_make_or_model() -> None:
    for model in (VlmClassPrediction, VlmCombinedReply):
        assert {'make', 'model'}.isdisjoint(model.model_fields)


def test_the_items_index_has_no_make_or_model_field() -> None:
    props = _items_body()['mappings']['properties']
    assert {'vlm_item_make', 'vlm_item_model'}.isdisjoint(props)


@pytest.mark.asyncio
async def test_volunteered_make_and_model_are_ignored_by_the_class_call() -> None:
    labeler = _labeler(
        [{'img': 1, 'class': 'widget', 'confidence': 'high', 'make': 'Acme', 'model': 'X1'}]
    )
    (pred,) = await labeler.label_item_batch([ItemCrop(img_id='c1', jpeg_bytes=b'j')], ['widget'])
    assert pred.class_name == 'widget'
    ctx = VlmContext(
        labeler=labeler,
        guard=ClassWriteGuard('t'),
        class_names=['widget'],
        name_to_id={'widget': 1},
        class_catalog='',
        pack_stamp='p@1',
        force_fit_bypass={'low_conf_skipped': 0, 'attempted': 0},
    )
    updates, _ = updates_from_predictions([pred], ctx)
    assert {'vlm_item_make', 'vlm_item_model'}.isdisjoint(updates['c1'])
