"""``detect`` deletes a stale machine item only if it is still unlocked when
the delete happens (a labeler can confirm it while the detector runs)."""

from __future__ import annotations

from typing import Any

import pytest

from curation.reprocess_fixtures import docs
from curation.test_reprocess_detect import D1, HUMAN, _factory, _req, _world
from src.services.curation.reprocess import apply_reprocess


@pytest.mark.asyncio
async def test_detect_keeps_an_item_a_human_labeled_during_detection(tmp_path, monkeypatch) -> None:
    fake, triton, service, image_id, (crop_id,) = await _world(tmp_path, monkeypatch, [D1])
    triton.detections = []  # the detector no longer finds it: stale, so delete
    real = service.detect_items

    async def detect_then_human_edit(*a: Any, **k: Any):
        out = await real(*a, **k)
        docs(fake)[crop_id].update(HUMAN)
        return out

    monkeypatch.setattr(service, 'detect_items', detect_then_human_edit)
    await apply_reprocess(fake, _req(image_id), service_factory=_factory(service))
    assert crop_id in docs(fake)
