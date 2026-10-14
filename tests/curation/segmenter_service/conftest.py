"""Fixtures that install the fake SAM 3 behind the real segmenter app."""

from __future__ import annotations

import pytest

from curation.segmenter_service._helpers import _FakeProcessor, _full_mask, sam3_backend, segmenter


@pytest.fixture
def processor() -> _FakeProcessor:
    """One candidate, a plausible box, a mask that fills it."""
    return _FakeProcessor(
        boxes=[[0.40, 0.60, 0.60, 0.68]],
        scores=[0.88],
        masks=_full_mask(),
    )


@pytest.fixture
def served(processor: _FakeProcessor):
    """Install a pool on the real app; restore whatever was there after."""
    previous = segmenter._pool
    segmenter._pool = sam3_backend.ProcessorPool([processor])
    try:
        yield processor
    finally:
        segmenter._pool = previous
