"""Shared fakes for the open-vocabulary tests: a registry that can grow, a
scriptable segmenter, an ingested image world, and a set decoder."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from PIL import Image

from curation.reprocess_fixtures import (
    FakeTriton,
    images_index,
    jpeg_bytes,
    make_fake,
    make_service,
    servable_root,
)
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.detection.open_vocab_set import decode_open_vocab_set
from src.services.detection.segmenter_http import SegmenterCallError


if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from curation.query_fakes import QueryFakeOpenSearch
    from src.services.curation.ingest import CurationIngestService

BOX = (0.1, 0.2, 0.3, 0.5)
POLY = ((0.1, 0.2), (0.3, 0.2), (0.3, 0.5))


class _Entry:
    def __init__(self, class_id: int, class_name: str) -> None:
        self.class_id, self.class_name, self.deprecated = class_id, class_name, False


class _File:
    def __init__(self, classes: list[_Entry]) -> None:
        self.classes = classes


class StatefulRegistry:
    """A class registry that supports ``add_class`` and remembers what was added."""

    def __init__(self, seed: tuple[str, ...] = ('gadget',)) -> None:
        self.entries = [_Entry(i, n) for i, n in enumerate(seed)]
        self.added: list[str] = []

    def load(self) -> _File:
        return _File(self.entries)

    def add_class(self, name: str, group: str = 'unknown', notes: str = '') -> int:  # noqa: ARG002
        self.added.append(name)
        self.entries.append(_Entry(len(self.entries), name))
        return len(self.entries) - 1


class FakeSegmenter:
    """Implements ``SegmentImage``; records every call, can be taken down."""

    def __init__(self) -> None:
        self.by_prompt: dict[str, list[RegionCandidate]] = {}
        self.default: list[RegionCandidate] | None = None
        self.calls: list[dict[str, Any]] = []
        self.down = False

    async def __call__(
        self,
        jpeg: bytes,
        prompt: str,
        *,
        min_score: float | None,
        max_candidates: int,
        return_masks: bool,
    ) -> list[RegionCandidate]:
        self.calls.append(
            {
                'prompt': prompt,
                'min_score': min_score,
                'max_candidates': max_candidates,
                'return_masks': return_masks,
                'size': Image.open(io.BytesIO(jpeg)).size,
            }
        )
        if self.down:
            raise SegmenterCallError('segmenter call failed: down')
        found = self.by_prompt.get(prompt, self.default or [])
        return list(found)

    @property
    def prompts(self) -> list[str]:
        return [c['prompt'] for c in self.calls]


def cand(box: tuple[float, float, float, float] = BOX, score: float = 0.9) -> RegionCandidate:
    return RegionCandidate(bbox_norm=box, score=score, source='sam3', mask_polygon=POLY)


def make_set(**kw: Any) -> Any:
    body: dict[str, Any] = {
        'targets': [{'prompt': 'traffic cone', 'class_name': 'cone'}],
        'max_enabled_targets': 8,
    }
    body.update(kw)
    return decode_open_vocab_set('street', body)


async def ingested_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: tuple[int, int] = (400, 300), n: int = 1
) -> tuple[QueryFakeOpenSearch, CurationIngestService, list[str]]:
    """``n`` ingested images (no detections, so no items) in a fake project."""
    root = servable_root(tmp_path, monkeypatch)
    fake = make_fake([])
    service = make_service(fake, FakeTriton([]))
    ids = []
    for i in range(n):
        path = root / f'{i}.jpg'
        path.write_bytes(jpeg_bytes(size, seed=i))
        res = await service.ingest_one(path.read_bytes(), str(path))
        assert res.status == 'success'
        ids.append(res.image_id)
    return fake, service, ids


def image_doc(fake: QueryFakeOpenSearch, image_id: str) -> dict[str, Any]:
    return fake.docs(images_index())[image_id]
