"""RR-10: the exact key identifies the upload only if the bytes kept equal the bytes sent.

The upload route persists the payload under a path named by the content key and hands the same
payload to the pipeline; nothing re-encodes it. The key is the SHA-256 of those bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.services.curation.ingest_models import BatchIngestResult, IngestSummary
from src.services.curation.upload_ingest import StagedEntry, stage_and_ingest


class _Pipeline:
    def __init__(self) -> None:
        self.received: list[bytes] = []
        self.paths: list[str] = []

    async def ingest_batch(self, data: list[bytes], paths: list[str], **_kw: Any) -> Any:
        self.received, self.paths = list(data), list(paths)
        return BatchIngestResult(results=[], summary=IngestSummary())


@pytest.mark.asyncio
async def test_the_stored_file_is_the_upload_and_is_named_by_its_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = SimpleNamespace(upload_root=tmp_path, upload_accepted_extensions=('.jpg',))
    payload = bytes(range(256)) * 40
    pipeline = _Pipeline()
    monkeypatch.setattr(
        'src.services.curation.upload_ingest.schedule_open_vocab_after_ingest', _noop
    )

    await stage_and_ingest(
        None,  # type: ignore[arg-type]
        pipeline,  # type: ignore[arg-type]
        [StagedEntry('client/a.jpg', payload, '.jpg')],
        config=cfg,  # type: ignore[arg-type]
        source='upload',
        run_id=None,
    )

    digest = hashlib.sha256(payload).hexdigest()
    (stored_path,) = pipeline.paths
    assert Path(stored_path).name == f'{digest}.jpg'
    assert Path(stored_path).read_bytes() == payload
    assert pipeline.received == [payload]


async def _noop(*_args: object, **_kw: object) -> None:
    return None
