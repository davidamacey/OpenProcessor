"""A job refused by the wrong-space guard ends failed, and the person reads the remedy in it.

The guard (``require_encoder_for``) refuses a project whose item or box space is of another
encoder or size, and its text ends with the remedy ("re-ingest the photos into a new project").
Three jobs reach it in the worker: the embed reprocess, the dataset import and the re-index. What
the person sees is the job record's ``error_code`` and ``error_message``, which the job wrapper
writes from the exception its handler ended with; so each job must let the refusal reach the
wrapper whole, and the wrapper must keep its text. The re-index job has no handler that catches
it, so only the wrapper step is shown for it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.config.job_kinds import KINDS
from src.domain import job_states as js
from src.domain.embedding_spaces import (
    EmbeddingSpaceError,
    EncoderBuild,
    Target,
    encoder_mismatch_message,
)
from src.services.curation import reprocess_images
from src.services.curation.dataset_import import runner
from src.services.curation.dataset_import.job import ImportRunError, finish
from src.services.curation.dataset_import.report import ImportReport
from src.services.curation.reprocess_embed_spec import EmbedSpec
from src.tasks.runtime import job_task


REMEDY = 're-ingest the photos into a new project'


def _refusal() -> EmbeddingSpaceError:
    message = encoder_mismatch_message(
        project='cars',
        space_key='item__pe_crop_encoder_336',
        target=Target.ITEM,
        space_build=EncoderBuild('pe_crop_encoder', 336),
        served_build=EncoderBuild('pe_crop_encoder', 224),
    )
    assert message is not None
    return EmbeddingSpaceError('embedding_model_unavailable', message)


def _record(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> dict[str, Any]:
    """What the wrapper writes to the job row when its handler ended with ``error``."""
    written: dict[str, Any] = {}

    def finish_row(_session: Any, _job_id: Any, _token: Any, **fields: Any) -> str:
        written.update(fields)
        return 'row'

    monkeypatch.setattr(job_task.repo, 'finish', finish_row)
    claim = job_task._Claim(job=SimpleNamespace(id='j', fencing_token=1), head=None)  # type: ignore[arg-type]
    job_task._write_terminal(
        None,  # type: ignore[arg-type]
        KINDS['embedding_reindex'],
        claim,
        None,  # type: ignore[arg-type]
        job_task._Ran('error', error=error),
    )
    return written


def _assert_the_person_reads_the_remedy(written: dict[str, Any]) -> None:
    assert written['status'] == js.FAILED
    message = written['error_message']
    assert REMEDY in message  # the last words of the refusal, not cut off
    assert 'cars' in message
    assert '336x336' in message
    assert '224x224' in message


def test_the_wrapper_keeps_the_whole_refusal_text_in_the_job_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-index job's path: its handler raises the refusal and nothing in between catches it."""
    _assert_the_person_reads_the_remedy(_record(monkeypatch, _refusal()))


@pytest.mark.asyncio
async def test_a_reprocess_refused_by_the_space_fails_with_the_text_not_a_count_of_failed_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def existing_images(_os: Any, ids: list[str], *, index: str) -> dict[str, Any]:
        return {i: {'image_id': i} for i in ids}

    async def embed_targets(_os: Any, docs: dict[str, Any], _spec: Any) -> list[str]:
        return list(docs)

    async def policy() -> Any:
        return SimpleNamespace(clustering=SimpleNamespace(automatic=False))

    async def refuse(*_a: Any, **_kw: Any) -> dict[str, int]:
        raise _refusal()

    monkeypatch.setattr(reprocess_images, 'existing_images', existing_images)
    monkeypatch.setattr(reprocess_images, 'embed_targets', embed_targets)
    monkeypatch.setattr(reprocess_images, 'reembed_items', refuse)
    monkeypatch.setattr(reprocess_images, 'get_ingest_policy', policy)
    monkeypatch.setattr(
        reprocess_images, 'get_curation_config', lambda: SimpleNamespace(images_index='img')
    )
    service = SimpleNamespace(pe_encoder=object(), config=SimpleNamespace(ingest_gpu_decode=False))

    with pytest.raises(EmbeddingSpaceError) as ended:
        await reprocess_images.process_images(
            object(),  # type: ignore[arg-type]
            service,  # type: ignore[arg-type]
            scopes=['embed'],
            image_ids=['a', 'b', 'c'],
            embed=EmbedSpec(parts=['crop']),
        )
    _assert_the_person_reads_the_remedy(_record(monkeypatch, ended.value))


class _Ledger:
    def __init__(self, chunks: int) -> None:
        self.pending = list(range(chunks))
        self.saved: list[list[dict[str, Any]]] = []

    async def seed(self, _entries: Any) -> None:
        return None

    async def finished_report(self) -> tuple[ImportReport, int]:
        return ImportReport(), 0

    async def claim(self, _limit: int, *, after: int) -> list[Any]:
        if not self.pending:
            return []
        order = self.pending.pop(0)
        return [SimpleNamespace(order=order, key=f'img{order}.jpg', prior=None)]

    async def save_report(self, _report: ImportReport, *, issues: list[dict[str, Any]]) -> None:
        self.saved.append(list(issues))


class _Job:
    def cancelled(self) -> bool:
        return False

    def paused(self) -> bool:
        return False

    async def progress(self, *_a: Any, **_kw: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_an_import_refused_by_the_space_stops_at_once_and_fails_with_the_whole_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    async def refuse(_ctx: Any, entries: list[Any], _prior: Any) -> Any:
        attempts.append(entries[0].rel_path)
        raise _refusal()

    monkeypatch.setattr(runner, 'import_chunk', refuse)
    monkeypatch.setattr(runner.limits, 'import_chunk_size', lambda: 1)
    ledger = _Ledger(chunks=12)
    entries = [SimpleNamespace(rel_path=f'img{i}.jpg', split=None) for i in range(12)]
    ctx = SimpleNamespace(ledger=ledger, uses_detector=False, profile=None)

    outcome = await runner.run_import_job(_Job(), ctx, entries)  # type: ignore[arg-type]

    assert attempts == ['img0.jpg']  # not the five consecutive failures a flaky chunk gets
    assert outcome.error is not None
    assert REMEDY in outcome.error
    assert outcome.issues[0]['code'] == 'embedding_model_unavailable'
    assert REMEDY in outcome.issues[0]['error']

    with pytest.raises(ImportRunError) as ended:
        await finish(
            None,  # type: ignore[arg-type]
            SimpleNamespace(freeze_test=False),  # type: ignore[arg-type]
            SimpleNamespace(scan={}),  # type: ignore[arg-type]
            outcome,
            total=12,
        )
    _assert_the_person_reads_the_remedy(_record(monkeypatch, ended.value))
