"""Regression tests for the per-scorer pool fix in ``item_scores/job.py``.

Before this fix, every scorer (including ``mistakenness``) was handed the
clustering *residual* pool
(``clustering.embedding_reduce.fetch_residual_embeddings_parallel``), which
explicitly excludes confidently machine-labeled items (``class_validated``,
every ``CONFIDENT_CLASS_SOURCES`` ``class_source``, ``class_excluded``).
That's backwards for mistakenness: it exists to audit machine labels, not
to score unlabeled residual items, and it never touches embeddings at all
(it reads ``probe_pred_*`` / ``class_name`` off the crop doc). Live symptom
before the fix: a 1,351-item residual pool yielded
``mistakenness n_scored=1``, ``n_skipped_outside_probe_classes=1350``, while
7,936 items were actually probe-scored.

Two things are exercised here:

1. ``_fetch_probe_scored_ids`` — the new pool-building query itself,
   against a small in-memory fake OpenSearch that supports scroll
   pagination, proving the query shape (include machine-labeled
   probe-scored items, exclude holdout/excluded/validated) and that
   pagination past a single page collects everything.
2. ``run_scoring_job``'s pool wiring — spy scorers standing in for the
   real ``uniqueness``/``mistakenness`` classes, proving each scorer is
   handed the pool matching its declared ``.pool``, and that a pool no
   requested scorer needs is never fetched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.services.curation.item_scores import job as scores_job
from src.services.curation.item_scores.base import ScoreResult


# =============================================================================
# 1. _fetch_probe_scored_ids — query shape + pagination
# =============================================================================


class _FakeScrollOpenSearch:
    """Minimal in-memory scroll-search fake: real ``exists``/``term``
    filtering over a fixed doc set, paginated ``page_size`` ids at a time,
    mirroring OpenSearch's "keep returning a scroll_id until a page comes
    back empty" contract (not "stop as soon as the last page is short")."""

    def __init__(self, docs: dict[str, dict[str, Any]], *, page_size: int) -> None:
        self.docs = docs
        self.page_size = page_size
        self.scroll_calls = 0
        self.clear_scroll_calls = 0
        self.last_query: dict[str, Any] | None = None
        self._ordered_ids: list[str] = []
        self._cursor = 0

    def _matches(self, doc: dict[str, Any], query: dict[str, Any]) -> bool:
        clause = query['bool']
        for f in clause.get('filter', []):
            field = f['exists']['field']
            if doc.get(field) is None:
                return False
        for mn in clause.get('must_not', []):
            [(field, value)] = mn['term'].items()
            if doc.get(field) == value:
                return False
        return True

    def _page(self) -> dict[str, Any]:
        page_ids = self._ordered_ids[self._cursor : self._cursor + self.page_size]
        self._cursor += len(page_ids)
        return {
            '_scroll_id': 'scroll-token',
            'hits': {'hits': [{'_id': i} for i in page_ids]},
        }

    async def search(self, *, index: str, body: dict[str, Any], scroll: str) -> dict[str, Any]:  # noqa: ARG002
        assert scroll, 'probe_scored fetch must scroll, not rely on a single page'
        assert body.get('_source') is False, 'probe_scored fetch must not read embeddings'
        self.last_query = body['query']
        self._ordered_ids = sorted(
            doc_id for doc_id, doc in self.docs.items() if self._matches(doc, body['query'])
        )
        self._cursor = 0
        return self._page()

    async def scroll(self, *, scroll_id: str, scroll: str) -> dict[str, Any]:  # noqa: ARG002
        assert scroll_id == 'scroll-token'
        self.scroll_calls += 1
        return self._page()

    async def clear_scroll(self, *, scroll_id: str) -> dict[str, Any]:  # noqa: ARG002
        self.clear_scroll_calls += 1
        return {}


def _probe_item(
    *,
    probe_pred_class: str | None = 'cat',
    test_holdout: bool = False,
    class_excluded: bool = False,
    class_validated: bool = False,
    class_source: str | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {'probe_pred_class': probe_pred_class}
    if test_holdout:
        doc['test_holdout'] = True
    if class_excluded:
        doc['class_excluded'] = True
    if class_validated:
        doc['class_validated'] = True
    if class_source is not None:
        doc['class_source'] = class_source
    return doc


@pytest.mark.asyncio
async def test_probe_scored_pool_includes_machine_labeled_excludes_gated() -> None:
    """A machine-labeled probe-scored item (the exact case the residual
    pool drops) must be IN the probe_scored pool; validated/excluded/
    holdout items must be OUT."""
    from src.services.curation.clustering.embedding_reduce import CONFIDENT_CLASS_SOURCES

    machine_label_source = next(iter(CONFIDENT_CLASS_SOURCES))
    docs = {
        'machine_labeled': _probe_item(class_source=machine_label_source),
        'plain_probe_scored': _probe_item(),
        'validated': _probe_item(class_validated=True),
        'excluded': _probe_item(class_excluded=True),
        'holdout': _probe_item(test_holdout=True),
        'not_probe_scored': _probe_item(probe_pred_class=None),
    }
    fake = _FakeScrollOpenSearch(docs, page_size=10)

    ids = await scores_job._fetch_probe_scored_ids(fake)

    assert set(ids) == {'machine_labeled', 'plain_probe_scored'}


@pytest.mark.asyncio
async def test_probe_scored_pool_paginates_past_one_page() -> None:
    docs = {f'item{i}': _probe_item() for i in range(5)}
    fake = _FakeScrollOpenSearch(docs, page_size=2)

    ids = await scores_job._fetch_probe_scored_ids(fake)

    assert sorted(ids) == sorted(docs)
    assert fake.scroll_calls >= 2, (
        'a 5-item pool at page_size=2 must take more than one scroll call'
    )
    assert fake.clear_scroll_calls == 1


# =============================================================================
# 2. run_scoring_job pool wiring
# =============================================================================


class _SpyScorer:
    """Stand-in CropScorer recording exactly what it was handed."""

    def __init__(self, name: str, pool: str) -> None:
        self.name = name
        self.pool = pool
        self.writes: tuple[str, ...] = (f'{name}_score',)
        self.version = 'test'
        self.calls: list[dict[str, Any]] = []

    async def score(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        *,
        opensearch: Any = None,  # noqa: ARG002
        progress: Any = None,  # noqa: ARG002
    ) -> ScoreResult:
        self.calls.append({'ids': list(ids), 'embeddings': embeddings})
        return ScoreResult(scorer=self.name, version=self.version, scored_at='t', fields={})


@pytest.fixture
def _job_state_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_SCORES_STATE_DIR', str(tmp_path / 'scores'))


def _seed_running_job(job_id: str, scorer_names: list[str]) -> None:
    state = scores_job._JobState(job_id=job_id, status='running', scorers=list(scorer_names))
    scores_job._atomic_write(state)


async def _run_with_spies(
    monkeypatch: pytest.MonkeyPatch,
    scorer_names: list[str],
    spies: dict[str, _SpyScorer],
    *,
    residual_pool: tuple[list[str], np.ndarray] | None = None,
    probe_pool_ids: list[str] | None = None,
    fail_residual_fetch: bool = False,
) -> None:
    job_id = 'job-1'
    _seed_running_job(job_id, scorer_names)

    monkeypatch.setattr(
        'src.services.curation.item_scores.get_scorer',
        lambda name, **kwargs: spies[name],  # noqa: ARG005
    )

    async def _fake_residual(client, **kwargs):
        if fail_residual_fetch:
            raise AssertionError('fetch_residual_embeddings_parallel must not be called')
        assert residual_pool is not None
        return residual_pool

    monkeypatch.setattr(
        'src.services.curation.clustering.embedding_reduce.fetch_residual_embeddings_parallel',
        _fake_residual,
    )

    async def _fake_probe_ids(client):
        assert probe_pool_ids is not None
        return probe_pool_ids

    monkeypatch.setattr(scores_job, '_fetch_probe_scored_ids', _fake_probe_ids)

    await scores_job.run_scoring_job(job_id, opensearch=object(), scorer_names=scorer_names)


@pytest.mark.usefixtures('_job_state_dir')
@pytest.mark.asyncio
async def test_uniqueness_still_receives_residual_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyScorer('uniqueness', 'residual')
    residual_ids = ['r1', 'r2', 'r3']
    residual_embeddings = np.ones((3, 4), dtype=np.float32)

    await _run_with_spies(
        monkeypatch,
        ['uniqueness'],
        {'uniqueness': spy},
        residual_pool=(residual_ids, residual_embeddings),
    )

    assert len(spy.calls) == 1
    assert spy.calls[0]['ids'] == residual_ids
    assert np.array_equal(spy.calls[0]['embeddings'], residual_embeddings)

    state = scores_job.get_state()
    assert state['status'] == 'completed'
    assert state['total'] == 3
    assert state['results']['uniqueness']['pool'] == 'residual'
    assert state['results']['uniqueness']['pool_size'] == 3


@pytest.mark.usefixtures('_job_state_dir')
@pytest.mark.asyncio
async def test_mistakenness_receives_probe_scored_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyScorer('mistakenness', 'probe_scored')
    probe_ids = ['machine_labeled_item', 'plain_item']

    await _run_with_spies(
        monkeypatch,
        ['mistakenness'],
        {'mistakenness': spy},
        probe_pool_ids=probe_ids,
        fail_residual_fetch=True,
    )

    assert len(spy.calls) == 1
    assert spy.calls[0]['ids'] == probe_ids
    assert spy.calls[0]['embeddings'].shape == (2, 0)

    state = scores_job.get_state()
    assert state['status'] == 'completed'
    assert state['results']['mistakenness']['pool'] == 'probe_scored'
    assert state['results']['mistakenness']['pool_size'] == 2


@pytest.mark.usefixtures('_job_state_dir')
@pytest.mark.asyncio
async def test_only_needed_pools_are_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    """scorers=['mistakenness'] must never call
    fetch_residual_embeddings_parallel — asserted by making that fake raise
    if it's invoked."""
    spy = _SpyScorer('mistakenness', 'probe_scored')

    await _run_with_spies(
        monkeypatch,
        ['mistakenness'],
        {'mistakenness': spy},
        probe_pool_ids=['a', 'b', 'c'],
        fail_residual_fetch=True,
    )

    state = scores_job.get_state()
    assert state['status'] == 'completed'
    assert len(spy.calls) == 1
    assert spy.calls[0]['ids'] == ['a', 'b', 'c']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
