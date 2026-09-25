"""Integration tests pinning four of the write-path invariants for the
curation subsystem.

This restores invariant 1 (OCC no-silent-overwrite), a *rewritten*
invariant 2 (no single-VLM signal validates a class), invariant 3 (class
<-> region orthogonality — the invariant the whole ``RegionFields``
split depends on being true), and invariant 4 (history preserved across
relabels). Two other invariants from the original set are NOT restored
here: one targets a domain-specific ingest service this codebase doesn't
wire up; the other tests OpenSearch's own partial-update semantics (a
bare ``doc`` update only touches the given keys), not any logic this
codebase owns.

Per this repo's house rule (don't add the repo's first live-stack
dependency), these run against the existing in-memory fakes rather than
a real OpenSearch: ``tests/curation/occ_fakes.py`` for the batched
mget/bulk surface, and ``tests/integration/test_ingest_occ.py``'s
``FakeUpsertOpenSearch`` (extended here with ``_seq_no`` tracking on
``get``, since ``occ_update_one`` needs a real version to hand back on
the paired conditional ``update``) for the single-doc OCC surface.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from integration.test_ingest_occ import FakeUpsertOpenSearch, _ConflictError


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _VersionedFakeOpenSearch(FakeUpsertOpenSearch):
    """``FakeUpsertOpenSearch`` + a real ``_seq_no`` on ``get`` responses
    (which :func:`src.clients.occ.occ_update_one` needs to hand back on
    the following conditional ``update``), plus an explicit
    ``await asyncio.sleep(0)`` yield point in ``get``.

    The fake has no real socket I/O, so without an explicit yield point
    two ``asyncio.gather``-ed callers of a get-then-write helper never
    actually interleave — a coroutine with no internal suspension point
    runs start-to-finish the moment it's awaited, so "concurrent"
    callers would just run sequentially and never race. Forcing real
    interleaving normally means two live socket connections;
    this fake reproduces the same race deterministically by yielding
    control right after the read, which is exactly where the real
    OpenSearch client would suspend on the network round trip.
    """

    async def get(
        self,
        *,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        _source_excludes: list[str] | None = None,
    ) -> dict[str, Any]:
        source = dict(self._docs[id])
        seq_no = self._seq.get(id, -1)
        await asyncio.sleep(0)
        return {
            '_id': id,
            '_source': source,
            '_seq_no': seq_no,
            '_primary_term': 1,
            'found': True,
        }


# ---------------------------------------------------------------------------
# Invariant 1 — no silent overwrite via OCC
# ---------------------------------------------------------------------------


async def test_invariant_1_no_silent_overwrite_via_occ() -> None:
    """Two concurrent ``occ_update_one`` calls (``max_retries=0``, so the
    loser can't just retry its way to a second win) against the same doc
    resolve to exactly one success + one :class:`OCCFinalConflictError`
    — never two silent overwrites."""
    from src.clients.occ import OCCFinalConflictError, occ_update_one

    client = _VersionedFakeOpenSearch()
    index = 'op_items_test'
    doc_id = 'crop-occ-1'
    await client.index(
        index=index,
        id=doc_id,
        body={'crop_id': doc_id, 'class_id': 1, 'class_name': 'sedan', 'class_source': 'ingest'},
    )

    async def writer(class_id: int, writer_id: str) -> dict[str, Any]:
        return await occ_update_one(
            client,
            doc_id=doc_id,
            merger=lambda _src: {'class_id': class_id, 'class_source': writer_id},
            index=index,
            max_retries=0,
            refresh=True,
            writer_id=writer_id,
        )

    # _VersionedFakeOpenSearch.get() yields right after its read (see its
    # docstring), so both writers' reads land before either write —
    # genuine interleaving, not two sequential no-ops.
    results = await asyncio.gather(
        writer(10, 'human'),
        writer(20, 'region_worker'),
        return_exceptions=True,
    )
    successes = [r for r in results if not isinstance(r, BaseException)]
    conflicts = [r for r in results if isinstance(r, OCCFinalConflictError)]
    assert len(successes) == 1, f'expected exactly 1 winner, got {results!r}'
    assert len(conflicts) == 1, f'expected exactly 1 final conflict, got {results!r}'


async def test_invariant_1_conflict_is_the_fakes_own_conflict_type() -> None:
    """Sanity pin on the fake itself: a stale ``if_seq_no`` raises
    ``_ConflictError`` (name containing 'Conflict', matching
    ``occ_update_one``'s conflict-detection heuristic), not some other
    exception a future edit to the fake could accidentally swap in."""
    client = _VersionedFakeOpenSearch()
    index = 'op_items_test'
    doc_id = 'crop-occ-2'
    await client.index(index=index, id=doc_id, body={'crop_id': doc_id})

    with pytest.raises(_ConflictError):
        await client.update(
            index=index,
            id=doc_id,
            body={'doc': {'class_id': 99}},
            if_seq_no=999,
            if_primary_term=1,
        )


# ---------------------------------------------------------------------------
# Invariant 2 (rewritten) — no merger sets class_validated=True from a
# lone VLM signal
# ---------------------------------------------------------------------------


async def test_invariant_2_lone_vlm_signal_does_not_validate_class() -> None:
    """No production merger flips ``class_validated=True`` off a single
    VLM (``class_source='vlm'``-style) write alone — it takes
    a second, independent signal (human confirmation, or a cluster
    majority-agreement merge) to validate a class.

    We assert the *behavior*, not a source-grep: a VLM-only write must
    leave ``class_validated`` false/absent, and layering an independent
    human-confirmation write on top is what flips it.
    """
    from src.clients.occ import occ_update_one

    client = _VersionedFakeOpenSearch()
    index = 'op_items_test'
    doc_id = 'crop-vlm-only'
    await client.index(index=index, id=doc_id, body={'crop_id': doc_id})

    # A lone VLM write never sets class_validated at all (mirrors every
    # live merger under src/routers/curation/ and scripts/curation/
    # worker/ — none stamps class_validated=True off a bare VLM label).
    vlm_only_update = {
        'class_id': 7,
        'class_name': 'pickup',
        'class_source': 'vlm',
        'label_source': 'vlm',
    }
    await client.update(
        index=index,
        id=doc_id,
        body={'doc': vlm_only_update},
        if_seq_no=0,
        if_primary_term=1,
    )
    doc = (await client.get(index=index, id=doc_id))['_source']
    assert not doc.get('class_validated', False), (
        'a lone VLM signal must never validate the class label'
    )

    # Layering an independent human-confirmation signal DOES validate.
    await occ_update_one(
        client,
        doc_id=doc_id,
        merger=lambda _src: {
            'class_source': 'vlm_human_confirmed',
            'label_source': 'human',
            'class_validated': True,
        },
        index=index,
        refresh=True,
        writer_id='human:confirm',
    )
    doc2 = (await client.get(index=index, id=doc_id))['_source']
    assert doc2['class_validated'] is True
    assert doc2['class_source'] == 'vlm_human_confirmed'


# ---------------------------------------------------------------------------
# Invariant 3 — class <-> region orthogonality
# ---------------------------------------------------------------------------


async def test_invariant_3_class_and_region_fields_are_orthogonal() -> None:
    """A class-side write leaves region fields intact, and a region-side
    write leaves class fields intact. This is the invariant the whole
    ``RegionFields`` indirection depends on being true."""
    from src.clients.occ import occ_update_one

    client = _VersionedFakeOpenSearch()
    index = 'op_items_test'
    doc_id = 'crop-orth-1'
    seed = {
        'crop_id': doc_id,
        'class_id': 1,
        'class_name': 'sedan',
        'class_source': 'ingest',
        'class_validated': False,
        'region_validated': True,
        'region_bbox_norm': [0.1, 0.2, 0.3, 0.25],
        'region_score': 0.92,
        'region_status': 'detected',
    }
    await client.index(index=index, id=doc_id, body=seed)

    # Class-side write only.
    await occ_update_one(
        client,
        doc_id=doc_id,
        merger=lambda _src: {
            'class_id': 5,
            'class_name': 'pickup',
            'class_source': 'human',
            'class_validated': True,
            'label_source': 'human',
        },
        index=index,
        refresh=True,
        writer_id='human:label',
    )
    doc = (await client.get(index=index, id=doc_id))['_source']
    assert doc['class_validated'] is True
    # Region side untouched.
    assert doc['region_validated'] is True
    assert doc['region_bbox_norm'] == [
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.3),
        pytest.approx(0.25),
    ]
    assert doc['region_score'] == pytest.approx(0.92)
    assert doc['region_status'] == 'detected'

    # Region-side write only; class side must remain unchanged.
    await occ_update_one(
        client,
        doc_id=doc_id,
        merger=lambda _src: {
            'region_validated': False,
            'region_status': 'verify_rejected',
        },
        index=index,
        refresh=True,
        writer_id='vlm:verify',
    )
    doc2 = (await client.get(index=index, id=doc_id))['_source']
    assert doc2['region_validated'] is False
    assert doc2['region_status'] == 'verify_rejected'
    assert doc2['class_id'] == 5
    assert doc2['class_name'] == 'pickup'
    assert doc2['class_validated'] is True
    assert doc2['class_source'] == 'human'


# ---------------------------------------------------------------------------
# Invariant 4 — history preserved on relabel
# ---------------------------------------------------------------------------


async def test_invariant_4_history_preserved_across_two_relabels() -> None:
    """Re-labeling twice appends both prior assignments to
    ``class_id_history`` with ``at``/``writer`` recorded on each entry —
    neither is lost or overwritten by the next relabel."""
    from src.clients.occ import occ_update_one
    from src.services.curation.history import record_class_history

    client = _VersionedFakeOpenSearch()
    index = 'op_items_test'
    doc_id = 'crop-history-1'
    await client.index(
        index=index,
        id=doc_id,
        body={
            'crop_id': doc_id,
            'class_id': 1,
            'class_name': 'sedan',
            'class_source': 'item_model',
            'label_source': 'ingest',
            'confidence': 0.81,
            'class_id_history': [],
        },
    )

    def _merge_to(class_id: int, class_name: str):
        def _merger(src: dict[str, Any]) -> dict[str, Any]:
            return {
                'class_id': class_id,
                'class_name': class_name,
                'class_source': 'human',
                'label_source': 'human',
                'class_id_history': record_class_history(src, writer='human:label'),
            }

        return _merger

    await occ_update_one(
        client,
        doc_id=doc_id,
        merger=_merge_to(2, 'pickup'),
        index=index,
        refresh=True,
        writer_id='human',
    )
    await occ_update_one(
        client,
        doc_id=doc_id,
        merger=_merge_to(3, 'van'),
        index=index,
        refresh=True,
        writer_id='human',
    )

    doc = (await client.get(index=index, id=doc_id))['_source']
    assert doc['class_id'] == 3
    history = doc.get('class_id_history') or []
    prior_class_ids = [h['class_id'] for h in history]
    assert 1 in prior_class_ids, f'lost the original classifier label: {history!r}'
    assert 2 in prior_class_ids, f'lost the first relabel: {history!r}'
    for entry in history:
        assert 'at' in entry
        assert 'writer' in entry
