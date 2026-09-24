"""Restoring region detector provenance lost to a same-box human confirm
(:mod:`src.services.curation.region_provenance_restore` and
``scripts/curation/restore_region_provenance.py``), against the
query-evaluating OpenSearch fake."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_region_fields
from src.services.curation.edit_history import EDIT_HISTORY_FIELD, EditKind, record_edit
from src.services.curation.region_provenance_restore import (
    SOURCE_BACKUP,
    SOURCE_CHAIN,
    SOURCE_HISTORY,
    apply_restores,
    plan_restores,
)
from src.services.detection.profile_registry import region_profile_or_neutral


F = get_region_fields()
ITEMS = 'test_items'
BACKUP = 'test_items_backup'
HUMAN = region_profile_or_neutral().human_detector_name
BOX = [0.643, 0.5797, 0.6966, 0.6243]
CHAIN = ['det_model::hit@2026-09-24T03:08:19+00:00', 'det_model::combined_verify_ok@2026']


def _model_state(**extra: Any) -> dict[str, Any]:
    return {
        F.bbox_norm: list(BOX),
        F.detector: 'det_model',
        F.detector_version: '1',
        F.score: 0.89404296875,
        F.detected_at: '2026-09-24T03:08:19.221513+00:00',
        F.detector_chain: list(CHAIN),
        **extra,
    }


def _confirmed(crop_id: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        F.bbox_norm: list(BOX),
        F.detector: HUMAN,
        F.detector_version: '1',
        F.score: 1.0,
        F.detected_at: '2026-09-24T10:54:26+00:00',
        F.detector_chain: list(CHAIN),
        F.status: 'detected',
        **extra,
    }


@pytest.fixture
def fake() -> QueryFakeOpenSearch:
    with_history = _confirmed('hist-1')
    with_history[EDIT_HISTORY_FIELD] = record_edit(
        {'crop_id': 'hist-1', **_model_state()},
        kind=EditKind.REGION,
        writer='human:set_crop_region',
    )
    return QueryFakeOpenSearch(
        {
            ITEMS: {
                'hist-1': with_history,
                'bak-1': _confirmed('bak-1'),
                'redrawn': _confirmed('redrawn'),
                'chain-only': _confirmed('chain-only'),
                'human-drawn': _confirmed('human-drawn', **{F.detector_chain: None}),
                'model': {'crop_id': 'model', **_model_state()},
            },
            BACKUP: {
                'bak-1': {'crop_id': 'bak-1', **_model_state()},
                'redrawn': {
                    'crop_id': 'redrawn',
                    **_model_state(**{F.bbox_norm: [0, 0, 0.1, 0.1]}),
                },
            },
        }
    )


@pytest.mark.asyncio
async def test_plans_pick_the_right_source(fake: QueryFakeOpenSearch) -> None:
    plans = {p.crop_id: p for p in await plan_restores(fake, index=ITEMS, backup_index=BACKUP)}
    assert set(plans) == {'hist-1', 'bak-1', 'redrawn', 'chain-only'}
    assert (plans['hist-1'].source, plans['hist-1'].applicable) == (SOURCE_HISTORY, True)
    assert (plans['bak-1'].source, plans['bak-1'].applicable) == (SOURCE_BACKUP, True)
    # The backup's box differs: a human redraw, not a same-box confirm.
    assert (plans['redrawn'].source, plans['redrawn'].applicable) == (SOURCE_CHAIN, False)
    chain = plans['chain-only']
    assert (chain.source, chain.applicable) == (SOURCE_CHAIN, False)
    assert chain.restore[F.detector] == 'det_model'
    assert chain.restore[F.score] is None
    for cid in ('hist-1', 'bak-1'):
        assert plans[cid].restore[F.score] == 0.89404296875
        assert plans[cid].restore[F.detector] == 'det_model'


@pytest.mark.asyncio
async def test_apply_restores_only_verified_plans(fake: QueryFakeOpenSearch) -> None:
    plans = await plan_restores(fake, index=ITEMS, backup_index=BACKUP)
    counts = await apply_restores(fake, plans, index=ITEMS)
    assert counts == {'restored': 2, 'skipped_changed': 0, 'not_applicable': 2, 'errors': 0}
    docs = fake.docs(ITEMS)
    for cid in ('hist-1', 'bak-1'):
        assert docs[cid][F.detector] == 'det_model'
        assert docs[cid][F.score] == 0.89404296875
        assert docs[cid][F.detected_at] == '2026-09-24T03:08:19.221513+00:00'
        assert docs[cid][EDIT_HISTORY_FIELD][-1]['state'][F.detector] == HUMAN
    assert docs['chain-only'][F.detector] == HUMAN
    assert docs['redrawn'][F.score] == 1.0


@pytest.mark.asyncio
async def test_apply_skips_items_changed_since_planning(fake: QueryFakeOpenSearch) -> None:
    plans = await plan_restores(fake, index=ITEMS, backup_index=BACKUP)
    fake.docs(ITEMS)['bak-1'][F.bbox_norm] = [0.1, 0.1, 0.2, 0.2]
    counts = await apply_restores(fake, plans, index=ITEMS)
    assert counts['skipped_changed'] == 1
    assert fake.docs(ITEMS)['bak-1'][F.detector] == HUMAN


@pytest.mark.asyncio
async def test_prefix_filter(fake: QueryFakeOpenSearch) -> None:
    plans = await plan_restores(fake, index=ITEMS, backup_index=BACKUP, id_prefix='bak')
    assert [p.crop_id for p in plans] == ['bak-1']


def _load_script() -> Any:
    path = Path(__file__).resolve().parents[2] / 'scripts/curation/restore_region_provenance.py'
    spec = importlib.util.spec_from_file_location('restore_region_provenance', path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules['restore_region_provenance'] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_script_dry_run_is_read_only(
    fake: QueryFakeOpenSearch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_script()
    args = mod.build_parser().parse_args(['--index', ITEMS, '--backup-index', BACKUP])
    assert isinstance(args, argparse.Namespace)
    assert args.dry_run
    before = {k: dict(v) for k, v in fake.docs(ITEMS).items()}
    assert await mod.run(args, fake) == 0
    assert fake.docs(ITEMS) == before
    out = capsys.readouterr().out
    assert '4 candidate(s); 2 restorable, 2 report-only' in out
    assert 'WILL RESTORE' in out
