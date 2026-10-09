"""Offline scoring of the registry-prior on/off comparison (#61 item 2)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / 'scripts/curation/bakeoff/vlm_prior_oracle.py'
_spec = importlib.util.spec_from_file_location('vlm_prior_oracle', _PATH)
assert _spec is not None
assert _spec.loader is not None
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def _rec(truth: str, answer: str) -> dict[str, str]:
    return {'truth': truth, 'answer': answer}


def test_score_counts_unanswered_as_wrong_overall_only() -> None:
    s = oracle.score([_rec('a', 'a'), _rec('b', 'x'), _rec('c', '')])
    assert s['n'] == 3
    assert s['answered'] == 2
    assert s['overall_accuracy'] == pytest.approx(1 / 3)
    assert s['answered_accuracy'] == pytest.approx(1 / 2)


def test_compare_reports_the_delta_over_shared_crops() -> None:
    off = {'1': _rec('a', ''), '2': _rec('b', 'b'), '3': _rec('c', 'x')}
    on = {'1': _rec('a', 'a'), '2': _rec('b', 'b'), '9': _rec('z', 'z')}
    r = oracle.compare(off, on)
    assert r['off']['n'] == r['on']['n'] == 2
    assert r['overall_accuracy_delta'] == pytest.approx(0.5)
    assert r['dropped_not_in_both'] == 2


def test_compare_refuses_disjoint_or_inconsistent_oracles() -> None:
    with pytest.raises(ValueError, match='share no crop'):
        oracle.compare({'1': _rec('a', 'a')}, {'2': _rec('a', 'a')})
    with pytest.raises(ValueError, match='oracle names differ'):
        oracle.compare({'1': _rec('a', 'a')}, {'1': _rec('b', 'a')})


def test_load_records_round_trip(tmp_path: Path) -> None:
    f = tmp_path / 'r.jsonl'
    f.write_text(json.dumps({'crop_id': 'c1', 'truth': 'a', 'answer': None}) + '\n\n')
    assert oracle.load_records(f) == {'c1': _rec('a', '')}


def test_dump_binds_the_project_without_a_nested_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``dump`` runs under ``asyncio.run``; the sync binder starts its own loop and
    crashed there on a real stack. The async binder must be the one used."""
    from src.services.projects import guard, script_binding

    bound: list[str] = []

    async def _abind(slug: str, *, opensearch_url: str | None = None) -> None:
        bound.append(slug)

    class _Client:
        async def search(self, **_kw: object) -> dict[str, object]:
            return {'hits': {'hits': []}}

        async def close(self) -> None:
            return None

    def _sync_bind(*_a: object, **_kw: object) -> None:
        raise AssertionError('sync bind_script_project must not be used inside a loop')

    monkeypatch.setattr(script_binding, 'abind_script_project', _abind)
    monkeypatch.setattr(script_binding, 'bind_script_project', _sync_bind)
    monkeypatch.setattr(guard, 'make_script_opensearch', lambda *_a, **_k: _Client())
    args = oracle.build_parser().parse_args(
        [
            'dump',
            '--project',
            'p',
            '--pack',
            'k',
            '--truth-field',
            't',
            '--out',
            str(tmp_path / 'o'),
        ]
    )
    assert asyncio.run(oracle._dump(args)) == 1  # no records written
    assert bound == ['p']
