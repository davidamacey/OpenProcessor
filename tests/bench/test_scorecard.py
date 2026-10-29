"""scorecard: envelope schema, acceptance pass/fail, the synthetic refusal, run metadata."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.bench import scorecard as sc


if TYPE_CHECKING:
    from pathlib import Path

REAL = {'name': 'coco-3000', 'synthetic': False, 'split': 'heldout', 'manifest_sha256': 'ab' * 32}
META = {'git_sha': 'deadbeef', 'seed': 1}


def _card(**over: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        'kind': 'K2',
        'level': 'items',
        'dataset': REAL,
        'run_meta': META,
        'metrics': {'bcubed_precision': 0.91},
    }
    args.update(over)
    return sc.build(**args)


def test_envelope_has_exactly_the_planned_keys() -> None:
    card = _card(strata={'low_res': {'bcubed_precision': 0.86}})
    assert set(card) == {
        'schema_version',
        'kind',
        'level',
        'dataset',
        'run_meta',
        'metrics',
        'strata',
        'acceptance',
    }
    assert card['schema_version'] == sc.SCHEMA_VERSION
    assert card['acceptance'] is None


def test_acceptance_passes_and_fails_on_the_bar() -> None:
    ok = _card(acceptance=sc.acceptance('bcubed_precision', 0.90, 0.91))['acceptance']
    assert ok == {'metric': 'bcubed_precision', 'bar': 0.90, 'value': 0.91, 'pass': True}
    bad = _card(acceptance=sc.acceptance('bcubed_precision', 0.90, 0.89))['acceptance']
    assert bad['pass'] is False
    edge = sc.acceptance('bcubed_precision', 0.90, 0.90)
    assert edge['pass'] is True  # the bar is inclusive


def test_acceptance_can_be_a_ceiling() -> None:
    assert sc.acceptance('peak_rss_gb', 2.0, 1.5, higher_is_better=False)['pass'] is True
    assert sc.acceptance('peak_rss_gb', 2.0, 2.5, higher_is_better=False)['pass'] is False


def test_a_missing_value_cannot_pass() -> None:
    assert sc.acceptance('x', 0.9, None)['pass'] is False


def test_acceptance_is_refused_off_the_heldout_half() -> None:
    for split in ('calibration', 'full'):
        with pytest.raises(sc.ScorecardError, match='heldout'):
            _card(
                dataset={**REAL, 'split': split},
                acceptance=sc.acceptance('bcubed_precision', 0.9, 0.95),
            )


def test_synthetic_dataset_refuses_quality_fields() -> None:
    synth = {'name': 'planted-1m', 'synthetic': True, 'split': None, 'manifest_sha256': None}
    ok = _card(dataset=synth, metrics={'wall_s': 3.2, 'peak_rss_mb': 900.0, 'p95_ms': 12.0})
    assert ok['metrics']['wall_s'] == 3.2
    for quality in ('bcubed_precision', 'pair_recall', 'recall_at_10', 'ami', 'ari'):
        with pytest.raises(sc.ScorecardError, match='synthetic'):
            _card(dataset=synth, metrics={'wall_s': 1.0, quality: 0.99})


def test_synthetic_dataset_refuses_strata_and_acceptance() -> None:
    synth = {'name': 'planted-1m', 'synthetic': True, 'split': None, 'manifest_sha256': None}
    with pytest.raises(sc.ScorecardError, match='synthetic'):
        _card(dataset=synth, metrics={'wall_s': 1.0}, strata={'a': {'wall_s': 1.0}})
    with pytest.raises(sc.ScorecardError, match='synthetic'):
        _card(
            dataset=synth,
            metrics={'wall_s': 1.0},
            acceptance=sc.acceptance('wall_s', 5.0, 1.0, higher_is_better=False),
        )


def test_real_dataset_needs_a_manifest_hash_and_a_known_split() -> None:
    with pytest.raises(sc.ScorecardError, match='manifest_sha256'):
        _card(dataset={**REAL, 'manifest_sha256': None})
    with pytest.raises(sc.ScorecardError, match='split'):
        _card(dataset={**REAL, 'split': 'train'})
    with pytest.raises(sc.ScorecardError, match='synthetic'):
        _card(dataset={'name': 'x', 'manifest_sha256': 'a', 'split': 'heldout'})


def test_tuning_on_the_scoring_split_is_refused() -> None:
    tuned = {**REAL, 'tuning': [{'manifest_sha256': REAL['manifest_sha256'], 'split': 'heldout'}]}
    with pytest.raises(sc.ScorecardError, match='tuned'):
        _card(dataset=tuned)
    fine = {
        **REAL,
        'tuning': [{'manifest_sha256': REAL['manifest_sha256'], 'split': 'calibration'}],
    }
    assert _card(dataset=fine)['dataset']['tuning'][0]['split'] == 'calibration'


def test_run_meta_requires_a_git_sha_and_a_seed() -> None:
    with pytest.raises(sc.ScorecardError, match='git_sha'):
        _card(run_meta={'seed': 1})
    with pytest.raises(sc.ScorecardError, match='seed'):
        _card(run_meta={'git_sha': 'x'})


def test_write_round_trips_json(tmp_path: Path) -> None:
    card = _card()
    out = tmp_path / 'sub' / 'card.json'
    sc.write(out, card)
    assert json.loads(out.read_text()) == card


def test_run_meter_records_load_wall_rss_and_flags_a_noisy_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = iter([(0.5, 0, 0), (20.0, 0, 0)])
    monkeypatch.setattr(sc.os, 'getloadavg', lambda: next(loads))
    monkeypatch.setattr(sc, '_nvidia_smi', lambda: 'GPU 2: idle')
    with sc.RunMeter(seed=3, gpu_id=2, embedding_field='embedding', git_sha='abc') as meter:
        pass
    meta = meter.meta
    assert meta['git_sha'] == 'abc'
    assert meta['seed'] == 3
    assert meta['gpu_id'] == 2
    assert meta['embedding_field'] == 'embedding'
    assert meta['load_avg_start'] == 0.5
    assert meta['load_avg_end'] == 20.0
    assert meta['noisy'] is True  # a 1-minute load above 16 at either end
    assert meta['nvidia_smi'] == 'GPU 2: idle'
    assert meta['wall_s'] >= 0
    assert meta['peak_rss_mb'] > 0
    sc.build('K2', 'items', REAL, meta, {'bcubed_precision': 0.9})


def test_run_meter_without_nvidia_smi_records_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc.shutil, 'which', lambda _name: None)
    with sc.RunMeter(seed=1, git_sha='x') as meter:
        pass
    assert meter.meta['nvidia_smi'] is None
    assert meter.meta['noisy'] is False or isinstance(meter.meta['noisy'], bool)
