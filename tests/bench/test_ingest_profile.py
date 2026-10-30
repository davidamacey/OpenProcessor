"""The profile driver (``scripts/bench/ingest_profile.py``): protocol values and the observer."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest

from scripts.bench import ingest_profile as ip


if TYPE_CHECKING:
    from pathlib import Path


def _ns(tmp_path: Path, manifest: Path, **over: object):
    argv = [
        'run', 'setB', str(manifest), '--api-url', 'http://api.test',
        '--api-container', 'a', '--triton-container', 't', '--out-root', str(tmp_path),
        '--triton-metrics-url', 'http://tr.test/metrics', '--path-map', '/h=/c',
    ]  # fmt: skip
    return ip._parser().parse_args(argv)


def test_the_protocol_values_are_fixed(manifest: Path, tmp_path: Path) -> None:
    args = ip.build_run_args(_ns(tmp_path, manifest))
    assert args.reps == 3
    assert args.gpu_interval == 0.5
    assert args.drop_cache is True
    assert args.stages == 'ingest'
    assert args.out == tmp_path / 'setB' / 'setB'
    assert args.slug_prefix == 'sp0-setb'  # project slugs are lower case
    assert args.path_map == ['/h=/c']
    assert args.triton_metrics_url == 'http://tr.test/metrics'


def test_observer_reports_api_cpu_delta_and_mean_triton_cpu() -> None:
    usage = iter([2_000_000, 7_500_000])

    def runner(*argv: str) -> str:
        if argv[0] == 'exec':
            return f'user_usec 1\nusage_usec {next(usage)}\nsystem_usec 2\n'
        return '200.00%\n'

    obs = ip.ContainerObserver('a', 't', runner=runner, every=0.01)
    obs.start()
    obs.sample_triton()  # at least one sample regardless of the sampler thread's timing
    out = obs.stop()
    assert out['api_cpu_s'] == pytest.approx(5.5)
    assert out['triton_cpu_pct_mean'] == 200.0
    assert out['triton_cpu_samples'] >= 1.0


def test_a_cgroup_file_without_usage_is_an_error() -> None:
    obs = ip.ContainerObserver('a', 't', runner=lambda *_a: 'nr_periods 1\n', every=3600.0)
    with pytest.raises(ValueError, match='no usage_usec'):
        obs.start()


def test_a_failed_triton_sample_is_left_out_of_the_mean() -> None:
    def runner(*argv: str) -> str:
        if argv[0] == 'exec':
            return 'usage_usec 1\n'
        raise subprocess.CalledProcessError(1, 'docker')

    obs = ip.ContainerObserver('a', 't', runner=runner, every=0.01)
    obs.start()
    out = obs.stop()
    assert 'triton_cpu_pct_mean' not in out


def test_table_prints_the_row(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from tests.bench.test_ingest_profile_report import _report

    row = tmp_path / 'setB'
    row.mkdir()
    for i in (1, 2, 3):
        (row / f'setB-rep{i}.json').write_text(json.dumps(_report(30.0 + i)))
    assert ip.main(['table', f'set B={row}']) == 0
    assert '### set B' in capsys.readouterr().out
