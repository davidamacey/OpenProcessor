"""Repetitions, page-cache drop, load average and the ``gpu_image_prep`` stage block (SP0)."""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

import httpx
import pytest

from scripts.bench import baseline_report as rep, run_baseline as rb
from tests.bench.test_run_baseline import FakeStack, _args


if TYPE_CHECKING:
    from pathlib import Path

TRITON = """\
# HELP op_gpu_prep_stage_seconds gpu_image_prep stage wall seconds
# TYPE op_gpu_prep_stage_seconds counter
op_gpu_prep_stage_seconds{{stage="decode"}} {decode}
op_gpu_prep_stage_seconds{{stage="crop_jpeg"}} {jpeg}
# HELP op_gpu_prep_stage_observations images that ran a gpu_image_prep stage
# TYPE op_gpu_prep_stage_observations counter
op_gpu_prep_stage_observations{{stage="decode"}} {n}
op_gpu_prep_stage_observations{{stage="crop_jpeg"}} {n}
"""


class TritonStack(FakeStack):
    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == '/triton-metrics':
            self.triton_reads = getattr(self, 'triton_reads', 0) + 1
            k = 0 if self.triton_reads % 2 == 1 else 1  # odd read = before, even = after
            return httpx.Response(200, text=TRITON.format(decode=3.0 * k, jpeg=1.0 * k, n=6 * k))
        return super().handler(request)


def _reps(args, stack, *, drops, loads, gpu=lambda: None):
    ticks = iter(x * 0.5 for x in range(100_000))
    return rb.run_reps(
        args,
        transport=httpx.MockTransport(stack.handler),
        sleep=lambda _s: None,
        clock=lambda: next(ticks),
        gpu_reader=gpu,
        drop_cache=lambda paths: drops.append(list(paths)),
        loadavg=lambda: loads.pop(0),
    )


def test_every_protocol_row_name_makes_a_valid_project_slug(manifest: Path, tmp_path: Path) -> None:
    from src.config.projects import PROJECT_SLUG_MAX_LEN, PROJECT_SLUG_RE

    for row in ('setA', 'setB', 'setP3000', 'setB-sync', 'setP3000-sync', 'setB-nometrics'):
        stack = FakeStack()
        args = _args(manifest, tmp_path, ['--reps', '3', '--slug-prefix', f'sp0-{row.lower()}'])
        slug = _reps(args, stack, drops=[], loads=[(1.0, 1.0, 1.0)] * 6)[0]['config']['project']
        assert re.fullmatch(PROJECT_SLUG_RE, slug), slug
        assert len(slug) <= PROJECT_SLUG_MAX_LEN, slug


def test_each_rep_gets_its_own_project_and_report_file(manifest: Path, tmp_path: Path) -> None:
    stack = FakeStack()
    args = _args(manifest, tmp_path, ['--reps', '3'])
    reports = _reps(args, stack, drops=[], loads=[(1.0, 1.0, 1.0)] * 6)

    slugs = [r['config']['project'] for r in reports]
    assert len(set(slugs)) == 3
    created = [
        json.loads(c[2])['slug']
        for c in stack.calls
        if c[1] == '/curation/projects' and c[0] == 'POST'
    ]
    assert created == slugs
    for i in (1, 2, 3):
        assert (tmp_path / f'report-rep{i}.json').is_file()
    assert not (tmp_path / 'report.json').exists()


def test_the_page_cache_is_dropped_once_per_rep_over_the_manifest(
    manifest: Path, tmp_path: Path
) -> None:
    drops: list[list[str]] = []
    args = _args(manifest, tmp_path, ['--reps', '2', '--drop-cache'])
    _reps(args, FakeStack(), drops=drops, loads=[(1.0, 1.0, 1.0)] * 4)
    assert len(drops) == 2
    assert drops[0] == rb.read_manifest(manifest)


def test_no_drop_without_the_flag(manifest: Path, tmp_path: Path) -> None:
    drops: list[list[str]] = []
    _reps(
        _args(manifest, tmp_path, ['--reps', '2']), FakeStack(), drops=drops, loads=[(1.0,) * 3] * 4
    )
    assert drops == []


def test_load_average_start_end_and_the_noisy_mark(manifest: Path, tmp_path: Path) -> None:
    loads = [(3.0, 2.0, 1.0), (4.0, 2.5, 1.5), (17.5, 9.0, 5.0), (12.0, 9.0, 5.0)]
    args = _args(manifest, tmp_path, ['--reps', '2'])
    quiet, noisy = _reps(args, FakeStack(), drops=[], loads=loads)
    assert quiet['host'] == {
        'loadavg_start': [3.0, 2.0, 1.0],
        'loadavg_end': [4.0, 2.5, 1.5],
        'noisy': False,
    }
    assert noisy['host']['noisy'] is True  # the 1-minute average at the start was above 16


def test_the_summary_has_the_median_and_range(manifest: Path, tmp_path: Path) -> None:
    args = _args(manifest, tmp_path, ['--reps', '3'])
    reports = _reps(args, FakeStack(), drops=[], loads=[(1.0, 1.0, 1.0)] * 6)
    summary = json.loads((tmp_path / 'report-summary.json').read_text())
    rates = sorted(r['ingest']['images_per_s'] for r in reports)
    assert summary['reps'] == 3
    assert summary['images_per_s'] == {'median': rates[1], 'min': rates[0], 'max': rates[2]}
    assert summary['noisy_reps'] == []


def test_gpu_prep_stage_block_from_the_triton_metrics_delta(manifest: Path, tmp_path: Path) -> None:
    args = _args(
        manifest,
        tmp_path,
        ['--triton-metrics-url', 'http://api.test/triton-metrics', '--reps', '1'],
    )
    (report,) = _reps(args, TritonStack(), drops=[], loads=[(1.0, 1.0, 1.0)] * 2)
    stages = report['gpu_prep_stages']
    assert stages['decode'] == {'seconds': 3.0, 'observations': 6.0, 'mean_s': 0.5, 'share': 0.75}
    assert stages['crop_jpeg']['share'] == 0.25
    assert 'gpu_prep_stages' in (tmp_path / 'report.json').read_text()
    assert '### gpu_image_prep stages' in (tmp_path / 'report.md').read_text()


def test_gpu_prep_stages_is_empty_without_the_series() -> None:
    assert rep.gpu_prep_stages({'nv_inference_count{model=x}': 5.0}) == {}


def test_gpu_util_summary_has_the_median(manifest: Path) -> None:
    summary = rep.summarize_gpu([{'0': (10.0, 1.0)}, {'0': (90.0, 1.0)}, {'0': (20.0, 1.0)}])
    assert summary['0']['util_p50_pct'] == 20.0


def test_drop_page_cache_advises_every_file(tmp_path: Path) -> None:
    files = []
    for i in range(3):
        f = tmp_path / f'{i}.bin'
        f.write_bytes(b'x' * 10)
        files.append(str(f))
    calls: list[tuple[int, int, int, int]] = []
    advised = rb.drop_page_cache(files, advise=lambda *a: calls.append(a))
    assert advised == 3
    assert {c[3] for c in calls} == {os.POSIX_FADV_DONTNEED}
    assert all(c[1:3] == (0, 0) for c in calls)  # whole file


def test_drop_page_cache_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rb.drop_page_cache([str(tmp_path / 'gone.jpg')])


def test_read_loadavg_parses_proc_format(tmp_path: Path) -> None:
    f = tmp_path / 'loadavg'
    f.write_text('7.50 6.25 5.00 3/1200 4242\n')
    assert rb.read_loadavg(f) == (7.5, 6.25, 5.0)


class _Observer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append('start')

    def stop(self) -> dict[str, float]:
        self.calls.append('stop')
        return {'api_cpu_s': 4.0}


def test_an_observer_wraps_each_rep_and_lands_in_the_host_block(
    manifest: Path, tmp_path: Path
) -> None:
    observer = _Observer()
    ticks = iter(x * 0.5 for x in range(100_000))
    reports = rb.run_reps(
        _args(manifest, tmp_path, ['--reps', '2']),
        transport=httpx.MockTransport(FakeStack().handler),
        sleep=lambda _s: None,
        clock=lambda: next(ticks),
        gpu_reader=lambda: None,
        loadavg=lambda: (1.0, 1.0, 1.0),
        observer=observer,
    )
    assert observer.calls == ['start', 'stop', 'start', 'stop']
    assert all(r['host']['api_cpu_s'] == 4.0 for r in reports)
