"""The run harness against a fake HTTP API (httpx.MockTransport): no network, no stack."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import httpx
import pytest

from scripts.bench import run_baseline as rb, select_baseline_set as sel


if TYPE_CHECKING:
    from pathlib import Path

PROM = """\
# TYPE op_pipeline_stage_seconds histogram
op_pipeline_stage_seconds_sum{{stage="decode"}} {sum}
op_pipeline_stage_seconds_count{{stage="decode"}} {count}
"""


class FakeStack:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.policy: dict | None = None
        self.metric_reads = 0
        self.drain_polls = 0
        self.autolabel_polls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:  # noqa: PLR0911
        path = request.url.path
        body = request.content.decode()
        self.calls.append((request.method, path, str(request.url.query, 'utf-8') + body))
        if path == '/metrics':
            self.metric_reads += 1
            n = 0 if self.metric_reads == 1 else 8
            return httpx.Response(200, text=PROM.format(sum=n * 0.5, count=n))
        if request.method == 'POST' and path == '/curation/projects':
            return httpx.Response(201, json={'project': {'slug': json.loads(body)['slug']}})
        m = re.fullmatch(r'/curation/projects/([^/]+)', path)
        if m and request.method == 'GET':
            return httpx.Response(
                200,
                json={'resources': {'indexes': {'items': 'op_p__items', 'images': 'op_p__images'}}},
            )
        if path.endswith('/ingest/policy') and request.method == 'GET':
            return httpx.Response(200, json={'revision': 3})
        if path.endswith('/ingest/policy') and request.method == 'PUT':
            self.policy = json.loads(body)
            return httpx.Response(200, json={'revision': 4})
        if path.endswith('/ingest/batch'):
            n = len(json.loads(body)['items'])
            return httpx.Response(
                200,
                json={
                    'status': 'success',
                    'summary': {
                        'successful': n,
                        'duplicates': 0,
                        'failed': 0,
                        'crops_indexed': 2 * n,
                        'n_embedded': 2 * n,
                    },
                    'results': [],
                },
            )
        if path.endswith('/ingest/region_drain'):
            self.drain_polls += 1
            return httpx.Response(
                200,
                json={
                    'drained': self.drain_polls >= 3,
                    'total_unfinished': 0 if self.drain_polls >= 3 else 5,
                },
            )
        if path.endswith('/pipeline/auto_label/start'):
            return httpx.Response(200, json={'job_id': 'j1', 'status': 'queued'})
        if path.endswith('/pipeline/auto_label/status/j1'):
            self.autolabel_polls += 1
            done = self.autolabel_polls >= 2
            return httpx.Response(
                200,
                json={
                    'status': 'completed' if done else 'running',
                    'stage_durations': {'cluster_residuals': 4.5} if done else {},
                },
            )
        if path == '/op_p__items,op_p__images/_refresh' or path.endswith('/_refresh'):
            return httpx.Response(200, json={})
        if path.endswith('/_forcemerge'):
            return httpx.Response(200, json={})
        if path.endswith('/_stats/store'):
            return httpx.Response(
                200, json={'_all': {'total': {'store': {'size_in_bytes': 1_000_000}}}}
            )
        return httpx.Response(404, json={'detail': path})


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    from PIL import Image

    cands = []
    for i in range(10):
        p = tmp_path / f'{i}.jpg'
        Image.new('RGB', (400, 400)).save(p, 'JPEG')
        cands.append(sel.Candidate(p, p.stat().st_size, 'unit'))
    out = tmp_path / 'm.txt'
    sel.write_manifest(out, cands, seed=1, today='2026-10-03')
    return out


def _args(manifest: Path, tmp_path: Path, extra: list[str] | None = None):
    return rb.build_parser().parse_args(
        [
            str(manifest),
            '--api-url',
            'http://api.test',
            '--slug-prefix',
            'bl',
            '--policy',
            'all',
            '--batch-size',
            '4',
            '--warmup',
            '4',
            '--out',
            str(tmp_path / 'report'),
            '--stages',
            'ingest,region,cluster',
            '--opensearch-url',
            'http://os.test',
            *(extra or []),
        ]
    )


def _run(args, stack: FakeStack):
    ticks = iter(x * 0.5 for x in range(10_000))
    return rb.run(
        args,
        transport=httpx.MockTransport(stack.handler),
        sleep=lambda _s: None,
        clock=lambda: next(ticks),
        gpu_reader=lambda: None,
    )


def test_full_run_report_math_and_calls(manifest: Path, tmp_path: Path) -> None:
    stack = FakeStack()
    report = _run(_args(manifest, tmp_path), stack)

    assert report['ingest']['images'] == 6  # 10 images minus the 4 warmup
    assert report['ingest']['ok'] == 6
    assert report['ingest']['items'] == 12
    assert report['ingest']['embedded'] == 12
    assert report['ingest']['images_per_s'] > 0
    assert report['manifest']['count'] == 10
    assert report['manifest']['name'] == 'm.txt'
    delta = report['metrics']['api']['histograms']['op_pipeline_stage_seconds{stage=decode}']
    assert delta['count'] == 8
    assert report['stages']['region_drain']['wall_s'] > 0
    assert report['stages']['cluster']['wall_s'] == 4.5
    assert report['storage']['store_bytes'] == 1_000_000
    assert report['storage']['store_bytes_per_image'] == 100_000.0
    assert report['gpu'] == {}

    paths = [c[1] for c in stack.calls]
    assert '/metrics' in paths
    created = next(c for c in stack.calls if c[0] == 'POST' and c[1] == '/curation/projects')
    assert json.loads(created[2])['slug'].startswith('bl-')
    assert stack.policy is not None
    assert stack.policy['embedding'] == {'mode': 'all', 'classes': []}
    assert stack.policy['expected_revision'] == 3
    batches = [c for c in stack.calls if c[1].endswith('/ingest/batch')]
    assert len(batches) == 3  # 4 warmup + 4 + 2
    start = next(c for c in stack.calls if c[1].endswith('/pipeline/auto_label/start'))
    assert 'train_clusters=true' in start[2]
    assert 'run_vlm=false' in start[2]


def test_outputs_contain_no_image_paths(manifest: Path, tmp_path: Path) -> None:
    _run(_args(manifest, tmp_path), FakeStack())
    for suffix in ('.json', '.md'):
        text = (tmp_path / f'report{suffix}').read_text()
        assert str(tmp_path / '0.jpg') not in text
        assert str(tmp_path) not in text


def test_only_requested_stages_run(manifest: Path, tmp_path: Path) -> None:
    stack = FakeStack()
    args = _args(manifest, tmp_path, ['--stages', 'ingest'])
    _run(args, stack)
    assert not any('region_drain' in c[1] or 'auto_label' in c[1] for c in stack.calls)


def test_path_map_rewrites_host_paths_for_the_container(manifest: Path, tmp_path: Path) -> None:
    stack = FakeStack()
    args = _args(manifest, tmp_path, ['--path-map', f'{tmp_path}=/data', '--stages', 'ingest'])
    _run(args, stack)
    first = next(c for c in stack.calls if c[1].endswith('/ingest/batch'))
    items = json.loads(first[2].split('{', 1)[0] + '{' + first[2].split('{', 1)[1])['items']
    assert all(i['path'].startswith('/data/') for i in items)


def test_selected_policy_requires_classes(manifest: Path, tmp_path: Path) -> None:
    args = _args(manifest, tmp_path, ['--policy', 'selected'])
    with pytest.raises(SystemExit):
        _run(args, FakeStack())


def test_tampered_manifest_is_rejected(manifest: Path, tmp_path: Path) -> None:
    lines = manifest.read_text().splitlines()
    i = next(n for n, line in enumerate(lines) if not line.startswith('#'))
    lines[i] = lines[i] + 'x'
    manifest.write_text('\n'.join(lines) + '\n')
    with pytest.raises(sel.ManifestError):
        _run(_args(manifest, tmp_path), FakeStack())


def test_gpu_sampler_skips_gracefully_without_nvidia_smi() -> None:
    assert rb.read_nvidia_smi(runner=_raise_missing) is None


def _raise_missing(*_a: object, **_k: object) -> str:
    raise FileNotFoundError('nvidia-smi')


def test_gpu_sampler_collects_samples() -> None:
    readings = iter(['0, 10, 100\n', '0, 30, 300\n', '0, 50, 200\n'])
    sampler = rb.GpuSampler(lambda: rb.parse_nvidia_smi(next(readings)), interval=0.0)
    for _ in range(3):
        sampler.sample_once()
    assert sampler.summary()['0']['mem_peak_mb'] == 300.0


def test_compare_cli_prints_delta_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps({'ingest': {'images_per_s': 10.0}}))
    b.write_text(json.dumps({'ingest': {'images_per_s': 20.0}}))
    assert rb.main(['--compare', str(a), str(b)]) == 0
    assert '| ingest.images_per_s | 10 | 20 | +10 | +100.0% |' in capsys.readouterr().out
