"""Tests for the pure helpers in scripts/bench/baseline_suite.py."""

from __future__ import annotations

from scripts.bench import baseline_suite as bs, suite_ingest, suite_probes
from scripts.bench.baseline_report import metrics_delta


def test_stage_table_joins_seconds_and_bytes() -> None:
    before: dict[str, float] = {}
    after = {
        'op_pipeline_stage_seconds_sum{stage=decode}': 4.0,
        'op_pipeline_stage_seconds_count{stage=decode}': 8.0,
        'op_pipeline_stage_bytes_total{stage=decode}': 800.0,
        'op_pipeline_stage_seconds_sum{stage=embed}': 3.0,
        'op_pipeline_stage_seconds_count{stage=embed}': 2.0,
    }
    table = suite_probes.stage_table(metrics_delta(before, after, ('op_',)))
    assert table['decode'] == {'calls': 8.0, 'seconds': 4.0, 'mean_ms': 500.0, 'bytes': 800.0}
    assert table['embed']['mean_ms'] == 1500.0


def test_knn_fields_finds_top_level_and_nested_vectors() -> None:
    props = {
        'pe_embedding': {'type': 'knn_vector'},
        'meta': {'type': 'keyword'},
        'region_box_embeddings': {
            'type': 'nested',
            'properties': {'embedding': {'type': 'knn_vector'}, 'box_id': {'type': 'keyword'}},
        },
    }
    assert suite_ingest._knn_fields(props) == [
        ('pe_embedding', {'nested': ''}),
        ('region_box_embeddings.embedding', {'nested': 'region_box_embeddings'}),
    ]


def test_chunks() -> None:
    assert [len(c) for c in suite_probes.chunks(list(range(10)), 4)] == [4, 4, 2]


def test_skip_flags_default_off_and_parse() -> None:
    base = [
        '--manifest', 'm.json', '--images', 'i', '--out', 'o.json', '--project', 'p',
        '--api-url', 'a', '--triton-url', 't', '--opensearch-url', 'o',
    ]  # fmt: skip
    parser = bs.build_parser()
    off = parser.parse_args(base)
    assert (off.skip_cluster, off.skip_settle) == (False, False)
    on = parser.parse_args([*base, '--skip-cluster', '--skip-settle'])
    assert (on.skip_cluster, on.skip_settle) == (True, True)


def test_set_embedding_mode_puts_lazy_at_the_current_revision() -> None:
    import httpx

    seen: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.content))
        if request.method == 'GET':
            return httpx.Response(200, json={'revision': 4})
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        suite_ingest.set_embedding_mode(client, 'http://api/curation/projects/p', 'all')
        assert seen == []
        suite_ingest.set_embedding_mode(client, 'http://api/curation/projects/p', 'lazy')
    assert [(m, p) for m, p, _ in seen] == [
        ('GET', '/curation/projects/p/ingest/policy'),
        ('PUT', '/curation/projects/p/ingest/policy'),
    ]
    assert b'"expected_revision":4' in seen[1][2].replace(b' ', b'')
    assert b'"lazy"' in seen[1][2]


def _perf_cmd(monkeypatch, tmp_path, extra: list[str]) -> list[str]:
    from scripts.bench import suite_phases

    seen: list[list[str]] = []
    monkeypatch.setattr(suite_phases, 'run_cmd', lambda cmd, **_: seen.append(cmd))
    args = bs.build_parser().parse_args(
        [
            '--manifest', 'm', '--images', 'i', '--out', 'o', '--project', 'p',
            '--api-url', 'a', '--triton-url', 't', '--opensearch-url', 'os',
            '--perf-dir', str(tmp_path), '--perf-models', 'yolo', '--perf-batches', '1',
            *extra,
        ]
    )  # fmt: skip
    suite_phases.phase_triton(args)
    return seen[0]


def test_perf_analyzer_defaults_keep_the_full_tier(monkeypatch, tmp_path) -> None:
    cmd = _perf_cmd(monkeypatch, tmp_path, [])
    assert cmd[cmd.index('--concurrency-range') + 1] == '1:16:5'
    assert cmd[cmd.index('--measurement-interval') + 1] == '5000'


def test_perf_analyzer_quick_knobs(monkeypatch, tmp_path) -> None:
    cmd = _perf_cmd(
        monkeypatch, tmp_path, ['--perf-concurrency', '1:1:1', '--perf-interval-ms', '2000']
    )
    assert cmd[cmd.index('--concurrency-range') + 1] == '1:1:1'
    assert cmd[cmd.index('--measurement-interval') + 1] == '2000'
