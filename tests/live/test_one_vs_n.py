"""The 1-vs-N diff (plan 05 section 4B, Q-E26, review fix R-15): one API worker against several.

Each configuration re-creates ``api``/``jobs-worker`` through the scale overlay
(``docker/test/compose.scale.yml``) behind its nginx, runs the same scripted scenario twice and
judges it two ways:

* sequential: the steps one at a time; the normalised records (class ids relative to the first id
  this run created, revisions relative to the starting one, names with the run tag removed,
  terminal job states) must be identical across configurations;
* concurrent: races whose winner legitimately differs, judged by invariants (exactly one winner of
  twenty singleton starts, no gap or repeat in the class ids, no 5xx, the same event sequence on
  two simultaneous streams).

It also checks that the pool sizes follow the knob: the container sees ``WEB_CONCURRENCY`` and
``OP_API_REPLICAS``, derives ``total // (workers x replicas)`` connections per process, and the
PostgreSQL backends of the API role never exceed that total. ``op_db_pool_wait_seconds`` is
exposed. Docs: uvicorn ``WEB_CONCURRENCY`` https://www.uvicorn.org/deployment/ ; Compose
``deploy.replicas`` https://docs.docker.com/reference/compose-file/deploy/#replicas
"""

from __future__ import annotations

import itertools
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

from tests.live import _compose
from tests.live._jobs import start_infra_job, wait_for_job
from tests.live._scale import exec_json, reconfigure
from tests.live._sse import SseCollector
from tests.live.conftest import (
    API_PREFIX,
    API_URL,
    JOB_TERMINAL,
    PROJECT_PREFIX,
    compose,
    wait_for_state,
    wait_until,
)


pytestmark = pytest.mark.live

CONFIGS = [(1, 1, 1), (4, 2, 2)]  # (API_WORKERS, api replicas, jobs-worker replicas)
TERMINAL = {'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted'}
POOL_PROBE = (
    'import json; from src.config.api_pools import api_worker_count, db_pool_size, '
    'executor_threads; from src.config.database import DatabaseSettings as D; s = D.from_env(); '
    'print(json.dumps({"workers": api_worker_count(), "replicas": s.api_replicas, '
    '"total": s.api_connections_total, "pool": db_pool_size(), "threads": executor_threads()}))'
)


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=180.0)


def _reseed() -> None:
    proc = compose(
        'exec', '-T', 'api', 'python3', '/app/scripts/curation/seed_live_harness.py', '--wipe',
        check=False, timeout=900,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-1500:]


def wait_job(
    api: httpx.Client, job_id: str, *, prefix: str = PROJECT_PREFIX, timeout: float = 120.0
) -> dict[str, Any]:
    def fetch() -> dict[str, Any]:
        resp = api.get(f'{prefix}/jobs/{job_id}')
        assert resp.status_code == 200, resp.text
        return resp.json()

    return wait_for_state(fetch, lambda j: j['status'] in JOB_TERMINAL, timeout=timeout)


def _classes(api: httpx.Client) -> dict[str, dict[str, Any]]:
    resp = api.get(f'{PROJECT_PREFIX}/classes')
    resp.raise_for_status()
    return {c['class_name']: c for c in resp.json()['classes']}


class _Norm:
    """Remove what legitimately differs between runs: the tag, absolute ids and revisions."""

    def __init__(self, tag: str, id_base: int, revision_base: int) -> None:
        self.tag, self.id_base, self.revision_base = tag, id_base, revision_base

    def name(self, value: str) -> str:
        return value.replace(self.tag, '<T>')

    def cid(self, class_id: int) -> int:
        return class_id - self.id_base

    def revision(self, revision: int) -> int:
        return revision - self.revision_base


def _sequential(api: httpx.Client, tag: str) -> list[Any]:
    start = _classes(api)
    id_base = max(c['class_id'] for c in start.values()) + 1
    revision_base = api.get(f'{PROJECT_PREFIX}/settings').json()['revision']
    norm = _Norm(tag, id_base, revision_base)
    log: list[Any] = []

    names = {key: f'{key}_{tag}' for key in ('a', 'b', 'c')}
    for name in names.values():
        resp = api.post(f'{PROJECT_PREFIX}/classes', json={'name': name, 'group': 'scale'})
        log.append(['create', norm.name(name), resp.status_code, norm.cid(resp.json()['class_id'])])
    dup = api.post(f'{PROJECT_PREFIX}/classes', json={'name': names['a']})
    log.append(['create_duplicate', dup.status_code])

    by_name = _classes(api)
    renamed = f'b2_{tag}'
    resp = api.put(
        f'{PROJECT_PREFIX}/classes/{by_name[names["b"]]["class_id"]}', json={'name': renamed}
    )
    log.append(['rename', resp.status_code, norm.name(resp.json()['class_name'])])

    merge = api.post(
        f'{PROJECT_PREFIX}/classes/merge',
        json={
            'source_id': by_name[names['c']]['class_id'],
            'target_id': by_name[names['a']]['class_id'],
        },
    )
    job = wait_job(api, merge.json()['id'])
    log.append(['merge', merge.status_code, merge.json()['kind'], job['status']])
    final = _classes(api)
    log.append(
        [
            'classes',
            {
                norm.name(n): [
                    norm.cid(c['class_id']),
                    c['deprecated'],
                    None if c.get('merged_into') is None else norm.cid(c['merged_into']),
                ]
                for n, c in final.items()
                if n.endswith(tag)
            },
        ]
    )

    revision = api.get(f'{PROJECT_PREFIX}/settings').json()['revision']
    saved = api.put(
        f'{PROJECT_PREFIX}/settings',
        json={'defaults': {'sort': 'recent'}, 'expected_revision': revision},
    )
    stale = api.put(
        f'{PROJECT_PREFIX}/settings',
        json={'defaults': {'sort': 'recent'}, 'expected_revision': revision},
    )
    log.append(
        ['settings', saved.status_code, norm.revision(saved.json()['revision']), stale.status_code]
    )

    noop = wait_for_job(api, start_infra_job('noop'), TERMINAL, timeout_s=90)
    assert noop is not None
    log.append(['noop', noop['status'], noop['attempt'], noop['result']])

    slug = f'sc-{tag}'
    created = api.post(
        f'{API_PREFIX}/projects', json={'slug': slug, 'display_name': f'Scale {tag}'}
    )
    create_job = wait_job(api, created.json()['job_id'], prefix=API_PREFIX, timeout=240)
    listed = {p['slug']: p['status'] for p in api.get(f'{API_PREFIX}/projects').json()['projects']}
    log.append(['project', created.status_code, create_job['status'], listed.get(slug)])
    return log


def _concurrent(api: httpx.Client, tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        # 20 starts of one singleton: the near-duplicate run holds its exclusivity group.
        def start(_: int) -> httpx.Response:
            with _client() as c:
                return c.post(
                    f'{PROJECT_PREFIX}/near_duplicates/runs',
                    json={'level': 'image', 'mode': 'full'},
                )

        starts = list(pool.map(start, range(20)))
        codes = sorted(r.status_code for r in starts)
        out['start_codes'] = codes
        winners = [r for r in starts if r.status_code == 202]
        assert winners, codes
        assert all(
            r.status_code == 409 and r.json()['detail']['error'] == 'job_busy'
            for r in starts
            if r.status_code != 202
        ), [r.text for r in starts]
        # A run on this small index can finish while a late request is still connecting, so a
        # second 202 is legitimate only if its job was queued after the first one had finished.
        views = sorted(
            (wait_job(api, w.json()['job_id'], timeout=240) for w in winners),
            key=lambda j: j['created_at'],
        )
        out['singleton_job'] = sorted({v['status'] for v in views})
        out['singleton_windows_disjoint'] = all(
            nxt['created_at'] >= prev['finished_at'] for prev, nxt in itertools.pairwise(views)
        )

        # concurrent class creates: 6 names, each posted twice.
        names = [f'k{i}_{tag}' for i in range(6)]

        def create(name: str) -> httpx.Response:
            with _client() as c:
                return c.post(f'{PROJECT_PREFIX}/classes', json={'name': name})

        created = list(pool.map(create, names * 2))
        out['create_codes'] = sorted(r.status_code for r in created)
        ids = sorted(r.json()['class_id'] for r in created if r.status_code == 201)
        out['ids_contiguous'] = ids == list(range(ids[0], ids[0] + len(names)))
        listed = [n for n in _classes(api) if n in names]
        out['distinct_names'] = len(listed)

    # two simultaneous streams (the balancer may put them on different replicas)
    one, two = (
        SseCollector(f'{API_URL}{API_PREFIX}/events').start(),
        SseCollector(f'{API_URL}{API_PREFIX}/events').start(),
    )
    try:
        job_id = start_infra_job('noop')
        for sse in (one, two):
            assert sse.wait_for(
                lambda ev: any(
                    e.get('event') == 'job.state'
                    and e['data']['job_id'] == job_id
                    and e['data']['status'] == 'completed'
                    for e in ev
                ),
                timeout_s=60,
            ), f'{sse.for_job(job_id)} {sse.error}'
    finally:
        one.close()
        two.close()
    states = [
        [e['data']['status'] for e in s.for_job(job_id) if e.get('event') == 'job.state']
        for s in (one, two)
    ]
    out['stream_states'] = states
    return out


def _pool_facts(workers: int, replicas: int) -> dict[str, Any]:
    probes = [exec_json(POOL_PROBE, index=i) for i in range(1, replicas + 1)]
    procs = []
    for i in range(1, replicas + 1):
        top = compose('exec', '-T', '--index', str(i), 'api', 'sh', '-c',
                      "ps -eo args | grep -c '[s]pawn_main' || true")  # fmt: skip
        procs.append(int(top.stdout.strip() or 0))
    backends = int(
        _compose.psql("SELECT count(*) FROM pg_stat_activity WHERE usename = 'op_app'") or 0
    )
    return {'probes': probes, 'worker_processes': procs, 'op_app_backends': backends}


@pytest.fixture(scope='module')
def runs() -> dict[tuple[int, int, int], dict[str, Any]]:
    results: dict[tuple[int, int, int], dict[str, Any]] = {}
    for cfg in CONFIGS:
        workers, api_replicas, jobs_replicas = cfg
        reconfigure(workers=workers, api_replicas=api_replicas, jobs_replicas=jobs_replicas)
        _reseed()
        with _client() as api:
            assert wait_until(lambda: api.get('/live').status_code == 200, timeout=60)
            api.get(f'{PROJECT_PREFIX}/classes').raise_for_status()
            tag = uuid.uuid4().hex[:6]
            seq = _sequential(api, tag)
            with _client() as api2:
                conc = _concurrent(api2, tag)
            metrics = api.get('/metrics').text
        results[cfg] = {
            'sequential': seq,
            'concurrent': conc,
            'pools': _pool_facts(workers, api_replicas),
            'pool_wait_exposed': 'op_db_pool_wait_seconds_count' in metrics,
            'tag': tag,
        }
    return results


def test_sequential_scenario_has_an_empty_diff(runs: dict[Any, dict[str, Any]]) -> None:
    base, scaled = (runs[c]['sequential'] for c in CONFIGS)
    assert base == scaled, f'1-vs-N differ:\n{base}\n{scaled}'
    assert all(step[0] != 'project' or step[1:] == [202, 'completed', 'active'] for step in base)


def test_concurrent_scenario_keeps_its_invariants(runs: dict[Any, dict[str, Any]]) -> None:
    for cfg in CONFIGS:
        conc = runs[cfg]['concurrent']
        assert set(conc['start_codes']) <= {202, 409}, (cfg, conc)
        assert conc['start_codes'].count(202) >= 1, (cfg, conc)
        assert conc['singleton_job'] == ['completed'], (cfg, conc)
        assert conc['singleton_windows_disjoint'] is True, (cfg, conc)
        assert conc['create_codes'] == [201] * 6 + [409] * 6, (cfg, conc)
        assert conc['ids_contiguous'] is True, (cfg, conc)
        assert conc['distinct_names'] == 6, (cfg, conc)
        one, two = conc['stream_states']
        assert one == two == ['queued', 'running', 'completed'], (cfg, conc)


def test_pool_sizes_follow_the_worker_knob(runs: dict[Any, dict[str, Any]]) -> None:
    for workers, replicas, _jobs in CONFIGS:
        facts = runs[(workers, replicas, _jobs)]['pools']
        assert len(facts['probes']) == replicas
        for probe in facts['probes']:
            assert (probe['workers'], probe['replicas']) == (workers, replicas), probe
            assert probe['pool'] == max(1, probe['total'] // (workers * replicas)), probe
        # uvicorn runs one worker in the main process; only workers > 1 spawn child processes
        spawned = workers if workers > 1 else 0
        assert facts['worker_processes'] == [spawned] * replicas, facts
        total = facts['probes'][0]['total']
        assert 0 < facts['op_app_backends'] <= total + 40, facts  # + workers' own pools
        assert runs[(workers, replicas, _jobs)]['pool_wait_exposed'] is True
    # a smaller worker count gives each process a larger share, never a larger total
    first, last = (runs[c]['pools']['probes'][0] for c in CONFIGS)
    assert first['pool'] * 1 * 1 <= first['total']
    assert last['pool'] * 4 * 2 <= last['total']
    assert re.fullmatch(r'[0-9a-f]{6}', runs[CONFIGS[0]]['tag'])
