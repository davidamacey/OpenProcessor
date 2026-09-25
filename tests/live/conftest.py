"""Fixtures + the safety guard for the live verification suite.

Nothing in this package may run against anything but the disposable
harness. The guard below is the first fixture every test depends on: it
refuses to run when the OpenSearch endpoint looks like a real deployment
or when any configured index name lacks the ``verify_`` prefix.

Run the suite with::

    docker compose -p op-live-verify -f docker/test/compose.yml up -d --wait
    python -m pytest tests/live -q --no-cov -m live
    docker compose -p op-live-verify -f docker/test/compose.yml down -v --remove-orphans
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - drives the harness's own compose project
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / 'docker' / 'test' / 'compose.yml'
COMPOSE_PROJECT = 'op-live-verify'
VERIFY_DATA_DIR = REPO_ROOT / 'docker' / 'test' / 'verify-data'
JOBS_DIR = VERIFY_DATA_DIR / 'jobs'
EXPORTS_DIR = VERIFY_DATA_DIR / 'exports'

API_URL = os.environ.get('VERIFY_API_URL', 'http://localhost:14701')
OPENSEARCH_URL = os.environ.get('VERIFY_OPENSEARCH_URL', 'http://localhost:14702')
FAKE_VLM_URL = os.environ.get('VERIFY_FAKE_VLM_URL', 'http://localhost:14704')

# Must match docker/test/compose.yml's OP_API_PREFIX + OP_*_INDEX values.
API_PREFIX = os.environ.get('VERIFY_API_PREFIX', '/curation')
INDEXES = {
    'images': 'verify_images',
    'items': 'verify_items',
    'labels_confirmed': 'verify_labels_confirmed',
    'classes': 'verify_classes',
    'clusters': 'verify_clusters',
    'settings': 'verify_settings',
}

# Ports a real deployment on this host is known to use. Hitting any of
# them means the harness is mis-wired and must not be written to.
FORBIDDEN_PORTS = {4600, 4601, 4602, 4603, 4604, 4605, 4606, 4607, 4608, 9200}

# Seeded cohorts (see scripts/curation/seed_live_harness.py::COHORTS).
CLASS_NAMES = ('box', 'envelope', 'tube', 'crate', 'pallet', 'drum', 'sack', 'canister')
PURE_CLUSTER_ID = 0
MIXED_CLUSTER_ID = 1
CANDIDATE_CLUSTER_ID = 10000
SMALL_CANDIDATE_CLUSTER_ID = 10002
REGION_CLUSTER_ID = 1
FP_REGION_CLUSTER_ID = -100
MERGE_SOURCE_CLASS_ID = 7
MERGE_TARGET_CLASS_ID = 6
FROZEN_CLASS_ID = 2


def compose(*args: str, check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a docker compose subcommand against the harness project."""
    cmd = [
        'docker',
        'compose',
        '-p',
        COMPOSE_PROJECT,
        '-f',
        str(COMPOSE_FILE),
        *args,
    ]
    return subprocess.run(  # nosec B603 - fixed argv, no shell
        cmd,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


def docker_ps_snapshot() -> str:
    """Stable, sorted `docker ps` view used to prove the GPU arbiter no-ops."""
    proc = subprocess.run(  # nosec B603 - fixed argv, no shell
        ['docker', 'ps', '--format', '{{.Names}}\t{{.Status}}\t{{.Image}}'],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Status carries an uptime counter ("Up 3 minutes") which ticks on its
    # own; strip it so the comparison is about container membership, which
    # is what a container-stopping arbiter would change.
    names = sorted(line.split('\t')[0] for line in proc.stdout.splitlines() if line.strip())
    return '\n'.join(names)


def wait_for_state(
    fetch: Any,
    done: Any,
    *,
    timeout: float = 120.0,
    interval: float = 1.0,
) -> Any:
    """Poll ``fetch()`` until ``done(state)`` holds; return the last state.

    Returns the final (possibly still-not-done) state rather than raising,
    so the caller's assertion can show what it actually saw.
    """
    deadline = time.monotonic() + timeout
    state = fetch()
    while time.monotonic() < deadline:
        if done(state):
            return state
        time.sleep(interval)
        state = fetch()
    return state


def wait_until(predicate: Any, *, timeout: float = 60.0, interval: float = 0.5) -> Any:
    """Poll ``predicate`` until it returns something truthy."""
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    return last


# Indexes OpenSearch itself creates (system, plugin bookkeeping). Anything
# else that is not `verify_`-prefixed means this is somebody's real cluster.
_INTERNAL_INDEX_PREFIXES = (
    '.',
    'top_queries',
    'security-auditlog',
    'ism-',
    'opensearch_dashboards',
)


def _is_internal_index(name: str) -> bool:
    return name.startswith(_INTERNAL_INDEX_PREFIXES)


@pytest.fixture(scope='session', autouse=True)
def harness_safety_guard() -> None:
    """Refuse to touch anything that is not the disposable harness."""
    for label, url in (('api', API_URL), ('opensearch', OPENSEARCH_URL)):
        parsed = urlparse(url)
        if parsed.hostname not in ('localhost', '127.0.0.1'):
            pytest.exit(f'live suite refuses a non-local {label} host: {url}', returncode=3)
        if parsed.port in FORBIDDEN_PORTS:
            pytest.exit(
                f'live suite refuses {label} port {parsed.port} — that band belongs '
                f'to a real deployment, not the harness ({url})',
                returncode=3,
            )
        if not (14700 <= int(parsed.port or 0) <= 14799):
            pytest.exit(
                f'live suite expects the harness {label} on a 147xx port, got {url}',
                returncode=3,
            )

    unprefixed = {k: v for k, v in INDEXES.items() if not v.startswith('verify_')}
    if unprefixed:
        pytest.exit(
            f'live suite refuses index names without the verify_ prefix: {unprefixed}',
            returncode=3,
        )

    try:
        resp = httpx.get(f'{OPENSEARCH_URL}/_cat/indices?format=json', timeout=10.0)
        resp.raise_for_status()
    except Exception as exc:
        pytest.exit(
            f'harness OpenSearch not reachable at {OPENSEARCH_URL} ({exc}). Start it with: '
            'docker compose -p op-live-verify -f docker/test/compose.yml up -d --wait',
            returncode=3,
        )
    stray = [
        name
        for name in (str(i.get('index', '')) for i in resp.json())
        if name and not name.startswith('verify_') and not _is_internal_index(name)
    ]
    if stray:
        pytest.exit(
            f'harness OpenSearch holds non-verify_ indexes {stray} — this is not a '
            'disposable instance, refusing to run',
            returncode=3,
        )


@pytest.fixture(scope='session')
def api(harness_safety_guard: None) -> Any:
    """HTTP client bound to the harness API (synchronous by design)."""
    with httpx.Client(base_url=API_URL, timeout=180.0) as client:
        try:
            client.get('/live').raise_for_status()
        except Exception as exc:
            pytest.exit(f'harness API not reachable at {API_URL}: {exc}', returncode=3)
        yield client


@pytest.fixture(scope='session')
def opensearch(harness_safety_guard: None) -> Any:
    with httpx.Client(base_url=OPENSEARCH_URL, timeout=60.0) as client:
        yield client


@pytest.fixture(scope='session')
def fake_vlm(harness_safety_guard: None) -> Any:
    with httpx.Client(base_url=FAKE_VLM_URL, timeout=30.0) as client:
        client.post('/__reset')
        yield client
        client.post('/__reset')


@pytest.fixture(scope='session', autouse=True)
def seeded(api: Any, opensearch: Any) -> dict[str, Any]:
    """Wipe + reseed the harness once per session, inside the API container.

    Running the seeder in-container (rather than from the host) keeps a
    single source of truth for every configured path: the item docs record
    exactly the ``image_path`` the API will later resolve, and the class
    registry is written to the same file the API reads.
    """
    proc = compose(
        'exec',
        '-T',
        'api',
        'python3',
        '/app/scripts/curation/seed_live_harness.py',
        '--wipe',
        check=False,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.exit(
            f'seeding failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}',
            returncode=3,
        )
    count = opensearch.get(f'/{INDEXES["items"]}/_count').json()['count']
    assert count >= 400, f'seed produced only {count} item docs'
    # The API caches its index bootstrap flag and registry mtime; a probe
    # request after seeding makes sure it has re-read both before the
    # first assertion-bearing call.
    api.get(f'{API_PREFIX}/classes').raise_for_status()
    return {'items': count, 'stdout': proc.stdout}


@pytest.fixture(scope='session')
def api_client(api: Any) -> Any:
    """Prefix-aware helper: ``api_client.get('/crops')`` hits ``{API_PREFIX}/crops``."""

    class _Prefixed:
        def __getattr__(self, name: str) -> Any:
            method = getattr(api, name)

            def call(path: str, *args: Any, **kwargs: Any) -> Any:
                return method(f'{API_PREFIX}{path}', *args, **kwargs)

            return call

    return _Prefixed()


@pytest.fixture
def items_index() -> str:
    return INDEXES['items']


def refresh(opensearch: Any, index: str) -> None:
    opensearch.post(f'/{index}/_refresh').raise_for_status()


def get_doc(opensearch: Any, index: str, doc_id: str) -> dict[str, Any]:
    resp = opensearch.get(f'/{index}/_doc/{doc_id}')
    resp.raise_for_status()
    return resp.json()


def search(opensearch: Any, index: str, body: dict[str, Any]) -> dict[str, Any]:
    resp = opensearch.post(
        f'/{index}/_search',
        content=json.dumps(body),
        headers={'Content-Type': 'application/json'},
    )
    resp.raise_for_status()
    return resp.json()


def crop_ids_in(opensearch: Any, prefix: str, limit: int = 50) -> list[str]:
    """Deterministically ordered crop ids from one seeded cohort."""
    body = {
        'size': limit,
        '_source': False,
        'query': {'prefix': {'crop_id': f'{prefix}_'}},
        'sort': [{'crop_id': {'order': 'asc'}}],
    }
    hits = search(opensearch, INDEXES['items'], body)['hits']['hits']
    return [h['_id'] for h in hits]
