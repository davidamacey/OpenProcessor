"""The isolation proof (projects_plan.md §10 P1): every scoped curation
route, swept first as one project and then as the other (alpha then beta,
and beta then alpha), reads and writes only the bound project's
OpenSearch indexes, directories, caches and event stream.

Why two passes: a process cache primed by the first project is read back
by the second one. A single sweep as one project can never see that.

Setup: three ordinary projects -- ``default``, ``alpha`` and ``beta`` --
on a fake OpenSearch transport behind the real
project guard. Each is seeded with its own items, images, labels and
classes, and with filesystem state under its own resources (job state,
a training run, an FP centroid store). Every id and class name embeds
the project slug, so a leak is greppable.

Per route, both passes assert:
- no OpenSearch access to another project's index (recorded *above* the
  guard with this file's own parser, so a refused or swallowed access
  still counts, and ``mget``/``msearch``/``bulk`` bodies are read whatever
  their Python type);
- the bound project never drifts;
- no other project's marker in the response, and every served URL is
  under the bound project's prefix;
- no event reaching another project's stream, and no event on the
  global stream;
- no 5xx unless listed in ``EXPECTED_5XX`` with a reason;
- a mutating route sends a valid body (a 422 means "unmapped body") and
  really writes (to the bound project's indexes or dirs) unless listed in
  ``NO_WRITE`` with a reason.
Across the second pass, the first project's directories are byte-for-byte
unchanged.

Routes are enumerated from the app's route table, so a route added later
is covered automatically; a new path parameter, streaming route or
mutating route fails the test until it is mapped below.
"""

from __future__ import annotations

import collections
import hashlib
import inspect
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from _route_helpers import api_routes
from fastapi.testclient import TestClient

from curation.cross_project_leak._constants import API, CLASS_NAMES, SCOPED, SLUGS
from curation.cross_project_leak._route_bodies import route_bodies, route_params
from curation.cross_project_leak._route_tables import (
    CROP_FOR,
    CROSS_PROJECT_CONFIG_READS,
    EXPECTED_5XX,
    NO_WRITE,
    STREAMING_ROUTES,
    UNBOUND_BY_DESIGN,
    UNSEEDED_WRITES,
)
from curation.cross_project_leak._seeders import NAME_FOR, PREPARE


if TYPE_CHECKING:
    from curation.cross_project_leak._transport import LeakEnv


def _served_urls(value: Any) -> list[str]:
    """Every string in a JSON body that *is* a curation URL."""
    if isinstance(value, str):
        return [value] if value.startswith(f'{API}/') else []
    if isinstance(value, dict):
        return [u for v in value.values() for u in _served_urls(v)]
    if isinstance(value, list):
        return [u for v in value for u in _served_urls(v)]
    return []


def _scoped_routes(app: Any) -> list[tuple[str, str]]:
    """Every (method, path template) mounted under the scoped prefix."""
    out: list[tuple[str, str]] = []
    for route in api_routes(app):
        path = route.path
        if not path.startswith(SCOPED):
            continue
        out.extend((method, path) for method in sorted(route.methods or ()) if method != 'HEAD')
    return out


def _fill(path: str, slug: str, key: tuple[str, str] | None = None) -> str:
    params = route_params(slug)
    if key in CROP_FOR:
        params['crop_id'] = f'{slug}-{CROP_FOR[key]}'
    if key in NAME_FOR:
        params['name'] = NAME_FOR[key]

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == 'project':
            return slug
        if name not in params:
            raise AssertionError(f'unmapped route {path}: add {name!r} to route_params')
        return params[name]

    return re.sub(r'\{(\w+)(?::\w+)?\}', _sub, path)


def _is_streaming(app: Any, path: str) -> bool:
    for route in api_routes(app):
        if route.path == path:
            annotation = inspect.signature(route.endpoint).return_annotation
            return 'StreamingResponse' in str(annotation)
    return False


def _markers(slug: str) -> tuple[str, ...]:
    return (
        # `_stored_prompt_pack`/
        # `_stored_region_profile` already name every seeded pack/profile
        # `f'{slug}-model'` -- but nothing scanned responses for that
        # pattern, so a config-store *content* leak (another project's
        # cached pack/profile showing up in a list response) was invisible
        # to this sweep even though the marker was right there in the doc
        # id/name the whole time.
        *(
            f'{slug}{kind}'
            for kind in ('-item', '-img', '-label', '-job', '-fp', '-campaign', '-model')
        ),
        CLASS_NAMES[slug],
    )


def _dir_digest(dirs: list[Path]) -> dict[str, str]:
    digest: dict[str, str] = {}
    for root in dirs:
        if root.is_file():
            digest[str(root)] = hashlib.sha256(root.read_bytes()).hexdigest()
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if path.is_file():
                digest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


Shape = tuple[str, str, str]  # (index role, HTTP verb, OpenSearch action)


def _shape(role_of: dict[str, str], verb: str, os_url: str, index: str) -> Shape:
    parts = [p for p in os_url.split('?', 1)[0].split('/') if p]
    action = next((p for p in parts if p.startswith('_')), '<index>')
    return role_of.get(index, 'foreign'), verb, action


def _sweep(
    env: LeakEnv, slug: str, only: Any = None
) -> tuple[list[str], set[tuple[str, str]], dict[tuple[str, str], collections.Counter[Shape]]]:
    """Call every scoped route (or those ``only`` accepts) as ``slug``.
    Returns (leaks, routes that wrote, the OpenSearch request shapes each
    route issued, counted per index role)."""
    from src.config.curation_roles import IndexRole

    app, records = env.app, env.records
    own_indexes = set(records[slug].resources.indexes.values())
    foreign_indexes = {
        name: other
        for other, record in records.items()
        if other != slug
        for name in record.resources.indexes.values()
    }
    foreign_configs = {
        record.resources.indexes[IndexRole.CONFIGS]: other
        for other, record in records.items()
        if other != slug
    }
    foreign_markers = [m for other in SLUGS if other != slug for m in _markers(other)]
    bodies = route_bodies(slug, records[slug].resources.export_root)

    role_of = {name: role.value for role, name in records[slug].resources.indexes.items()}
    # The deployment-wide VLM registry is a process cache shared by every
    # project by design, so it is excluded from the per-role parity below.
    from src.services.config_store.global_store import global_configs_index

    role_of[global_configs_index()] = 'registry'
    leaks: list[str] = []
    wrote: set[tuple[str, str]] = set()
    roles: dict[tuple[str, str], collections.Counter[Shape]] = {}
    client = TestClient(app, raise_server_exceptions=False)
    # Reads first, against the seeded state; then every write.
    ordered = sorted(_scoped_routes(app), key=lambda route: route[0] != 'GET')
    for method, template in ordered:
        if template in STREAMING_ROUTES:
            continue
        key = (method, template[len(SCOPED) :])
        if only is not None and not only(key):
            continue
        url = _fill(template, slug, key)
        before_access, before_write, before_events = (
            len(env.accesses),
            len(env.transport.writes),
            len(env.events),
        )
        if key in PREPARE:
            PREPARE[key](env, slug)
        tree_before = _dir_digest([env.root]) if method != 'GET' else {}
        kwargs = bodies.get(key, {} if method in ('GET', 'DELETE') else {'json': {}})
        response = client.request(method, url, **kwargs)
        route_accesses = env.accesses[before_access:]
        route_writes = env.transport.writes[before_write:]
        route_events = env.events[before_events:]
        tag = f'[{slug}] {method} {key[1]}'
        roles[key] = collections.Counter(
            shape
            for _b, verb, os_url, index in route_accesses
            if (shape := _shape(role_of, verb, os_url, index))[0] != 'registry'
        )

        for bound, _verb, os_url, index in route_accesses:
            if (
                key in CROSS_PROJECT_CONFIG_READS
                and _verb == 'GET'
                and foreign_configs.get(index) == bound
            ):
                continue
            if index == '*' or index in foreign_indexes:
                leaks.append(
                    f'{tag}: bound={bound} reached {foreign_indexes.get(index, "every")!r} '
                    f'index {index} ({os_url})'
                )
            if bound != slug and key not in UNBOUND_BY_DESIGN:
                leaks.append(f'{tag}: OpenSearch call bound to {bound!r}')
        leaks.extend(
            f'{tag}: event {event.get("type")} went to project {event.get("project")!r}'
            for event in route_events
            if event.get('project') != slug and key not in UNBOUND_BY_DESIGN
        )

        body = response.text
        leaks.extend(f'{tag}: response carries {m!r}' for m in foreign_markers if m in body)
        if response.headers.get('content-type', '').startswith('application/json'):
            leaks.extend(
                f'{tag}: served URL {u!r} is not under {slug} prefix'
                for u in _served_urls(response.json())
                if u != f'{API}/projects/{slug}' and not u.startswith(f'{API}/projects/{slug}/')
            )
        if response.status_code >= 500 and key not in EXPECTED_5XX:
            leaks.append(f'{tag}: unexpected {response.status_code} {body[:300]}')
        if method != 'GET':
            if response.status_code == 422 and key not in NO_WRITE:
                leaks.append(f'{tag}: unmapped body (422 {body[:200]})')
            # A write is an OpenSearch write to its own index or any file it
            # created/changed (another project's dirs are checked apart).
            if (
                any(w in own_indexes for w in route_writes)
                or _dir_digest([env.root]) != tree_before
            ):
                wrote.add(key)
            elif key not in NO_WRITE and key not in UNSEEDED_WRITES:
                leaks.append(
                    f'{tag}: mutating route wrote nothing ({response.status_code} {body[:200]})'
                )
    return leaks, wrote, roles


def _cache_parity(
    first: str,
    second: str,
    shapes_first: dict[tuple[str, str], collections.Counter[Shape]],
    shapes_second: dict[tuple[str, str], collections.Counter[Shape]],
) -> list[str]:
    """A process cache keyed without the project answers the second project
    from the first one's data, so the route skips (some of) its own
    queries. Each route is the first call of its kind for each project, so
    the same route must issue the same requests, per index role and
    OpenSearch action, as either project."""
    return [
        f'[{second}] {m} {p}: issued {sorted(shapes_second.get((m, p), {}).items())}, '
        f'but as {first} {sorted(shapes_first[(m, p)].items())} (unkeyed cache?)'
        for (m, p) in shapes_first
        if shapes_first[(m, p)] != shapes_second.get((m, p))
    ]


@pytest.mark.parametrize(('first', 'second'), [('alpha', 'beta'), ('beta', 'alpha')])
def test_every_scoped_route_stays_inside_the_bound_project(
    leak_env: LeakEnv, first: str, second: str
) -> None:
    app = leak_env.app
    routes = _scoped_routes(app)
    assert len(routes) > 100, f'expected the full scoped surface, got {len(routes)} routes'
    streaming_unmapped = sorted({p for _m, p in routes if _is_streaming(app, p)} - STREAMING_ROUTES)
    assert not streaming_unmapped, f'unmapped streaming route(s): {streaming_unmapped}'
    mutating = {(m, p[len(SCOPED) :]) for m, p in routes if m != 'GET'}
    every = {(m, p[len(SCOPED) :]) for m, p in routes}
    mapped = set(NO_WRITE) | set(route_bodies('x', Path('/unused'))) | set(CROP_FOR) | set(PREPARE)
    stale = sorted((mapped - mutating) | (set(UNSEEDED_WRITES) - every))
    assert not stale, f'entries for routes that no longer exist: {stale}'

    leaks_first, _, roles_first = _sweep(leak_env, first)
    first_dirs = _dir_digest(leak_env.own_dirs[first])
    leaks_second, wrote, roles_second = _sweep(leak_env, second)
    skipped = _cache_parity(first, second, roles_first, roles_second)
    changed = sorted(
        path
        for path, digest in _dir_digest(leak_env.own_dirs[first]).items()
        if first_dirs.get(path) != digest
    ) + sorted(set(first_dirs) - set(_dir_digest(leak_env.own_dirs[first])))

    leaks = leaks_first + leaks_second + skipped
    leaks.extend(f"[{second}] changed {first}'s file {p}" for p in changed)
    assert not leaks, 'cross-project leak(s):\n' + '\n'.join(sorted(set(leaks)))
    # Not vacuous: every mutating route not excused in NO_WRITE really wrote.
    expected_writers = mutating - set(NO_WRITE) - set(UNSEEDED_WRITES)
    assert expected_writers <= wrote, f'routes that never wrote: {sorted(expected_writers - wrote)}'
    stale_excuses = sorted(set(NO_WRITE) & wrote)
    assert not stale_excuses, f'these routes do write; drop them from NO_WRITE: {stale_excuses}'
