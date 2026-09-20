"""Static route-parity guard between this backend and Cropwright (the
labeler frontend) — cropwright_backend_integration_plan.md §5.1 (T-D1).

Two independent checks, both pure import + introspection (no GPU, no
OpenSearch, no browser, no frontend checkout required at test time):

1. Every path in the checked-in fixture
   (``tests/fixtures/labeler_call_sites.txt`` — a snapshot of Cropwright's
   executable ``/legacy/...`` call sites, prefix-stripped and
   param-normalized) resolves to a route actually registered under
   ``CurationConfig.api_prefix``.
2. **The critical clause** (this is what would have caught the
   ``plate_thumbnail`` bug, plan §1.3): every ``f'{config.api_prefix}/...'``
   literal found in a curation router's *response payload* (as opposed to
   an ``APIRouter(prefix=...)`` declaration, which legitimately builds the
   route table itself) is scanned and checked against the same route
   table. A handler that emits a URL pointing at a segment nothing
   registers — the exact shape of the T-A1 bug — fails this test, not a
   live GET-only smoke test that never noticed the string inside a 200
   response was dead.

Regenerating the fixture (frontend call sites may drift):

    rg -o "/legacy/[A-Za-z0-9_./{}$-]+" \\
      /path/to/cropwright/src/lib/api.ts \\
      /path/to/cropwright/src/lib/sse.ts \\
      /path/to/cropwright/src/routes/export/+page.svelte \\
    | sort -u

That one-liner over-matches (rg doesn't parse out comments/JSDoc), so the
raw output needs a hand pass to drop comment-only hits before it's usable
— see docs/design/cropwright_backend_integration_plan.md §1.2 for the
audited executable/comment split as of the commit this fixture snapshots.
After filtering, strip the leading ``/legacy``, collapse ``${...}``
interpolations to the literal token ``{param}``, dedupe, sort, and update
``tests/fixtures/labeler_call_sites.txt``'s header commit hash.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / 'tests' / 'fixtures' / 'labeler_call_sites.txt'

_CURATION_ROUTER_FILES: list[Path] = [
    *sorted((REPO_ROOT / 'src' / 'routers' / 'curation').glob('*.py')),
    REPO_ROOT / 'src' / 'routers' / 'curation_images.py',
    REPO_ROOT / 'src' / 'routers' / 'curation_train.py',
    REPO_ROOT / 'src' / 'routers' / 'curation_umap.py',
]

# Known, plan-documented gaps (cropwright_backend_integration_plan.md
# §0.2/§4/§1.3) — real frontend call sites that do NOT resolve against
# this backend, either by design or because a paired fix hasn't landed on
# the frontend side of this cross-repo migration yet. Excluded so this
# guard catches NEW accidental drift, not these already-decided/in-flight
# items. Keep this dict and the fixture's own "Excluded on purpose"
# comment block in sync.
KNOWN_GAPS: dict[str, str] = {
    '/export/lpr': (
        'proprietary single-class LPR dataset export — Bucket B, '
        'deliberately never ported (plan §4.1).'
    ),
    '/export/lpr/status': 'status endpoint for the export above, same reason.',
    '/gemma/label_batch': (
        "frontend still calls the pre-rename segment; this backend's "
        'generic route is POST /vlm/label_batch (plan §3). No /gemma/* '
        'alias exists or ever will. Frontend fix is T-C1, tracked in the '
        'wt-cropwright-integration repo, not here.'
    ),
}


_PARAM_SEGMENT_RE = re.compile(r'^\{[^{}]+\}$')


def _segments(path: str) -> list[str]:
    return [s for s in path.strip('/').split('/') if s]


def _segment_matches(fixture_seg: str, route_seg: str) -> bool:
    """A route segment shaped ``{name}`` (a FastAPI path param) matches
    any single concrete path segment — including a literal frontend
    value like ``class_registry.json`` matching the backend's
    ``{artifact}`` — because a route param inherently accepts any
    non-slash string at that position. Otherwise segments must be
    literally equal."""
    if _PARAM_SEGMENT_RE.match(route_seg):
        return True
    return fixture_seg == route_seg


def _path_resolves(path: str, route_segment_lists: list[list[str]]) -> bool:
    segs = _segments(path)
    for route_segs in route_segment_lists:
        if len(route_segs) != len(segs):
            continue
        if all(_segment_matches(a, b) for a, b in zip(segs, route_segs, strict=True)):
            return True
    return False


def _registered_relative_routes() -> list[list[str]]:
    """Every ``/curation/...`` route's path, relative to the configured
    prefix, split into segments. Imports ``src.main`` fresh — this is
    pure introspection against ``app.routes``, no server, no client."""
    from src.config import get_curation_config
    from src.main import app

    prefix = get_curation_config().api_prefix
    routes: list[list[str]] = []
    for route in app.routes:
        path = getattr(route, 'path', None)
        if path and path.startswith(prefix):
            routes.append(_segments(path[len(prefix) :]))
    assert routes, 'sanity: no /curation routes found — app.routes enumeration is broken'
    return routes


def _load_fixture() -> list[str]:
    lines = []
    for raw in FIXTURE_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        lines.append(line)
    assert lines, 'sanity: labeler_call_sites.txt fixture is empty'
    return lines


def test_known_gaps_are_not_present_in_the_fixture() -> None:
    """Catches drift the other direction: someone re-adding a known-gap
    path to the fixture without removing the matching KNOWN_GAPS entry
    (which would silently stop this guard from checking it)."""
    fixture = set(_load_fixture())
    overlap = fixture & set(KNOWN_GAPS)
    assert not overlap, f'known-gap path(s) present in the fixture: {sorted(overlap)}'


def test_every_frontend_call_site_resolves_to_a_registered_route() -> None:
    """§5.1: every fixture entry (Cropwright's executable call paths,
    prefix-stripped + param-normalized) must resolve to something this
    backend actually serves."""
    routes = _registered_relative_routes()
    fixture = _load_fixture()
    unresolved = [p for p in fixture if not _path_resolves(p, routes)]
    assert not unresolved, (
        'frontend call site(s) do not resolve to any registered /curation '
        f'route: {unresolved}. If this is a newly-decided, permanent gap, '
        'add it to KNOWN_GAPS (with a reason) and remove it from the '
        'fixture instead of relaxing this assertion.'
    )


def _extract_response_payload_prefix_literals() -> list[tuple[str, int, str]]:
    """Every ``f'{config.api_prefix}/...'``-shaped literal in the curation
    router modules, excluding ``APIRouter(prefix=...)`` declarations (those
    legitimately build the route table; they are not a response payload).

    Returns ``(relative_file_path, line_number, normalized_suffix)``
    tuples, e.g. ``('src/routers/curation/regions.py', 80,
    '/crops/{param}/region_thumbnail')``.
    """
    fstring_re = re.compile(r"f(['\"])\{config\.api_prefix\}([^'\"]*)\1")
    interp_re = re.compile(r'\{[^{}]+\}')
    router_kwarg_re = re.compile(r'\bprefix\s*=\s*$')

    found: list[tuple[str, int, str]] = []
    for path in _CURATION_ROUTER_FILES:
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in fstring_re.finditer(line):
                before = line[: m.start()]
                if router_kwarg_re.search(before):
                    continue  # router registration (builds the route table itself)
                suffix = m.group(2)
                normalized = interp_re.sub('{param}', suffix)
                found.append((str(path.relative_to(REPO_ROOT)), lineno, normalized))
    return found


def test_response_payload_url_literals_resolve_to_registered_routes() -> None:
    """The clause that would have caught the plate_thumbnail bug (plan
    §1.3/§5.1): a response handler must never build a URL pointing at a
    path segment nothing registers. A GET-only smoke test can't see this
    — the endpoint still returns 200, just with a dead string inside."""
    literals = _extract_response_payload_prefix_literals()
    assert literals, (
        "sanity: found zero f'{config.api_prefix}/...' response-payload "
        'literals at all — the scan regex itself is likely broken '
        '(regions.py is known to have at least two: thumbnail_url and '
        'plate_thumbnail_url).'
    )

    routes = _registered_relative_routes()
    unresolved = [
        (file, lineno, suffix)
        for file, lineno, suffix in literals
        if not _path_resolves(suffix, routes)
    ]
    assert not unresolved, (
        f'response-payload URL literal(s) point at unregistered routes: {unresolved}'
    )


if __name__ == '__main__':
    import pytest

    pytest.main([__file__, '-v'])
