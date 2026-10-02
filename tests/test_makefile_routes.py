"""Every API route a Makefile target calls must exist.

Found when ``make opensearch-reset-indexes`` called ``DELETE /index`` and
``POST /index/create``, neither of which has ever been a route.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    'check_docs_vs_code', REPO_ROOT / 'scripts' / 'docs' / 'check_docs_vs_code.py'
)
assert _SPEC is not None
assert _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
sys.modules['check_docs_vs_code'] = checker
_SPEC.loader.exec_module(checker)

_CALL = re.compile(r'\$\(API_PORT\)(/(?:[^\s"\'?$)]|\$\([^)]*\))*)')


def makefile_paths(text: str) -> set[str]:
    return {
        checker._norm(re.sub(r'\$\([^)]*\)', '{}', raw))
        for raw in _CALL.findall(text)
        if raw.strip('/')
    }


def test_every_route_a_make_target_calls_exists() -> None:
    known = {path for _method, path in checker.known_routes()}
    prefixes = ('', '/curation/projects/{}')
    missing = sorted(
        p
        for p in makefile_paths((REPO_ROOT / 'Makefile').read_text())
        if not any(prefix + p in known for prefix in prefixes)
    )
    assert missing == []


def test_the_scan_sees_a_made_up_route() -> None:
    assert makefile_paths('curl http://localhost:$(API_PORT)/index/create') == {'/index/create'}
