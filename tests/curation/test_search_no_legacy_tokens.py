"""Guard test: the semantic-search files never reintroduce the legacy
``/search/*`` surface, ``visual_search_*``, or ``MobileCLIP``.

Mirrors what ``scripts/codegen/check_no_legacy_search.py`` /
``check_no_mobileclip.py`` enforce at pre-commit time (neither guard is
wired up here — this test stands in for them at file scope), scoped
explicitly to the files this task added — an independent, in-repo
double-check that
``/curation/search/text`` is the only ``/search/*``-shaped route string
these files ever emit.
"""

from __future__ import annotations

import re
from pathlib import Path


NEW_FILES = [
    'src/routers/curation/search.py',
    'src/services/curation/semantic_search.py',
]

# Any '/search/...' occurrence that ISN'T one of the pre-blessed suffixes.
# The curation router is mounted with a configurable prefix elsewhere (not
# visible in these files individually), so what actually appears in source
# is bare '/search/text' / '/search/image' — check the suffix, not a full
# '/curation/search/...' string.
_SEARCH_PATH_RE = re.compile(r'/search/([A-Za-z0-9_]+)')
_ALLOWED_SEARCH_SUFFIXES = {'text', 'image'}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_no_disallowed_search_paths():
    for rel in NEW_FILES:
        text = (_repo_root() / rel).read_text(encoding='utf-8')
        for match in _SEARCH_PATH_RE.finditer(text):
            suffix = match.group(1)
            assert suffix in _ALLOWED_SEARCH_SUFFIXES, (
                f'{rel} contains a disallowed /search/* path: {match.group(0)!r} '
                f'(only /search/{_ALLOWED_SEARCH_SUFFIXES} are permitted)'
            )


def test_no_visual_search_or_mobileclip_tokens():
    forbidden = re.compile(r'\bvisual_search\w*\b|\bMobileCLIP\b', re.IGNORECASE)
    for rel in NEW_FILES:
        text = (_repo_root() / rel).read_text(encoding='utf-8')
        hits = forbidden.findall(text)
        assert not hits, f'{rel} contains forbidden legacy tokens: {hits}'
