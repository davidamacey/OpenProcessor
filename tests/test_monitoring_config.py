"""Pin the Grafana Alloy log-shipper config's container-name filters.

Bug (installer plan, Wave 0): docker-compose.yml's container_name is
``${COMPOSE_PROJECT_NAME:-openprocessor}-triton`` /
``${COMPOSE_PROJECT_NAME:-openprocessor}-api``, never a bare
``triton-server`` or ``yolo-api``/``pytorch-api`` container. The old
Alloy regexes (``/triton-server.*``, ``/(yolo-api|pytorch-api).*``)
never matched THIS compose project's own containers under any project
name, so Loki only ever received a different stack's logs (or nothing).

Alloy's config format (``.alloy``, River/HCL-like) has no lightweight,
already-vendored Python parser in this repo, and this file's structure
is simple enough (flat blocks, no interpolation) that adding an HCL
dependency just for this test isn't worth it. Instead this test:

(a) parses the file into its ``discovery.relabel "<name>" { ... }``
    blocks with a structural (brace-balance) check, so a syntactically
    broken config fails loudly instead of silently at container startup;
(b) extracts each block's `regex = "..."` filter value(s) and compiles
    them as regexes;
(c) asserts those regexes actually match this compose's own
    ``container_name`` pattern (``${COMPOSE_PROJECT_NAME:-openprocessor}-*``)
    for several project names, including the default and a custom one;
(d) asserts the old, broken literal patterns are gone.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOY_CONFIG_PATH = REPO_ROOT / 'monitoring' / 'alloy-config.alloy'

_BLOCK_RE = re.compile(
    r'discovery\.relabel\s+"(?P<name>[^"]+)"\s*\{(?P<body>.*?)\n\}',
    re.DOTALL,
)
_REGEX_VALUE_RE = re.compile(r'regex\s*=\s*"(?P<pattern>[^"]+)"')

# Project names an operator might reasonably run this stack under --
# the default (unset COMPOSE_PROJECT_NAME) and a couple of installer-style
# isolated-stack names (see docs/design/openprocessor_internal
# /one_line_installer_plan.md §5.1 -- `opinst-<id>` is the live-test
# convention).
_PROJECT_NAMES = ('openprocessor', 'opinst-w0', 'opfinal', 'op_fresh2')


def _read_config() -> str:
    return ALLOY_CONFIG_PATH.read_text(encoding='utf-8')


def test_config_file_exists() -> None:
    assert ALLOY_CONFIG_PATH.is_file(), f'missing {ALLOY_CONFIG_PATH}'


def test_config_braces_are_balanced() -> None:
    """Minimal structural parse: a stray/missing brace is the most common
    way a hand-edited .alloy file breaks silently at container start."""
    text = _read_config()
    depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            assert depth >= 0, f"unbalanced '}}' at offset {i}"
    assert depth == 0, f'unbalanced braces: depth ended at {depth}'


def _relabel_blocks() -> dict[str, str]:
    text = _read_config()
    blocks = {m.group('name'): m.group('body') for m in _BLOCK_RE.finditer(text)}
    assert blocks, 'no discovery.relabel blocks found -- config parsing regex is stale'
    return blocks


def test_expected_relabel_blocks_present() -> None:
    blocks = _relabel_blocks()
    assert set(blocks) == {'triton', 'fastapi'}, blocks


def _keep_regexes(block_body: str) -> list[re.Pattern[str]]:
    """The container-name `keep` rule is always the block's first
    `regex = "..."` (the later `stream`/`job` rules don't filter by name)."""
    matches = _REGEX_VALUE_RE.findall(block_body)
    assert matches, 'no regex = "..." filters found in block'
    return [re.compile(pattern) for pattern in matches]


def test_triton_filter_matches_this_composes_container_under_any_project_name() -> None:
    blocks = _relabel_blocks()
    keep_regex = _keep_regexes(blocks['triton'])[0]
    for project in _PROJECT_NAMES:
        container_name = f'/{project}-triton'
        assert keep_regex.search(container_name), (
            f'triton filter {keep_regex.pattern!r} does not match {container_name!r}'
        )


def test_fastapi_filter_matches_this_composes_container_under_any_project_name() -> None:
    blocks = _relabel_blocks()
    keep_regex = _keep_regexes(blocks['fastapi'])[0]
    for project in _PROJECT_NAMES:
        container_name = f'/{project}-api'
        assert keep_regex.search(container_name), (
            f'fastapi filter {keep_regex.pattern!r} does not match {container_name!r}'
        )


def test_filters_do_not_cross_match_unrelated_services() -> None:
    """The two filters must stay disjoint -- Triton logs tagged `job=fastapi`
    (or vice versa) would be a quieter, harder-to-notice regression than a
    filter that matches nothing at all."""
    blocks = _relabel_blocks()
    triton_regex = _keep_regexes(blocks['triton'])[0]
    fastapi_regex = _keep_regexes(blocks['fastapi'])[0]
    for project in _PROJECT_NAMES:
        assert not fastapi_regex.search(f'/{project}-triton')
        assert not triton_regex.search(f'/{project}-api')
        # Unrelated curation worker containers must not be swept up by
        # either filter (they aren't shipped to Loki by this config today).
        assert not triton_regex.search(f'/{project}-detection-worker')
        assert not fastapi_regex.search(f'/{project}-detection-worker')


def test_old_broken_literal_patterns_are_gone() -> None:
    """Checks the actual `regex = "..."` filter values (not just any
    occurrence in the file, which would also match this module's/the
    config's own explanatory comments about the old bug)."""
    blocks = _relabel_blocks()
    triton_patterns = [p.pattern for p in _keep_regexes(blocks['triton'])]
    fastapi_patterns = [p.pattern for p in _keep_regexes(blocks['fastapi'])]
    assert '/triton-server.*' not in triton_patterns, (
        f'old bug: bare triton-server filter still present: {triton_patterns}'
    )
    assert '/(yolo-api|pytorch-api).*' not in fastapi_patterns, (
        f'old bug: bare yolo-api/pytorch-api filter still present: {fastapi_patterns}'
    )
