"""A path-keyed pre-commit hook can fail OPEN if a listed path is
renamed out from under it and nobody notices. This test converts that
into a loud test failure instead of a silent no-op.

Scope (see ``docs/design/curation_design_rationale.md`` §5 for the
ratchet-exemption rationale): the hooks this repo actually
authors/extends for curation-path ratcheting — the ``max-file-size``
exclude list (oversize ports get an entry each, in the commit that adds
the file) and ``check_no_literal_region_fields.py``'s ``PORTED_PATHS``
allowlist (see the same doc's §4). The repo's pre-existing top-level
``exclude:`` block (cache
dirs, ``.venv/``, etc.) is intentionally out of scope — those name
runtime artifacts that legitimately don't exist in a fresh checkout,
so "must exist on disk" is the wrong assertion for them.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_precommit_config() -> dict:
    with (REPO_ROOT / '.pre-commit-config.yaml').open() as fh:
        return yaml.safe_load(fh)


def _iter_hooks(config: dict):
    for repo in config.get('repos', []):
        yield from repo.get('hooks', [])


def _extract_path_fragments(exclude_regex: str) -> list[str]:
    """Pull literal-ish path fragments out of a `(?x)^(a|b|c)` style
    alternation block, as used by the max-file-size ratchet exclude.
    """
    # Strip the `(?x)^(` verbose-mode/anchor wrapper and trailing `)`.
    body = re.sub(r'^\s*\(\?x\)\s*', '', exclude_regex.strip())
    body = body.strip()
    body = re.sub(r'^\^?\(', '', body)
    body = re.sub(r'\)\s*$', '', body)

    fragments = []
    for raw in body.split('|'):
        frag = raw.strip()
        frag = frag.replace('\\.', '.')
        frag = frag.strip()
        if frag:
            fragments.append(frag.rstrip('/'))
    return fragments


def test_max_file_size_exclude_paths_exist_on_disk() -> None:
    config = _load_precommit_config()
    hook = next(h for h in _iter_hooks(config) if h.get('id') == 'max-file-size')
    exclude = hook.get('exclude', '')
    fragments = _extract_path_fragments(exclude)
    assert fragments, 'expected at least one excluded path in max-file-size'

    missing = [f for f in fragments if not (REPO_ROOT / f).exists()]
    assert not missing, (
        f'max-file-size exclude list names paths that no longer exist '
        f'on disk (renamed without updating the hook?): {missing}'
    )


def test_check_no_literal_region_fields_hook_is_registered() -> None:
    config = _load_precommit_config()
    hook = next(h for h in _iter_hooks(config) if h.get('id') == 'check-no-literal-region-fields')
    assert 'check_no_literal_region_fields.py' in hook['entry']
    assert (REPO_ROOT / 'scripts/codegen/check_no_literal_region_fields.py').is_file()


def test_check_no_literal_region_fields_allowlist_paths_exist_on_disk() -> None:
    from scripts.codegen.check_no_literal_region_fields import PORTED_PATHS

    missing = [p for p in PORTED_PATHS if not (REPO_ROOT / p).exists()]
    assert not missing, (
        f'check_no_literal_region_fields.PORTED_PATHS names paths that '
        f'do not exist on disk (renamed without updating the allowlist?): '
        f'{missing}'
    )


def test_check_no_literal_region_fields_allowlist_only_grows_with_real_ports() -> None:
    """The allowlist is expected to grow monotonically as files are
    ported — see `test_...allowlist_paths_exist_on_disk`
    above for the standing invariant. This test only pins that entries
    are unique (a duplicate entry would be a copy-paste mistake, not a
    real new port)."""
    from scripts.codegen.check_no_literal_region_fields import PORTED_PATHS

    assert len(PORTED_PATHS) == len(set(PORTED_PATHS)), 'duplicate PORTED_PATHS entry'
