"""A path-keyed pre-commit hook can fail OPEN if a listed path is
renamed out from under it and nobody notices. This test converts that
into a loud test failure instead of a silent no-op.

Scope (see ``docs/design/curation_design_rationale.md`` §4 and §5): the
``max-file-size`` hook, which carries no exemptions, and
``check_no_literal_region_fields.py``'s ``PORTED_PATHS`` allowlist. The
repo's top-level ``exclude:`` block (cache dirs, ``.venv/``, etc.) is out
of scope — those name runtime artifacts that legitimately don't exist in a
fresh checkout, so "must exist on disk" is the wrong assertion for them.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_precommit_config() -> dict:
    with (REPO_ROOT / '.pre-commit-config.yaml').open() as fh:
        return yaml.safe_load(fh)


def _iter_hooks(config: dict):
    for repo in config.get('repos', []):
        yield from repo.get('hooks', [])


def test_max_file_size_hook_has_no_exemptions() -> None:
    """Every source file is under the cap, so the hook carries no exclude.
    An exclude with an empty alternation (``^()``) would match every path and
    silently disable the ratchet, so any exclude at all fails here."""
    config = _load_precommit_config()
    hook = next(h for h in _iter_hooks(config) if h.get('id') == 'max-file-size')
    assert 'exclude' not in hook, 'split the oversize module instead of excluding it'


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
