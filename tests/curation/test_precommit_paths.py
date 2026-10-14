"""A path-keyed pre-commit hook can fail OPEN if a listed path is
renamed out from under it and nobody notices. This test converts that
into a loud test failure instead of a silent no-op.

Scope (see ``docs/design/curation_design_rationale.md`` §4 and §5): the
``max-file-size`` hook, which carries no exemptions. That every other
hook's ``files``/``exclude`` path exists is asserted in
``tests/test_tooling_config.py``.
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
