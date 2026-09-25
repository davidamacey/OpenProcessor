"""CLI safety-default guard for ``scripts/curation/backfill_scores.py``.

The script can write new fields onto all 347,837 the configured items index's docs
if run unbounded — ``--dry-run`` must be the default so an operator has to
explicitly opt into `--apply` rather than trip a full-corpus write by
omission. Loaded via ``importlib`` (the ``scripts/`` tree isn't a package)
so this exercises the real ``argparse`` wiring in ``main()``, not a
reimplementation of it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from types import ModuleType


def _load_script() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / 'scripts' / 'curation' / 'backfill_scores.py'
    spec = importlib.util.spec_from_file_location('curation_backfill_scores_cli_test', script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backfill_defaults_to_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_script()
    captured: dict[str, object] = {}

    async def fake_async_main(args):
        captured['dry_run'] = args.dry_run
        captured['limit'] = args.limit
        return 0

    monkeypatch.setattr(mod, '_async_main', fake_async_main)
    monkeypatch.setattr(sys, 'argv', ['backfill_scores.py'])

    rc = mod.main()

    assert rc == 0
    assert captured['dry_run'] is True, 'backfill_scores.py must default to --dry-run'


def test_backfill_apply_flag_disables_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_script()
    captured: dict[str, object] = {}

    async def fake_async_main(args):
        captured['dry_run'] = args.dry_run
        return 0

    monkeypatch.setattr(mod, '_async_main', fake_async_main)
    monkeypatch.setattr(sys, 'argv', ['backfill_scores.py', '--apply'])

    mod.main()

    assert captured['dry_run'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
