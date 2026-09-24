"""S7: RegionStatus -> TypeScript codegen and its --check drift guard."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.config.region_state import PENDING_STATUSES, TERMINAL_STATUSES, RegionStatus


if TYPE_CHECKING:
    import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / 'scripts' / 'codegen' / 'export_region_status_to_ts.py'
COMMITTED = REPO_ROOT / 'contracts' / 'ts' / 'regionStatus.ts'


def _load_module():
    spec = importlib.util.spec_from_file_location('export_region_status_to_ts', SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_passes_on_committed_output() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), '--check'],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_committed_output_uses_generic_vocabulary() -> None:
    text = COMMITTED.read_text(encoding='utf-8')
    for member in RegionStatus:
        assert f"  {member.name} = '{member.value}'," in text
    assert "'no_region_box'" in text
    assert "'no_region_visible'" in text
    assert 'plate' not in text


def test_subsets_follow_python_sets() -> None:
    mod = _load_module()
    text = mod.render_ts()
    terminal_block = text.split('TERMINAL_REGION_STATUSES')[1].split('] as const;')[0]
    pending_block = text.split('PENDING_REGION_STATUSES')[1].split('] as const;')[0]
    for member in RegionStatus:
        assert (f"'{member.value}'" in terminal_block) == (member in TERMINAL_STATUSES)
        assert (f"'{member.value}'" in pending_block) == (member in PENDING_STATUSES)


def test_check_fails_on_stale_file(tmp_path: Path) -> None:
    mod = _load_module()
    stale = tmp_path / 'regionStatus.ts'
    stale.write_text(mod.render_ts().replace("'no_region_box'", "'no_box'"), encoding='utf-8')
    assert mod.main(['--check', str(stale)]) == 1


def test_check_fails_on_missing_file(tmp_path: Path) -> None:
    mod = _load_module()
    assert mod.main(['--check', str(tmp_path / 'absent.ts')]) == 1


def test_write_then_check_roundtrip(tmp_path: Path) -> None:
    mod = _load_module()
    target = tmp_path / 'nested' / 'regionStatus.ts'
    assert mod.main([str(target)]) == 0
    assert mod.main(['--check', str(target)]) == 0


def test_check_detects_enum_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A new Python member must make the old committed file fail --check."""
    mod = _load_module()
    target = tmp_path / 'regionStatus.ts'
    target.write_text(mod.render_ts(), encoding='utf-8')
    real = mod._members()
    monkeypatch.setattr(mod, '_members', lambda: [*real, ('NEW_STATE', 'new_state')])
    assert mod.main(['--check', str(target)]) == 1


def test_human_writable_subset_and_roles_follow_python() -> None:
    from src.config.region_state import HUMAN_WRITABLE_STATUSES, REGION_STATUS_INFO

    text = _load_module().render_ts()
    block = text.split('HUMAN_WRITABLE_REGION_STATUSES')[1].split('] as const;')[0]
    for member in RegionStatus:
        assert (f"'{member.value}'" in block) == (member in HUMAN_WRITABLE_STATUSES)
        assert f"  {member.value}: '{REGION_STATUS_INFO[member].role}'," in text
