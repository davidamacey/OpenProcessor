"""An exporter's generated ``config.pbtxt`` never silently replaces a tracked,
differing one (``make export-models`` used to dirty the git tree)."""

from __future__ import annotations

import sys
from pathlib import Path


EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

from config_write import write_generated_config  # noqa: E402


def test_a_new_config_is_created(tmp_path: Path) -> None:
    target = tmp_path / 'config.pbtxt'
    assert write_generated_config(target, 'generated') == target
    assert target.read_text() == 'generated'


def test_an_identical_config_is_not_rewritten(tmp_path: Path) -> None:
    target = tmp_path / 'config.pbtxt'
    target.write_text('same')
    before = target.stat().st_mtime_ns
    assert write_generated_config(target, 'same') == target
    assert target.stat().st_mtime_ns == before


def test_a_differing_config_is_kept_and_the_generated_one_goes_beside_it(tmp_path: Path) -> None:
    target = tmp_path / 'config.pbtxt'
    target.write_text('hand tuned')
    written = write_generated_config(target, 'generated')
    assert written == tmp_path / 'config.pbtxt.generated'
    assert written.read_text() == 'generated'
    assert target.read_text() == 'hand tuned'


def test_overwrite_replaces_a_differing_config(tmp_path: Path) -> None:
    target = tmp_path / 'config.pbtxt'
    target.write_text('hand tuned')
    assert write_generated_config(target, 'generated', overwrite=True) == target
    assert target.read_text() == 'generated'
    assert not (tmp_path / 'config.pbtxt.generated').exists()
