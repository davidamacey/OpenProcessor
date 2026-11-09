"""ST-4: keep-last retention for auto-named export directories.

Before this module existed, ``export_dataset`` had no retention at all --
every timestamped export accumulated forever. These tests lock down the
planning logic in isolation (tmp filesystem roots, no OpenSearch/Triton).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.curation import export_retention
from src.services.curation.export_retention import (
    apply_export_prune,
    collect_export_pins,
    plan_export_prune,
)


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _make_export_dir(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / 'manifest.json').write_text('{}')
    return d


def _timestamped_names(n: int) -> list[str]:
    # Strictly increasing YYYYMMDDTHHMMSSZ names, oldest first.
    return [f'202601{(i // 24) + 1:02d}T{i % 24:02d}0000Z' for i in range(n)]


class TestPlanExportPrune:
    def test_keeps_last_n_skips_pinned_never_touches_custom_or_current(
        self, tmp_path: Path
    ) -> None:
        export_root = tmp_path / 'exports'
        export_root.mkdir()
        names = _timestamped_names(8)
        dirs = {name: _make_export_dir(export_root, name) for name in names}
        _make_export_dir(export_root, 'custom_named_export')  # custom-named, never touched

        # `current` points at the 3rd-newest (names sorted descending -> index 2).
        sorted_desc = sorted(names, reverse=True)
        current_target = dirs[sorted_desc[2]]
        (export_root / 'current').symlink_to(current_target, target_is_directory=True)

        # Pin one additional older dir via a fake job.json / manifest.json pin.
        pinned_by_job = dirs[sorted_desc[5]]
        pinned_by_run = dirs[sorted_desc[6]]

        pins = {current_target.resolve(), pinned_by_job.resolve(), pinned_by_run.resolve()}

        keep_last = 3
        plan = plan_export_prune(export_root, keep_last, pins)
        plan_names = {p.name for p in plan}

        # Never the custom-named dir.
        assert 'custom_named_export' not in plan_names
        # Never `current`'s target.
        assert sorted_desc[2] not in plan_names
        # Never a pinned dir.
        assert sorted_desc[5] not in plan_names
        assert sorted_desc[6] not in plan_names
        # Never one of the newest `keep_last`.
        assert not (set(sorted_desc[:keep_last]) & plan_names)
        # Everything else beyond keep_last and unpinned IS planned.
        expected = set(sorted_desc[keep_last:]) - {sorted_desc[5], sorted_desc[6]}
        assert plan_names == expected

    def test_keep_last_zero_keeps_everything(self, tmp_path: Path) -> None:
        export_root = tmp_path / 'exports'
        export_root.mkdir()
        for name in _timestamped_names(5):
            _make_export_dir(export_root, name)
        assert plan_export_prune(export_root, 0, set()) == []

    def test_missing_export_root_is_a_noop(self, tmp_path: Path) -> None:
        assert plan_export_prune(tmp_path / 'does-not-exist', 5, set()) == []


class TestCollectExportPins:
    def test_collects_run_and_bakeoff_dataset_pins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        export_root = tmp_path / 'exports'
        export_root.mkdir()
        run_export = _make_export_dir(export_root, '20260101T000000Z')
        bakeoff_export = _make_export_dir(export_root, '20260103T000000Z')

        monkeypatch.setattr(export_retention, '_run_export_dirs', lambda: {str(run_export)})
        monkeypatch.setattr(
            export_retention,
            'active_bakeoff_specs',
            lambda: [{'datasets': [{'path': str(bakeoff_export)}]}],
        )

        pins = collect_export_pins(export_root=export_root)
        assert run_export.resolve() in pins
        assert bakeoff_export.resolve() in pins

    def test_no_runs_and_no_bakeoffs_pin_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(export_retention, '_run_export_dirs', set)
        monkeypatch.setattr(export_retention, 'active_bakeoff_specs', list)
        export_root = tmp_path / 'exports'
        export_root.mkdir()

        assert collect_export_pins(export_root=export_root) == set()


class TestApplyExportPrune:
    def test_removes_planned_dirs_and_reports_bytes(self, tmp_path: Path) -> None:
        export_root = tmp_path / 'exports'
        export_root.mkdir()
        d = _make_export_dir(export_root, '20260101T000000Z')
        (d / 'images').mkdir()
        (d / 'images' / 'a.jpg').write_bytes(b'x' * 1000)

        result = apply_export_prune([d])
        assert result['removed'] == 1
        assert result['removed_bytes'] >= 1000
        assert not d.exists()
