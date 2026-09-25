"""ST-4: keep-last retention for auto-named export directories.

Before this module existed, ``export_dataset`` had no retention at all --
every timestamped export accumulated forever. These tests lock down the
planning logic in isolation (tmp filesystem roots, no OpenSearch/Triton).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.services.curation.export_retention import (
    apply_export_prune,
    collect_export_pins,
    plan_export_prune,
)


if TYPE_CHECKING:
    from pathlib import Path


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
    def test_collects_job_manifest_and_bakeoff_dataset_pins(self, tmp_path: Path) -> None:
        export_root = tmp_path / 'exports'
        export_root.mkdir()
        job_export = _make_export_dir(export_root, '20260101T000000Z')
        run_export = _make_export_dir(export_root, '20260102T000000Z')
        bakeoff_export = _make_export_dir(export_root, '20260103T000000Z')

        jobs_dir = tmp_path / 'jobs'
        jobs_dir.mkdir()
        (jobs_dir / 'job1.job.json').write_text(json.dumps({'dataset_export_dir': str(job_export)}))
        (jobs_dir / 'job1.manifest.json').write_text(
            json.dumps({'lineage': {'export_dir': str(run_export)}})
        )

        bakeoff_jobs_dir = tmp_path / 'bakeoff_jobs'
        bakeoff_jobs_dir.mkdir()
        (bakeoff_jobs_dir / 'b1.job.json').write_text(
            json.dumps({'datasets': [{'path': str(bakeoff_export)}]})
        )

        pins = collect_export_pins(
            export_root=export_root, jobs_dir=jobs_dir, bakeoff_jobs_dir=bakeoff_jobs_dir
        )
        assert job_export.resolve() in pins
        assert run_export.resolve() in pins
        assert bakeoff_export.resolve() in pins

    def test_missing_dirs_and_malformed_json_are_skipped(self, tmp_path: Path) -> None:
        export_root = tmp_path / 'exports'
        export_root.mkdir()
        jobs_dir = tmp_path / 'jobs'
        jobs_dir.mkdir()
        (jobs_dir / 'bad.job.json').write_text('{not json')
        bakeoff_jobs_dir = tmp_path / 'missing_bakeoff_jobs'

        pins = collect_export_pins(
            export_root=export_root, jobs_dir=jobs_dir, bakeoff_jobs_dir=bakeoff_jobs_dir
        )
        assert pins == set()


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
