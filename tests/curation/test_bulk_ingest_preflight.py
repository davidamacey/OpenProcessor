"""Bulk-ingest preflight: what a folder, manifest or S3 prefix holds and what ingesting it
would cost, with no ingestion. The walk is bounded and cancellable, an unreadable root
refuses, and every estimate figure is a measured one from docs/PERFORMANCE.md."""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import pytest
from _fake_object_store import FakeObjectStore
from PIL import Image

from src.services.curation import image_serving
from src.services.curation.bulk_ingest import preflight, preflight_rates
from src.services.curation.bulk_ingest.object_store import S3Config, S3NotAllowedError
from src.services.curation.bulk_ingest.source import BulkSource, InvalidSourceError, validate_source
from src.services.curation.bulk_ingest.walk import WalkStats, iter_image_paths


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        image_serving,
        '_configured_roots',
        lambda config=None: (tmp_path.resolve(),),  # noqa: ARG005
    )


def _img(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, (10, 20, 30)).save(path, format='JPEG')


def _folder(tmp_path: Path) -> Path:
    root = tmp_path / 'photos'
    for i in range(3):
        _img(root / f'a{i}.jpg')
    _img(root / 'sub' / 'b.JPG')
    Image.new('RGB', (8, 8)).save(root / 'sub' / 'c.png')
    (root / 'notes.txt').write_text('x')
    return root


def _preview(source: BulkSource, **kw: object) -> preflight.PreviewResult:
    return preflight.preview_source(source, **kw)  # type: ignore[arg-type]


def test_folder_counts_bytes_and_extensions(tmp_path: Path) -> None:
    root = _folder(tmp_path)
    result = _preview(BulkSource('folder', str(root)))
    assert result.files == 5
    assert result.total_bytes == sum(
        p.stat().st_size for p in root.rglob('*') if p.suffix.lower() in {'.jpg', '.png'}
    )
    assert {e: v.files for e, v in result.by_extension.items()} == {'.jpg': 4, '.png': 1}
    assert result.partial is False
    assert result.stop_reason is None


def test_symlinks_and_unreadable_directories_are_reported_not_hidden(tmp_path: Path) -> None:
    root = _folder(tmp_path)
    (root / 'link.jpg').symlink_to(root / 'a0.jpg')
    locked = root / 'locked'
    _img(locked / 'x.jpg')
    locked.chmod(0)
    try:
        result = _preview(BulkSource('folder', str(root)))
    finally:
        locked.chmod(0o755)
    assert result.files == 5
    assert result.symlinks_skipped == 1
    assert result.unreadable_dirs == 1
    assert result.unreadable_samples == [str(locked)]


def test_unreadable_root_refuses(tmp_path: Path) -> None:
    root = _folder(tmp_path)
    root.chmod(0)
    try:
        with pytest.raises(InvalidSourceError, match='not readable'):
            _preview(BulkSource('folder', str(root)))
        with pytest.raises(InvalidSourceError, match='not readable'):
            validate_source(BulkSource('folder', str(root)))
    finally:
        root.chmod(0o755)


def test_a_root_outside_the_source_roots_refuses(tmp_path: Path) -> None:
    with pytest.raises(InvalidSourceError, match='source root'):
        _preview(BulkSource('folder', '/etc'))


def test_the_file_cap_stops_the_walk_and_says_partial(tmp_path: Path) -> None:
    root = _folder(tmp_path)
    result = _preview(BulkSource('folder', str(root)), limits=preflight.PreviewLimits(max_files=2))
    assert result.files == 2
    assert (result.partial, result.stop_reason) == (True, 'file_cap')


def test_the_time_cap_stops_the_walk(tmp_path: Path) -> None:
    root = _folder(tmp_path)
    ticks = iter(range(0, 1000, 10))
    result = _preview(
        BulkSource('folder', str(root)),
        limits=preflight.PreviewLimits(max_seconds=25),
        clock=lambda: float(next(ticks)),
    )
    assert (result.partial, result.stop_reason) == (True, 'time_cap')
    assert result.files < 5


def test_a_cancel_stops_the_walk(tmp_path: Path) -> None:
    root = _folder(tmp_path)
    result = _preview(BulkSource('folder', str(root)), should_stop=lambda: True)
    assert (result.partial, result.stop_reason) == (True, 'cancelled')


def test_closing_the_walk_early_does_not_scan_the_rest_of_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / 'wide'
    for i in range(40):
        _img(root / f'd{i}' / 'x.jpg')
    scanned: list[str] = []
    real = os.scandir

    def counting(path: str | os.PathLike[str]) -> object:
        scanned.append(str(path))
        return real(path)

    monkeypatch.setattr(os, 'scandir', counting)
    walk = iter_image_paths(root, frozenset({'.jpg'}), workers=1)
    next(walk)
    walk.close()
    assert len(scanned) < 41


def test_a_stalled_directory_listing_cannot_hold_a_stopped_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _folder(tmp_path)
    gate = threading.Event()
    real = os.scandir

    def stalled(path: str | os.PathLike[str]) -> object:
        gate.wait(10)
        return real(path)

    monkeypatch.setattr(os, 'scandir', stalled)
    started = time.monotonic()
    try:
        found = list(iter_image_paths(root, frozenset({'.jpg'}), 1, should_stop=lambda: True))
        assert found == []
        assert time.monotonic() - started < 3
    finally:
        gate.set()


def test_walk_stats_count_what_the_walk_skips(tmp_path: Path) -> None:
    root = _folder(tmp_path)
    (root / 'l.jpg').symlink_to(root / 'a0.jpg')
    stats = WalkStats()
    list(iter_image_paths(root, frozenset({'.jpg'}), workers=2, stats=stats))
    assert stats.symlinks == 1
    assert stats.unreadable_dirs == 0


def test_the_megapixel_sample_buckets_headers_without_decoding(tmp_path: Path) -> None:
    root = tmp_path / 'mix'
    for i in range(4):
        _img(root / f'small{i}.jpg', (100, 100))  # 0.01 MP
    for i in range(2):
        _img(root / f'big{i}.jpg', (2400, 1600))  # 3.84 MP, above the 3.0 MP gate
    result = _preview(BulkSource('folder', str(root)), gpu_decode=True, gpu_min_megapixels=3.0)
    sample = result.megapixels
    assert sample.sampled == 6
    assert sample.unreadable == 0
    assert sample.share_at_or_above_gate == pytest.approx(2 / 6)
    assert sum(sample.histogram.values()) == 6


def test_the_sample_is_bounded(tmp_path: Path) -> None:
    root = tmp_path / 'many'
    for i in range(30):
        _img(root / f'{i}.jpg')
    limits = preflight.PreviewLimits(header_sample=10)
    first = _preview(BulkSource('folder', str(root)), limits=limits)
    second = _preview(BulkSource('folder', str(root)), limits=limits)
    assert first.megapixels.sampled == 10
    assert first.megapixels == second.megapixels


def test_estimate_uses_the_measured_rates_and_footprint_only(tmp_path: Path) -> None:
    root = tmp_path / 'm'
    for i in range(4):
        _img(root / f's{i}.jpg', (100, 100))
    rates = {r.route: r for r in preflight_rates.RATES}
    result = _preview(BulkSource('folder', str(root)), gpu_decode=True, gpu_min_megapixels=3.0)
    est = result.estimate
    assert est.basis == 'megapixel_sample'
    assert (
        est.duration_s_min
        == est.duration_s_max
        == pytest.approx(4 / rates['cpu_small'].images_per_s)
    )
    assert est.vectors == pytest.approx(4 * preflight_rates.EMBEDDINGS_PER_IMAGE.value)
    assert est.opensearch_bytes == round(4 * preflight_rates.OPENSEARCH_BYTES_PER_IMAGE.value)


def test_large_photos_use_the_cpu_rate_when_gpu_decode_is_off(tmp_path: Path) -> None:
    root = tmp_path / 'm'
    _img(root / 'big.jpg', (2400, 1600))
    rates = {r.route: r for r in preflight_rates.RATES}
    on = _preview(BulkSource('folder', str(root)), gpu_decode=True, gpu_min_megapixels=3.0)
    off = _preview(BulkSource('folder', str(root)), gpu_decode=False, gpu_min_megapixels=3.0)
    assert on.estimate.duration_s_min == pytest.approx(1 / rates['gpu_large'].images_per_s)
    assert off.estimate.duration_s_min == pytest.approx(1 / rates['cpu_large'].images_per_s)


def test_without_headers_the_estimate_is_a_range_across_the_measured_routes() -> None:
    est = preflight_rates.estimate(files=100, share_large=None, gpu_decode=True)
    rates = [r.images_per_s for r in preflight_rates.RATES]
    assert est.basis == 'rate_range'
    assert est.duration_s_min == pytest.approx(100 / max(rates))
    assert est.duration_s_max == pytest.approx(100 / min(rates))


def test_every_rate_and_footprint_figure_is_in_performance_md() -> None:
    text = (REPO / 'docs' / 'PERFORMANCE.md').read_text(encoding='utf-8')
    figures = [r.images_per_s for r in preflight_rates.RATES] + [
        preflight_rates.EMBEDDINGS_PER_IMAGE.value,
        preflight_rates.OPENSEARCH_BYTES_PER_IMAGE.value,
    ]
    for value in figures:
        shown = str(value) if isinstance(value, int) else f'{value:.1f}'
        assert re.search(rf'(?<![\d.]){re.escape(shown)}(?![\d])', text), value
    for measured in [
        *preflight_rates.RATES,
        preflight_rates.EMBEDDINGS_PER_IMAGE,
        preflight_rates.OPENSEARCH_BYTES_PER_IMAGE,
    ]:
        assert measured.source_section in text


def test_manifest_counts_missing_and_out_of_root_entries(tmp_path: Path) -> None:
    root = _folder(tmp_path)
    manifest = tmp_path / 'list.txt'
    manifest.write_text(
        '\n'.join(
            [
                '# comment',
                str(root / 'a0.jpg'),
                str(root / 'a1.jpg'),
                str(root / 'gone.jpg'),
                '/etc/passwd.jpg',
                '',
            ]
        )
    )
    result = _preview(BulkSource('manifest', str(manifest)))
    assert result.files == 2
    assert result.missing == 1
    assert result.outside_roots == 1


def _s3_store() -> FakeObjectStore:
    store = FakeObjectStore(page_size=2)
    for i in range(7):
        store.put('b', f'p/{i}.jpg', b'x' * 10)
    store.put('b', 'p/readme.txt', b'no')
    return store


def test_s3_prefix_is_counted_by_listing_only() -> None:
    store = _s3_store()
    config = S3Config(allowed=(('b', 'p/'),))
    result = preflight.preview_s3(store, config, 'b', 'p/', ('.jpg',))
    assert (result.files, result.total_bytes) == (7, 70)
    assert result.megapixels.sampled == 0
    assert result.estimate.basis == 'rate_range'
    assert store.get_calls == []


def test_s3_listing_is_capped_and_outside_the_allow_list_refuses() -> None:
    config = S3Config(allowed=(('b', 'p/'),))
    capped = preflight.preview_s3(
        _s3_store(), config, 'b', 'p/', ('.jpg',), limits=preflight.PreviewLimits(max_files=3)
    )
    assert (capped.partial, capped.stop_reason) == (True, 'file_cap')
    with pytest.raises(S3NotAllowedError):
        preflight.preview_s3(_s3_store(), config, 'other', 'p/', ('.jpg',))
