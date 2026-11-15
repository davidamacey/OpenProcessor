"""``scripts/bench/frame_engine_sets.py``: the lists of the N0 comparison, on tiny fixture JPEGs
whose headers are patched to the sizes under test (the script reads headers only)."""

from __future__ import annotations

import hashlib
import io
import json
import random
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image, ImageCms

from scripts.bench import frame_engine_sets as fes


if TYPE_CHECKING:
    from pathlib import Path

#: sRGB-class and not-RGB profiles as littleCMS knows them
SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
LAB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile('LAB')).tobytes()


def _srgb_class(icc: bytes | None) -> str:
    from src.utils.icc_srgb import profile_kind

    return profile_kind(icc)


def jpeg(
    mode: str = 'RGB',
    *,
    progressive: bool = False,
    icc: bytes | None = None,
    size: tuple[int, int] = (16, 16),
) -> bytes:
    out = io.BytesIO()
    kwargs: dict[str, Any] = {'progressive': progressive}
    if icc is not None:
        kwargs['icc_profile'] = icc
    Image.new(mode, size).save(out, 'JPEG', **kwargs)
    return out.getvalue()


def resized(data: bytes, width: int, height: int) -> bytes:
    """The same file with the frame header claiming ``width`` x ``height``: all the script reads."""
    i = next(
        k for k in range(len(data) - 1) if data[k] == 0xFF and data[k + 1] in (0xC0, 0xC1, 0xC2)
    )
    patched = bytearray(data)
    patched[i + 5 : i + 9] = height.to_bytes(2, 'big') + width.to_bytes(2, 'big')
    return bytes(patched)


def reason_of(data: bytes, tmp_path: Path, window: tuple[float, float] | None = None) -> str | None:
    path = tmp_path / 'x.jpg'
    path.write_bytes(data)
    return fes.exclusion(fes.read_header(path), window, _srgb_class)


def test_a_plain_baseline_rgb_jpeg_is_kept(tmp_path: Path) -> None:
    assert reason_of(jpeg(), tmp_path) is None


@pytest.mark.parametrize(
    ('data', 'reason'),
    [
        (jpeg(progressive=True), fes.SOF_NOT_SEQUENTIAL),
        (jpeg('L'), fes.COMPONENTS),
        (jpeg('CMYK'), fes.COMPONENTS),
        (b'\x89PNG\r\n\x1a\n' + b'0' * 64, fes.NOT_JPEG),
        (b'\xff\xd8' + b'\x00' * 64, fes.NO_FRAME_HEADER),
        (jpeg(icc=LAB_ICC), fes.PROFILE),
    ],
)
def test_each_unsuitable_file_is_excluded_under_its_own_reason(
    data: bytes, reason: str, tmp_path: Path
) -> None:
    assert reason_of(data, tmp_path) == reason


def test_an_srgb_profile_is_kept_and_a_converting_one_is_not() -> None:
    header = fes.scan(jpeg(icc=SRGB_ICC))
    assert header.icc == SRGB_ICC
    assert fes.exclusion(header, None, _srgb_class) is None
    assert fes.exclusion(header, None, lambda _icc: 'convert') == fes.PROFILE


def test_an_arithmetic_or_12_bit_frame_header_is_excluded() -> None:
    base = fes.scan(jpeg())
    assert fes.exclusion(base._replace(sof=0xC9), None, _srgb_class) == fes.SOF_NOT_SEQUENTIAL
    assert fes.exclusion(base._replace(sof=0xC1), None, _srgb_class) is None
    assert fes.exclusion(base._replace(precision=12), None, _srgb_class) == fes.PRECISION


@pytest.mark.parametrize(
    ('window', 'size', 'reason'),
    [
        ((0.0, 2.0), (1600, 1000), None),
        ((0.0, 2.0), (2000, 1000), None),  # exactly 2 MP is inside
        ((0.0, 2.0), (2001, 1000), fes.ABOVE_WINDOW),
        ((18.0, 24.5), (5000, 4000), None),
        ((18.0, 24.5), (4000, 3000), fes.BELOW_WINDOW),
        ((18.0, 24.5), (6000, 4084), fes.ABOVE_WINDOW),  # 24.504 MP
        ((18.0, 24.5), (6125, 4000), None),  # exactly 24.5 MP
    ],
)
def test_the_pixel_window_comes_from_the_header_not_the_file(
    window: tuple[float, float], size: tuple[int, int], reason: str | None, tmp_path: Path
) -> None:
    assert reason_of(resized(jpeg(), *size), tmp_path, window) == reason


def test_a_large_icc_profile_is_read_whole_from_a_bounded_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = bytes(random.Random(7).randbytes(300_000))
    path = tmp_path / 'big.jpg'
    # the photo's own bytes follow the header: a megabytes-long tail the script must not read
    path.write_bytes(jpeg(icc=profile) + b'\x00' * 3_000_000)
    read = 0
    real_open = type(path).open

    class Counting:
        def __init__(self, f: Any) -> None:
            self.f = f

        def __enter__(self) -> Counting:
            return self

        def __exit__(self, *exc: object) -> None:
            self.f.close()

        def seek(self, n: int) -> None:
            self.f.seek(n)

        def read(self, n: int) -> bytes:
            nonlocal read
            data: bytes = self.f.read(n)
            read += len(data)
            return data

    monkeypatch.setattr(
        type(path), 'open', lambda self, *a, **k: Counting(real_open(self, *a, **k))
    )
    header = fes.read_header(path)
    assert header.icc == profile
    assert header.sof == 0xC0
    assert read < 1_000_000 + 400_000, f'{read} bytes read of a 3.3 MB file'


def test_the_kept_files_are_shuffled_by_the_seed_in_source_order(tmp_path: Path) -> None:
    paths = []
    for k in range(10):
        p = tmp_path / f'{k}.jpg'
        p.write_bytes(jpeg())
        paths.append(p)
    rule = fes.Rule('t', seed=1, n=4, min_mp=None, max_mp=None)
    sel = fes.select(paths, rule, _srgb_class)
    # the order the standard library's Random with seed 1 gives ten items: 6, 8, 9, 7, 5, ...
    assert [p.name for p in sel.paths] == ['6.jpg', '8.jpg', '9.jpg', '7.jpg']
    assert (sel.considered, sel.kept, sel.excluded) == (10, 10, {})


def test_exclusions_are_counted_once_each_by_reason_and_missing_files_by_name(
    tmp_path: Path,
) -> None:
    files = {
        'ok.jpg': jpeg(),
        'prog.jpg': jpeg(progressive=True),
        'prog2.jpg': jpeg(progressive=True),
        'gray.jpg': jpeg('L'),
        'big.jpg': resized(jpeg(), 4000, 4000),
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    paths = [tmp_path / n for n in [*files, 'gone.jpg']]
    rule = fes.Rule('t', seed=1, n=10, min_mp=None, max_mp=2.0)
    sel = fes.select(paths, rule, _srgb_class)
    assert [p.name for p in sel.paths] == ['ok.jpg']
    assert sel.excluded == {
        fes.ABOVE_WINDOW: 1,
        fes.COMPONENTS: 1,
        fes.MISSING: 1,
        fes.SOF_NOT_SEQUENTIAL: 2,
    }
    assert sum(sel.excluded.values()) + sel.kept == sel.considered


def test_presets_carry_the_plans_numbers() -> None:
    assert fes.PRESETS['coco'] == fes.Rule('coco', seed=1, n=1000, min_mp=None, max_mp=2.0)
    assert fes.PRESETS['hires'] == fes.Rule('hires', seed=3, n=120, min_mp=18.0, max_mp=24.5)


def test_main_writes_a_list_and_its_report_with_the_files_own_sha256(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    images = tmp_path / 'images'
    images.mkdir()
    names = ['a.jpg', 'b.jpg', 'c.jpg']
    for name in names:
        (images / name).write_bytes(jpeg())
    (images / 'c.jpg').write_bytes(jpeg('L'))
    manifest = tmp_path / 'pin.json'
    manifest.write_text(json.dumps({'images': [{'file_name': n} for n in names]}))
    out = tmp_path / 'lists'
    code = fes.main(
        [
            '--preset', 'custom', '--name', 'tiny', '--seed', '1', '--n', '5',
            '--manifest', str(manifest), '--images-dir', str(images), '--out-dir', str(out),
        ]
    )  # fmt: skip
    assert code == 0
    report = json.loads((out / 'tiny.json').read_text())
    listing = out / 'tiny.txt'
    assert report['list_sha256'] == hashlib.sha256(listing.read_bytes()).hexdigest()
    assert report['photos'] == 2
    assert report['excluded_by_reason'] == {fes.COMPONENTS: 1}
    assert report['source']['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert sorted(ln.rsplit('/', 1)[1] for ln in listing.read_text().split()) == ['a.jpg', 'b.jpg']
    assert json.loads(capsys.readouterr().out)['list_sha256'] == report['list_sha256']
