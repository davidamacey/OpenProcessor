"""Exports honour the project's location policy (CM-5): a `drop` project's byte-copied images lose
their GPS tags losslessly; `store` and `store_coarse` copy the bytes as they are.

The files are generated: Pillow writes the JPEG and the real `exiftool` writes the GPS into it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from curation.test_export_per_image import _FilteringOpenSearch, _registry
from src.config import CurationConfig
from src.services.curation import export as export_module
from src.services.curation.export import GenericYoloExportService


def _jpeg(path: Path, *tags: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (40, 20), (10, 20, 30)).save(path, format='JPEG')
    if tags:
        subprocess.run(
            ['exiftool', '-overwrite_original', '-q', *tags, str(path)], check=True, timeout=60
        )


GPS_TAGS = (
    '-GPSLatitude=47.5',
    '-GPSLatitudeRef=N',
    '-GPSLongitude=8.25',
    '-GPSLongitudeRef=E',
    '-XMP:GPSLatitude=47.5',
    '-Make=Acme',
    '-DateTimeOriginal=2024:03:05 14:22:11',
)


def _read(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        ['exiftool', '-j', '-G', '-n', str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    import json

    return dict(json.loads(out)[0])


def _doc(image_id: str) -> dict[str, Any]:
    return {
        'crop_id': f'{image_id}-1',
        'image_id': image_id,
        'image_path': f'{image_id}.jpg',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_id': 1,
        'class_name': 'alpha',
        'class_validated': True,
        'test_holdout': False,
    }


async def _export(tmp_path: Path, policy: str | None) -> Path:
    cfg = CurationConfig(export_root=tmp_path / 'exports', source_root=tmp_path / 'src')

    async def source() -> str:
        assert policy is not None
        return policy

    service = GenericYoloExportService(
        _FilteringOpenSearch([_doc('gps'), _doc('plain')]),
        config=cfg,
        registry=_registry(tmp_path / 'reg'),
        location_policy=None if policy is None else source,
    )
    result = await service.export_dataset(seed=1)
    return Path(result.export_dir)


def _exported(export_dir: Path, stem: str) -> Path:
    [found] = list((export_dir / 'images').glob(f'*/{stem}.jpg'))
    return found


@pytest.fixture(autouse=True)
def _no_prune(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_module, 'prune_exports_after_write', lambda _config: None)


@pytest.fixture
def sources(tmp_path: Path) -> dict[str, Path]:
    gps, plain = tmp_path / 'src' / 'gps.jpg', tmp_path / 'src' / 'plain.jpg'
    _jpeg(gps, *GPS_TAGS)
    _jpeg(plain)
    return {'gps': gps, 'plain': plain}


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', ['store', 'store_coarse'])
async def test_store_and_store_coarse_copy_the_bytes_as_they_are(
    tmp_path: Path, sources: dict[str, Path], policy: str
) -> None:
    export_dir = await _export(tmp_path, policy)
    assert _exported(export_dir, 'gps').read_bytes() == sources['gps'].read_bytes()
    assert _exported(export_dir, 'plain').read_bytes() == sources['plain'].read_bytes()


@pytest.mark.asyncio
async def test_drop_removes_every_gps_tag_and_keeps_pixels_and_other_tags(
    tmp_path: Path, sources: dict[str, Path]
) -> None:
    export_dir = await _export(tmp_path, 'drop')

    stripped = _read(_exported(export_dir, 'gps'))
    original = _read(sources['gps'])
    assert any('GPS' in key for key in original)
    assert not [key for key in stripped if 'GPS' in key]
    assert stripped['EXIF:Make'] == 'Acme'
    assert stripped['EXIF:DateTimeOriginal'] == original['EXIF:DateTimeOriginal']
    with Image.open(_exported(export_dir, 'gps')) as got, Image.open(sources['gps']) as want:
        assert list(got.getdata()) == list(want.getdata())  # no re-encode: the same pixels


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', ['drop', 'store', 'store_coarse'])
async def test_a_file_without_gps_exports_byte_identical_under_every_policy(
    tmp_path: Path, sources: dict[str, Path], policy: str
) -> None:
    export_dir = await _export(tmp_path, policy)
    assert _exported(export_dir, 'plain').read_bytes() == sources['plain'].read_bytes()


@pytest.mark.asyncio
async def test_without_a_policy_source_the_gps_is_stripped(
    tmp_path: Path, sources: dict[str, Path]
) -> None:
    export_dir = await _export(tmp_path, None)
    assert not [key for key in _read(_exported(export_dir, 'gps')) if 'GPS' in key]
    assert sources['gps'].exists()


@pytest.mark.asyncio
async def test_a_failed_strip_fails_that_image_and_never_copies_it_whole(
    tmp_path: Path, sources: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.clients import exiftool_client
    from src.services.curation import export_support

    broken = exiftool_client.ExifToolClient(executable='/nonexistent/exiftool', timeout_s=5)
    monkeypatch.setattr(export_support, 'get_exiftool_client', lambda: broken)
    # run the copy in this process so the patched client is the one used
    _, ok, error = export_support._copy_or_resize_one(
        str(sources['gps']), str(tmp_path / 'out' / 'gps.jpg'), None, 640, None, True
    )
    assert not ok
    assert 'GPS strip' in (error or '')
    assert not (tmp_path / 'out' / 'gps.jpg').exists()
