"""The converter registry and the generic YOLO writer in ``datasets``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.bakeoff_harness._helpers import HARNESS, _img
from scripts.curation.bakeoff import datasets
from scripts.curation.bakeoff.profile import GENERIC_PROFILE, BakeoffProfile


if TYPE_CHECKING:
    from pathlib import Path


# --- datasets.py: registry + generic writer ------------------------------------


def test_datasets_has_no_hardcoded_class_constant() -> None:
    assert not hasattr(datasets, 'LPR_CLASS_ID')
    src = (HARNESS / 'datasets.py').read_text()
    assert 'license_plate' not in src


def test_builtin_converters_are_generic_only() -> None:
    names = set(datasets.available_converters())
    assert {'voc', 'yolo'} <= names
    assert all(
        c.example_for is None
        for c in datasets.available_converters().values()
        if c.name in {'voc', 'yolo'}
    )


def test_writer_data_yaml_uses_profile_class_names(tmp_path: Path) -> None:
    prof = BakeoffProfile(name='shop', class_names=('pallet', 'box'))
    w = datasets.YoloWriter.for_profile(tmp_path, prof, split='test')
    w.write_data_yaml()
    yaml = (tmp_path / 'data.yaml').read_text()
    assert 'nc: 2\n' in yaml
    assert '  0: pallet\n  1: box\n' in yaml
    with pytest.raises(ValueError, match='outside nc'):
        w.write('a', _img(tmp_path / 'src/a.png'), [(5, 0.5, 0.5, 0.1, 0.1)])


def test_writer_single_class_collapses_to_zero(tmp_path: Path) -> None:
    w = datasets.YoloWriter(tmp_path, ('thing',))
    assert not hasattr(w, 'target_class_id')
    assert (w.map_class_id(7), w.map_class_name('anything')) == (0, 0)
    with pytest.raises(TypeError):
        datasets.YoloWriter(tmp_path, ('thing',), target_class_id=0)  # type: ignore[call-arg]


def _voc(path: Path, objects: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    objs = ''.join(
        f'<object><name>{n}</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>'
        f'<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>'
        for n, b in objects
    )
    path.write_text(
        f'<annotation><size><width>100</width><height>50</height></size>{objs}</annotation>'
    )


def test_voc_converter_maps_names_multiclass(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    img = _img(src / 'f1.png')
    _voc(
        img.with_suffix('.xml'),
        [('Box', (0, 0, 50, 50)), ('pallet', (50, 0, 100, 50)), ('cat', (10, 10, 20, 20))],
    )
    prof = BakeoffProfile(name='shop', class_names=('pallet', 'box'))
    out = tmp_path / 'out'
    assert datasets.convert('voc', src, out, prof) == 1
    rows = sorted((out / 'labels/test/f1.txt').read_text().split('\n')[:-1])
    assert [r.split()[0] for r in rows] == ['0', '1']  # 'cat' dropped
    assert (out / 'images/test/f1.png').is_symlink()


def test_voc_converter_collapses_single_class(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    img = _img(src / 'f1.png')
    _voc(img.with_suffix('.xml'), [('anything', (0, 0, 50, 50))])
    out = tmp_path / 'out'
    datasets.convert('voc', src, out, GENERIC_PROFILE)
    line = (out / 'labels/test/f1.txt').read_text().split()
    assert line[0] == '0'
    assert [float(v) for v in line[1:]] == pytest.approx([0.25, 0.5, 0.5, 1.0])
    assert '  0: object\n' in (out / 'data.yaml').read_text()


def test_yolo_passthrough_keeps_or_drops_ids(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    _img(src / 'images/f1.png')
    (src / 'labels').mkdir(parents=True)
    (src / 'labels/f1.txt').write_text('1 0.5 0.5 0.2 0.2\n7 0.1 0.1 0.1 0.1\nbad line\n')
    prof = BakeoffProfile(name='two', class_names=('a', 'b'))
    out = tmp_path / 'out'
    datasets.convert('yolo', src, out, prof)
    assert (out / 'labels/test/f1.txt').read_text() == '1 0.500000 0.500000 0.200000 0.200000\n'


def test_unknown_format_error_names_registry() -> None:
    with pytest.raises(ValueError, match='unknown dataset format'):
        datasets.get_converter('no-such-format')
