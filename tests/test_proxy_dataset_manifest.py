"""Manifest, verification, selection and label derivation of the proxy dataset fetchers.

No network: every input is a small hand-built tree or CSV string.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from scripts.datasets import fetch_face_proxy as ff, proxy_commons as pc, proxy_openimages as oi
from scripts.datasets._common import FetchError
from scripts.datasets.proxy_manifest import (
    Licence,
    build_manifest,
    file_entry,
    label_row,
    load_manifest,
    pin_of,
    read_labels,
    safe_relpath,
    verify_dataset_dir,
    verify_manifest,
    write_dataset,
)
from scripts.datasets.vehicle_labels import (
    CompCarsRef,
    compcars_names,
    parse_compcars_image,
    parse_compcars_view,
    parse_stanford_class,
)


REPO = Path(__file__).resolve().parents[1]
LIC = Licence('test', 'https://example.test/licence', False)


def _dataset(tmp_path: Path) -> tuple[Path, dict]:
    (tmp_path / 'images').mkdir()
    for name, body in (('a.jpg', b'aaa'), ('b.jpg', b'bbbb')):
        (tmp_path / 'images' / name).write_bytes(body)
    files = [file_entry(tmp_path, f'images/{n}') for n in ('a.jpg', 'b.jpg')]
    return tmp_path, build_manifest('t', 'https://example.test', LIC, files)


class TestManifest:
    def test_verify_ok_then_each_failure_kind(self, tmp_path: Path) -> None:
        root, manifest = _dataset(tmp_path)
        assert verify_manifest(manifest, root).ok
        (root / 'images/a.jpg').write_bytes(b'aab')  # same size, different bytes
        (root / 'images/b.jpg').write_bytes(b'b')  # different size
        (root / 'images/c.jpg').write_bytes(b'extra')
        report = verify_manifest(manifest, root)
        assert report.hash_mismatch == ['images/a.jpg']
        assert report.size_mismatch == ['images/b.jpg']
        assert report.unlisted == ['images/c.jpg']
        (root / 'images/a.jpg').unlink()
        assert verify_manifest(manifest, root).missing == ['images/a.jpg']

    def test_partial_downloads_and_metadata_are_not_unlisted(self, tmp_path: Path) -> None:
        root, manifest = _dataset(tmp_path)
        (root / 'images/x.jpg.part').write_bytes(b'x')
        (root / 'labels.jsonl').write_text('')
        assert verify_manifest(manifest, root).ok

    def test_load_rejects_tampered_rows(self, tmp_path: Path) -> None:
        root, manifest = _dataset(tmp_path)
        path = root / 'm.json'
        path.write_text(json.dumps(manifest))
        assert load_manifest(path)['count'] == 2
        manifest['files'][0]['sha256'] = '0' * 64
        path.write_text(json.dumps(manifest))
        with pytest.raises(FetchError, match='manifest_sha256'):
            load_manifest(path)

    def test_digest_covers_extra_per_file_fields(self, tmp_path: Path) -> None:
        root, _ = _dataset(tmp_path)
        a = build_manifest('t', 'u', LIC, [file_entry(root, 'images/a.jpg', author='x')])
        b = build_manifest('t', 'u', LIC, [file_entry(root, 'images/a.jpg', author='y')])
        assert a['manifest_sha256'] != b['manifest_sha256']

    @pytest.mark.parametrize('rel', ['', '/etc/passwd', '../x', 'a/../b', 'a\\b', './a'])
    def test_unsafe_paths_are_refused(self, rel: str) -> None:
        with pytest.raises(FetchError):
            safe_relpath(rel)

    def test_duplicate_paths_are_refused(self, tmp_path: Path) -> None:
        root, _ = _dataset(tmp_path)
        e = file_entry(root, 'images/a.jpg')
        with pytest.raises(FetchError, match='duplicate'):
            build_manifest('t', 'u', LIC, [e, e])

    def test_dataset_dir_roundtrip_and_orphan_label(self, tmp_path: Path) -> None:
        root, manifest = _dataset(tmp_path)
        labels = [label_row('images/a.jpg', 'face', ['face'], identity='p1')]
        pin = tmp_path.parent / f'{tmp_path.name}_pin.json'
        write_dataset(root, manifest, labels, pin)
        assert verify_dataset_dir(root).ok
        assert read_labels(root / 'labels.jsonl')[0]['identity'] == 'p1'
        assert json.loads(pin.read_text()) == pin_of(manifest)
        write_dataset(root, manifest, [label_row('images/zz.jpg', 'face', ['face'])], None)
        assert not verify_dataset_dir(root).ok

    def test_label_taxonomy_must_end_with_class(self) -> None:
        with pytest.raises(FetchError, match='taxonomy_path'):
            label_row('images/a.jpg', 'Toyota Corolla', ['Toyota'])


class TestStanfordLabels:
    @pytest.mark.parametrize(
        ('name', 'make', 'model', 'body', 'year'),
        [
            ('BMW M3 Coupe 2012', 'BMW', 'M3', 'Coupe', 2012),
            ('AM General Hummer SUV 2000', 'AM General', 'Hummer', 'SUV', 2000),
            (
                'Aston Martin V8 Vantage Convertible 2012',
                'Aston Martin',
                'V8 Vantage',
                'Convertible',
                2012,
            ),
            ('Chevrolet Express Cargo Van 2007', 'Chevrolet', 'Express', 'Cargo Van', 2007),
            ('Chevrolet Express Van 2007', 'Chevrolet', 'Express', 'Van', 2007),
            (
                'Chevrolet Silverado 1500 Extended Cab 2012',
                'Chevrolet',
                'Silverado 1500',
                'Extended Cab',
                2012,
            ),
            ('Ram C/V Cargo Van Minivan 2012', 'Ram', 'C/V Cargo Van', 'Minivan', 2012),
            (
                'Mercedes-Benz 300-Class Convertible 1993',
                'Mercedes-Benz',
                '300-Class',
                'Convertible',
                1993,
            ),
            ('Acura Integra Type R 2001', 'Acura', 'Integra Type R', None, 2001),
        ],
    )
    def test_parse(self, name: str, make: str, model: str, body: str | None, year: int) -> None:
        got = parse_stanford_class(name)
        assert (got.make, got.model, got.body, got.year) == (make, model, body, year)
        assert got.taxonomy_path == [make, f'{make} {model}']

    @pytest.mark.parametrize('bad', ['BMW M3 Coupe', 'BMW 2012', ''])
    def test_unparseable_is_refused(self, bad: str) -> None:
        with pytest.raises(FetchError):
            parse_stanford_class(bad)


class TestCompCarsLabels:
    def test_ids_to_paths_and_names(self) -> None:
        ref = parse_compcars_image('image/2/3/2012/abc.jpg')
        assert ref == CompCarsRef(2, 3, '2012', 'abc.jpg')
        assert ref.label_path == 'label/2/3/2012/abc.txt'
        assert compcars_names(ref, ['A', 'B'], ['m1', 'm2', 'm3']) == ('B', 'm3')

    @pytest.mark.parametrize(
        'rel', ['image/2/3/abc.jpg', 'img/2/3/2012/a.jpg', 'image/x/3/2012/a.jpg']
    )
    def test_malformed_paths_are_refused(self, rel: str) -> None:
        with pytest.raises(FetchError):
            parse_compcars_image(rel)

    def test_name_ids_out_of_range_fail_closed(self) -> None:
        ref = CompCarsRef(3, 1, '2012', 'a.jpg')
        with pytest.raises(FetchError, match='outside'):
            compcars_names(ref, ['A', 'B'], ['m'])

    @pytest.mark.parametrize(('text', 'view'), [('1\n1\n0 0 1 1', 'front'), ('5\n', 'rear-side')])
    def test_view_codes(self, text: str, view: str) -> None:
        assert parse_compcars_view(text) == view

    @pytest.mark.parametrize('text', ['', '0\n', '6\n', 'x\n'])
    def test_unknown_view_is_refused(self, text: str) -> None:
        with pytest.raises(FetchError):
            parse_compcars_view(text)


BBOX = (
    'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n'
    'i1,x,/m/car,1,0.0,0.5,0.0,0.5,0,0,0,0,0\n'  # area 0.25, Car
    'i1,x,/m/bus,1,0.0,0.3,0.0,0.3,0,0,0,0,0\n'  # 0.09: smaller, same image
    'i2,x,/m/car,1,0.0,0.5,0.0,0.5,0,0,1,0,0\n'  # group box: dropped
    'i3,x,/m/car,1,0.0,0.5,0.0,0.5,0,0,0,1,0\n'  # depiction: dropped
    'i4,x,/m/bus,1,0.0,0.01,0.0,0.01,0,0,0,0,0\n'  # too small
    'i5,x,/m/bus,1,0.0,0.6,0.0,0.6,0,0,0,0,0\n'
    'i6,x,/m/other,1,0.0,0.9,0.0,0.9,0,0,0,0,0\n'  # unwanted class
)
BY = oi.LICENCE_URL
IMAGES = (
    'ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author\n'
    f'i1,v,u1,l,{BY},p,Ann\n'
    f'i5,v,u5,l,{BY},p,Bob\n'
    'i6,v,u6,l,https://creativecommons.org/licenses/by-nc/2.0/,p,Cy\n'
)


class TestOpenImagesSelection:
    def test_boxes_filter_and_dominant_class(self) -> None:
        wanted = {'/m/car': 'Car', '/m/bus': 'Bus'}
        boxes = oi.qualifying_boxes(BBOX, wanted, 0.05)
        assert set(boxes) == {'i1', 'i5'}
        assert [b['class_name'] for b in boxes['i1']] == ['Car', 'Bus']
        chosen = oi.select_images(boxes, oi.eligible_image_ids(IMAGES), per_class=5, seed=1)
        assert [(r['image_id'], r['class_name']) for r in chosen] == [('i1', 'Car'), ('i5', 'Bus')]
        assert oi.summarize(chosen)['per_class'] == {'Car': 1, 'Bus': 1}

    def test_non_ccby2_licence_is_not_eligible(self) -> None:
        assert set(oi.eligible_image_ids(IMAGES)) == {'i1', 'i5'}

    def test_selection_is_seeded_and_capped(self) -> None:
        boxes = {
            f'i{n:02d}': [{'class_name': 'Car', 'box': [0, 0, 1, 1], 'area': 1.0}]
            for n in range(30)
        }
        elig = {k: {'License': BY, 'Author': '', 'OriginalURL': ''} for k in boxes}
        a = oi.select_images(boxes, elig, 7, 3)
        assert a == oi.select_images(boxes, elig, 7, 3)
        assert len(a) == 7
        assert a != oi.select_images(boxes, elig, 7, 4)

    def test_missing_class_name_is_an_error(self) -> None:
        with pytest.raises(FetchError, match='not found'):
            oi.class_ids('/m/car,Car\n', ('Car', 'Hovercraft'))


class TestCommons:
    def test_ambiguous_files_are_dropped_everywhere(self) -> None:
        per = {'a': [{'page_id': 1}, {'page_id': 2}], 'b': [{'page_id': 2}, {'page_id': 3}]}
        assert pc.drop_ambiguous(per) == {'a': [{'page_id': 1}], 'b': [{'page_id': 3}]}

    def test_small_models_are_dropped_and_named(self) -> None:
        per = {'big': [{'page_id': i} for i in range(10)], 'small': [{'page_id': 99}]}
        chosen, dropped = pc.choose(per, per_model_n=4, min_per_model=3, seed=1)
        assert list(chosen) == ['big']
        assert len(chosen['big']) == 4
        assert dropped == ['small (1)']

    @pytest.mark.parametrize(
        ('title', 'kept'),
        [
            ('Category:Toyota Corolla (E210) interior', False),
            ('Category:Toyota Corolla (E210) engines', False),
            ('Category:Toyota Corolla (E210) in Germany', True),
            ('Category:Toyota Corolla (E210) hybrids', True),
        ],
    )
    def test_subcategory_filter_keeps_whole_words_only(self, title: str, kept: bool) -> None:
        assert pc.keep_subcategory(title) is kept

    def test_candidate_filters_licence_mime_and_size(self) -> None:
        def page(lic: str, mime: str = 'image/jpeg', w: int = 2000) -> dict:
            return {
                'pageid': 7,
                'title': 'File:A.jpg',
                'imageinfo': [
                    {
                        'mime': mime,
                        'width': w,
                        'height': 900,
                        'thumburl': 'https://t/x.jpg',
                        'extmetadata': {
                            'LicenseShortName': {'value': lic},
                            'Artist': {'value': '<a href="x">Ann &amp; Bo</a>'},
                        },
                    }
                ],
            }

        ok = pc.candidate(page('CC BY-SA 4.0'), 1000)
        assert ok is not None
        assert ok['author'] == 'Ann & Bo'
        assert pc.candidate(page('CC BY-NC 4.0'), 1000) is None
        assert pc.candidate(page('CC BY 4.0', mime='image/png'), 1000) is None
        assert pc.candidate(page('CC BY 4.0', w=500), 1000) is None

    def test_review_sample_is_ten_percent_and_seeded(self) -> None:
        paths = [f'images/{i}.jpg' for i in range(200)]
        s = pc.review_sample(paths, 0.1, 5)
        assert len(s) == 20
        assert s == pc.review_sample(paths, 0.1, 5)

    def test_category_file_validation(self, tmp_path: Path) -> None:
        p = tmp_path / 'c.json'
        p.write_text(json.dumps([{'make': 'A', 'model': 'B', 'generation': 'C'}]))
        with pytest.raises(FetchError, match='lacks'):
            pc.load_categories(p)

    def test_committed_category_list_is_valid(self) -> None:
        rows = pc.load_categories(REPO / 'scripts/datasets/manifests/commons_set_c_categories.json')
        assert len(rows) >= 60


class TestLfwExtraction:
    def _tar(self, path: Path, members: dict[str, bytes]) -> None:
        import io

        with tarfile.open(path, 'w:gz') as tar:
            for name, body in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(body)
                tar.addfile(info, io.BytesIO(body))

    def test_extracts_expected_layout(self, tmp_path: Path) -> None:
        self._tar(tmp_path / 'a.tgz', {'lfw/Ann_Lee/Ann_Lee_0001.jpg': b'x'})
        assert ff.extract_lfw(tmp_path / 'a.tgz', tmp_path / 'images') == 1
        assert (tmp_path / 'images/Ann_Lee/Ann_Lee_0001.jpg').read_bytes() == b'x'

    @pytest.mark.parametrize(
        'name', ['../evil.jpg', 'lfw/../../evil.jpg', 'lfw/Ann/Bob_0001.jpg', 'x/y/z.jpg']
    )
    def test_unexpected_members_are_refused(self, tmp_path: Path, name: str) -> None:
        self._tar(tmp_path / 'a.tgz', {name: b'x'})
        with pytest.raises(FetchError, match='unexpected member'):
            ff.extract_lfw(tmp_path / 'a.tgz', tmp_path / 'images')
        assert not (tmp_path / 'evil.jpg').exists()
