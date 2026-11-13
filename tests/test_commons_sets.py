"""Commons face-identity and larger vehicle sets: selection, licence filter, ambiguity, manifest, resume.

No network: the MediaWiki API is replaced by small recorded-shape JSON documents.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scripts.datasets import _common, commons_faces as cf, proxy_commons as pc
from scripts.datasets.proxy_manifest import read_labels, verify_dataset_dir


REPO = Path(__file__).resolve().parents[1]


def _page(
    pid: int, title: str, *, mime='image/jpeg', w=2000, h=1500, lic='CC BY-SA 4.0'
) -> dict[str, Any]:
    return {
        'pageid': pid,
        'title': f'File:{title}',
        'imageinfo': [
            {
                'mime': mime,
                'width': w,
                'height': h,
                'thumburl': f'https://thumb.example/{pid}/1600px-{title}',
                'descriptionurl': f'https://commons.example/wiki/File:{title}',
                'extmetadata': {
                    'LicenseShortName': {'value': lic},
                    'LicenseUrl': {'value': 'https://creativecommons.org/licenses/by-sa/4.0'},
                    'Artist': {'value': '<a href="x">Jane &amp; Co</a>'},
                },
            }
        ],
    }


class TestCandidate:
    def test_keeps_free_jpeg_with_licence_url_and_clean_author(self) -> None:
        c = pc.candidate(_page(1, 'a.jpg'), 600)
        assert c is not None
        assert c['license_url'].startswith('https://creativecommons.org')
        assert c['author'] == 'Jane & Co'

    @pytest.mark.parametrize(
        'page',
        [
            _page(2, 'b.png', mime='image/png'),
            _page(3, 'c.jpg', lic='CC BY-NC 2.0'),
            _page(4, 'd.jpg', w=500, h=400),
        ],
    )
    def test_rejects_wrong_mime_licence_or_size(self, page: dict[str, Any]) -> None:
        assert pc.candidate(page, 600) is None

    def test_thumb_width_is_requested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[dict[str, str]] = []

        def fake(**params: str) -> dict[str, Any]:
            seen.append(params)
            return {'query': {'pages': {}}}

        monkeypatch.setattr(pc, '_get', fake)
        pc.list_files('Category:X', thumb_width=1600)
        assert seen[0]['iiurlwidth'] == '1600'


class TestDepicts:
    def test_count_and_paged_listing_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, str]] = []

        def fake(**params: str) -> dict[str, Any]:
            calls.append(params)
            if params.get('list') == 'search':
                return {'query': {'searchinfo': {'totalhits': 321}}}
            n = len(calls)
            pages = {str(i): _page(i, f'{i}.jpg') for i in range(n * 10, n * 10 + 3)}
            return {'query': {'pages': pages}, 'continue': {'gsroffset': 50}}

        monkeypatch.setattr(pc, '_get', fake)
        assert pc.depicts_count('Q76') == 321
        assert 'haswbstatement:P180=Q76' in calls[0]['srsearch']
        found = pc.collect_search(pc.depicts_expr('Q76'), 600, thumb_width=1600, max_files=5)
        assert 5 <= len(found) <= 6
        assert all(c['url'].startswith('https://thumb') for c in found)


class TestIdentitySource:
    def test_source_selects_the_search_expression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []

        def fake(expression: str, *a: Any, **k: Any) -> list[dict[str, Any]]:
            seen.append(expression)
            return [{'page_id': 1, 'title': 'Ann.jpg'}, {'page_id': 2, 'title': 'Ann painting.jpg'}]

        monkeypatch.setattr(pc, 'collect_search', fake)
        base = {'identity': 'Ann', 'wikidata_qid': 'Q5', 'category': 'Ann A'}
        out = cf.collect_identity({**base, 'source': 'deepcategory'}, 600, 1600, 50)
        cf.collect_identity({**base, 'source': 'depicts'}, 600, 1600, 50)
        assert seen[0] == 'deepcategory:"Ann A" filetype:bitmap'
        assert seen[1] == 'haswbstatement:P180=Q5 filetype:bitmap'
        assert [c['page_id'] for c in out] == [1]


class TestAmbiguity:
    def test_file_in_two_identities_is_dropped_from_both(self) -> None:
        a = {'page_id': 1}, {'page_id': 2}
        b = {'page_id': 2}, {'page_id': 3}
        out = pc.drop_ambiguous({'A': list(a), 'B': list(b)})
        assert [c['page_id'] for c in out['A']] == [1]
        assert [c['page_id'] for c in out['B']] == [3]


class TestCache:
    def test_second_call_does_not_hit_the_network(self, tmp_path: Path) -> None:
        calls = []

        def fn() -> list[dict[str, Any]]:
            calls.append(1)
            return [{'page_id': 7}]

        cache = tmp_path / 'cands.json'
        assert pc.cached('k', cache, fn) == [{'page_id': 7}]
        assert pc.cached('k', cache, fn) == [{'page_id': 7}]
        assert len(calls) == 1


class TestDownloadRetryAfter:
    def test_429_waits_retry_after_then_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(_common.time, 'sleep', sleeps.append)
        state = {'n': 0}

        class Resp(io.BytesIO):
            def __enter__(self) -> Resp:
                return self

            def __exit__(self, *a: object) -> None:
                return None

        def fake_open(req: object, timeout: int = 0) -> Resp:
            state['n'] += 1
            if state['n'] == 1:
                raise urllib.error.HTTPError('u', 429, 'slow', {'Retry-After': '7'}, None)  # type: ignore[arg-type]
            return Resp(b'data')

        monkeypatch.setattr(_common.urllib.request, 'urlopen', fake_open)
        _common.download('https://x.test/a.jpg', tmp_path / 'a.jpg', retries=3)
        assert (tmp_path / 'a.jpg').read_bytes() == b'data'
        assert 7.0 in sleeps


SPARQL = {
    'results': {
        'bindings': [
            {'p': {'value': 'http://www.wikidata.org/entity/Q20'}},
            {'p': {'value': 'http://www.wikidata.org/entity/Q3'}},
            {'p': {'value': 'http://www.wikidata.org/entity/Q3'}},
        ]
    }
}


def _entity(qid: str, **claims: Any) -> dict[str, Any]:
    def snak(value: Any) -> dict[str, Any]:
        return {'mainsnak': {'datavalue': {'value': value}}}

    return {
        'id': qid,
        'labels': {'en': {'value': 'Ann'}},
        'claims': {k: [snak(v)] for k, v in claims.items()},
    }


class TestIdentities:
    def test_parse_qids_dedupes_and_sorts_numerically(self) -> None:
        assert cf.parse_qids(SPARQL) == ['Q3', 'Q20']

    def test_person_row_reads_claims(self) -> None:
        ent = _entity(
            'Q1', P373='Ann A', P21={'id': 'Q6581072'}, P27={'id': 'Q142'},
            P569={'time': '+1970-01-01T00:00:00Z'},
        )  # fmt: skip
        assert cf.person_row(ent, {'Q142': 'France'}) == {
            'identity': 'Ann', 'wikidata_qid': 'Q1', 'category': 'Ann A',
            'gender': 'female', 'country': 'France', 'born': 1970,
        }  # fmt: skip

    def test_person_row_without_category_is_dropped_and_gaps_are_unknown(self) -> None:
        assert cf.person_row(_entity('Q2'), {}) is None
        row = cf.person_row(_entity('Q2', P373='Ann'), {})
        assert row is not None
        assert row['gender'] == 'unknown'
        assert row['born'] is None

    def _pool(self) -> list[dict[str, Any]]:
        return [
            {
                'identity': f'P{i}', 'wikidata_qid': f'Q{i}', 'category': f'P{i}',
                'gender': 'female' if i % 3 == 0 else 'male',
                'country': f'C{i % 12}', 'born': 1940 + i % 50, 'hits': 50 + i,
            }
            for i in range(120)
        ]  # fmt: skip

    def test_pick_is_seeded_balanced_and_respects_min_hits(self) -> None:
        pool = self._pool()
        a = cf.pick_identities(pool, 30, min_hits=80, female_share=0.4, seed=1)
        assert a == cf.pick_identities(pool, 30, min_hits=80, female_share=0.4, seed=1)
        assert len(a) == 30
        assert all(r['hits'] >= 80 for r in a)
        assert sum(r['gender'] == 'female' for r in a) == 12
        assert len({r['country'] for r in a}) >= 10

    def test_committed_lists_are_valid(self) -> None:
        v = pc.load_categories(REPO / 'scripts/datasets/manifests/commons_vehicles_categories.json')
        c = pc.load_categories(REPO / 'scripts/datasets/manifests/commons_set_c_categories.json')
        assert len(v) >= 200
        assert {r['category'] for r in c} <= {r['category'] for r in v}
        faces = REPO / 'scripts/datasets/manifests/commons_faces_identities.json'
        if faces.is_file():
            assert len(cf.load_identities(faces)) >= 100

    def test_load_identities_validates(self, tmp_path: Path) -> None:
        p = tmp_path / 'i.json'
        p.write_text(json.dumps([{'identity': 'A'}]))
        with pytest.raises(_common.FetchError, match='lacks'):
            cf.load_identities(p)


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (8, 8), 'red').save(buf, 'JPEG')
    return buf.getvalue()


class TestFaceDataset:
    def _cand(self, pid: int) -> dict[str, Any]:
        return {
            'page_id': pid, 'title': f'{pid}.jpg', 'url': f'https://x.test/{pid}.jpg',
            'description_url': f'https://c.test/File:{pid}.jpg', 'license': 'CC BY 4.0',
            'license_url': 'https://creativecommons.org/licenses/by/4.0', 'author': 'Z',
        }  # fmt: skip

    def test_build_writes_manifest_labels_review_and_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        idents = [
            {'identity': 'Ann A', 'wikidata_qid': 'Q1', 'category': 'Ann A', 'gender': 'female', 'country': 'F', 'born': 1970},
            {'identity': 'Bob B', 'wikidata_qid': 'Q2', 'category': 'Bob B', 'gender': 'male', 'country': 'G', 'born': 1960},
            {'identity': 'Few', 'wikidata_qid': 'Q3', 'category': 'Few', 'gender': 'male', 'country': 'H', 'born': 1960},
        ]  # fmt: skip
        per = {
            'Q1': [self._cand(i) for i in (1, 2, 3, 99)],
            'Q2': [self._cand(i) for i in (4, 5, 6, 99)],  # 99 is in both: ambiguous
            'Q3': [self._cand(7)],
        }
        fetched: list[str] = []

        def fake_download(url: str, dest: Path, **kw: Any) -> Path:
            if dest.is_file():
                return dest
            fetched.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_jpeg())
            return dest

        monkeypatch.setattr(cf, 'download', fake_download)
        out = tmp_path / 'faces'
        manifest = cf.build_dataset(
            idents, per, out, n=3, min_per_identity=3, seed=3, review_fraction=0.5
        )
        assert manifest['count'] == 6
        assert manifest['dropped_ambiguous'] == 2
        assert manifest['dropped_identities'] == ['Few (1)']
        assert len(fetched) == 6
        assert verify_dataset_dir(out).ok
        rows = read_labels(out / 'labels.jsonl')
        row = next(r for r in rows if r['identity'] == 'Ann A')
        for key in ('wikidata_qid', 'file_page', 'author', 'licence', 'licence_url', 'sha256'):
            assert row[key]
        assert (out / 'review_sample.csv').read_text().startswith('path,label_correct')
        fetched.clear()
        again = cf.build_dataset(
            idents, per, out, n=3, min_per_identity=3, seed=3, review_fraction=0.5
        )
        assert fetched == []
        assert again['manifest_sha256'] == manifest['manifest_sha256']


class TestOneFaceFilter:
    def test_keeps_exactly_one_face_and_reports(self, tmp_path: Path) -> None:
        from scripts.datasets import filter_one_face as f1

        labels = [{'path': f'images/a/{i}.jpg', 'identity': 'A'} for i in range(4)]
        counts = {
            'images/a/0.jpg': 1,
            'images/a/1.jpg': 0,
            'images/a/2.jpg': 2,
            'images/a/3.jpg': 1,
        }
        kept, report = f1.filter_rows(labels, lambda rel: counts[rel])
        assert [r['path'] for r in kept] == ['images/a/0.jpg', 'images/a/3.jpg']
        assert report == {'checked': 4, 'kept': 2, 'no_face': 1, 'multi_face': 1, 'error': 0}

    def test_detector_error_is_counted_not_kept(self) -> None:
        from scripts.datasets import filter_one_face as f1

        def boom(rel: str) -> int:
            raise f1.DetectError('503')

        kept, report = f1.filter_rows([{'path': 'p'}], boom)
        assert kept == []
        assert report['error'] == 1


class TestIndexPartialVehicles:
    def _setup(
        self, tmp_path: Path
    ) -> tuple[list[dict[str, str]], dict[str, list[dict[str, Any]]]]:
        from scripts.datasets import fetch_vehicle_proxy as fv

        rows = [
            {'make': 'A', 'model': 'One', 'generation': 'I', 'category': 'A One I'},
            {'make': 'B', 'model': 'Two', 'generation': 'II', 'category': 'B Two II'},
            {'make': 'C', 'model': 'Three', 'generation': 'III', 'category': 'C Three III'},
        ]

        def cand(pid: int) -> dict[str, Any]:
            return {
                'page_id': pid, 'title': f'{pid}.jpg', 'url': f'https://x.test/{pid}', 'author': 'Z',
                'description_url': f'https://c.test/{pid}', 'license': 'CC BY 4.0',
                'license_url': 'https://creativecommons.org/licenses/by/4.0',
            }  # fmt: skip

        per = {
            'A/One/I': [cand(1), cand(2), cand(3), cand(99)],
            'B/Two/II': [cand(4), cand(99)],
            'C/Three/III': [cand(5)],
        }
        for slug, ids in {'A_One_I': (1, 2, 3), 'B_Two_II': (4,)}.items():
            d = tmp_path / 'images' / slug
            d.mkdir(parents=True)
            for i in ids:
                (d / f'{i}.jpg').write_bytes(_jpeg())
        assert fv.commons.slug(rows[0]) == 'A_One_I'
        return rows, per

    def test_lists_disk_files_drops_small_classes_and_removes_their_files(
        self, tmp_path: Path
    ) -> None:
        from scripts.datasets import fetch_vehicle_proxy as fv

        rows, per = self._setup(tmp_path)
        files, dropped = fv.index_partial(rows, per, tmp_path, min_per_model=3)
        assert [f['page_id'] for f in files] == [1, 2, 3]
        assert files[0]['license_url'].startswith('https://creativecommons.org')
        assert files[0]['make'] == 'A'
        assert dropped == ['B/Two/II (1)']
        assert not (tmp_path / 'images' / 'B_Two_II').exists()

    def test_unknown_file_on_disk_fails_closed(self, tmp_path: Path) -> None:
        from scripts.datasets import fetch_vehicle_proxy as fv

        rows, per = self._setup(tmp_path)
        (tmp_path / 'images' / 'A_One_I' / '777.jpg').write_bytes(_jpeg())
        with pytest.raises(_common.FetchError, match='777'):
            fv.index_partial(rows, per, tmp_path, min_per_model=3)
