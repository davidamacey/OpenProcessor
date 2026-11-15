"""Tests for the pure parts of scripts/datasets/fetch_hires_set.py (no network)."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from email.message import Message
from typing import TYPE_CHECKING, Any

import pytest

from scripts.datasets import _common, fetch_hires_set as f
from scripts.datasets.bench_set import build_pin, load_pin


if TYPE_CHECKING:
    from pathlib import Path


def page(
    page_id: int,
    width: int,
    height: int,
    *,
    mime: str = 'image/jpeg',
    license_name: str = 'CC BY-SA 4.0',
    artist: str = '<a href="x">Jane &amp; Co</a>',
) -> dict[str, Any]:
    return {
        'pageid': page_id,
        'title': f'File:Photo {page_id}.jpg',
        'imageinfo': [
            {
                'url': f'https://upload.example/p{page_id}.jpg?utm_source=x',
                'descriptionurl': f'https://commons.example/wiki/{page_id}',
                'width': width,
                'height': height,
                'mime': mime,
                'extmetadata': {
                    'LicenseShortName': {'value': license_name},
                    'Artist': {'value': artist},
                },
            }
        ],
    }


def test_candidate_rows_filters_size_mime_and_license_and_sorts() -> None:
    pages = [
        page(30, 4000, 3000),  # 12 MP, kept
        page(10, 5472, 3648),  # 20 MP, kept
        page(20, 640, 480),  # too small
        page(21, 8000, 6000),  # 48 MP, too big
        page(22, 4000, 3000, mime='image/png'),
        page(23, 4000, 3000, license_name='All rights reserved'),
        page(24, 4000, 3000, license_name='Public domain'),
        page(25, 4000, 3000, license_name='CC0'),
        {'pageid': 99, 'title': 'File:no info.jpg'},
    ]
    rows = f.candidate_rows(pages, 12_000_000, 24_000_000)
    assert [r['page_id'] for r in rows] == [10, 24, 25, 30]
    first = rows[0]
    assert first['url'] == 'https://upload.example/p10.jpg'
    assert first['author'] == 'Jane & Co'
    assert first['title'] == 'Photo 10.jpg'


def test_license_pattern_does_not_accept_a_nonfree_variant() -> None:
    assert f.ALLOWED_LICENSE.match('CC BY 2.0')
    assert f.ALLOWED_LICENSE.match('CC BY-SA 3.0')
    assert not f.ALLOWED_LICENSE.match('CC BY-NC 2.0')
    assert not f.ALLOWED_LICENSE.match('CC BY-ND 2.0')


def test_pin_rows_load_as_a_bench_pin() -> None:
    cand = f.candidate_rows([page(7, 4000, 3000)], 1, 10**9)[0]
    row = f.pin_row(cand, 5_000_000, 'ab' * 32)
    pin = build_pin([row], seed=1, licenses=['CC BY-SA 4.0'], annotations_sha256='n/a')
    # load_pin accepts the same shape as the COCO pin
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'pin.json'
        path.write_text(json.dumps(pin))
        loaded = load_pin(path)
    assert loaded['images'][0]['file_name'] == '7.jpg'
    assert loaded['images'][0]['width'] * loaded['images'][0]['height'] == 12_000_000


def test_candidate_rows_empty_window() -> None:
    assert f.candidate_rows([page(1, 4000, 3000)], 20_000_000, 24_000_000) == []


# --- the pinned fetch: missing upstream files, rate limits, an existing local copy -------------


def body(page_id: int) -> bytes:
    return f'photo {page_id}'.encode()


def url(page_id: int) -> str:
    return f'https://upload.example/p{page_id}.jpg'


def make_pin(ids: list[int]) -> dict[str, Any]:
    rows = [
        {
            'image_id': i,
            'file_name': f'{i}.jpg',
            'split': 'wikimedia_commons',
            'license_name': 'CC0',
            'flickr_url': url(i),
            'width': 4000,
            'height': 3000,
            'bytes': len(body(i)),
            'sha256': hashlib.sha256(body(i)).hexdigest(),
        }
        for i in ids
    ]
    return build_pin(rows, seed=1, licenses=['CC0'], annotations_sha256='n/a')


class Server:
    """``urlopen`` stand-in: ``script[url]`` is a list of answers (an int is an HTTP status), the
    last one repeating."""

    def __init__(self, script: dict[str, list[int | bytes]]) -> None:
        self.script = script
        self.asked: list[str] = []

    def __call__(self, req: Any, timeout: float = 0) -> io.BytesIO:
        target = req.full_url
        self.asked.append(target)
        answers = self.script[target]
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, int):
            headers = Message()
            if answer == 429:
                headers['Retry-After'] = '7'
            raise urllib.error.HTTPError(target, answer, 'x', headers, None)
        return io.BytesIO(answer)


@pytest.fixture
def bed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    sleeps: list[float] = []
    monkeypatch.setattr(_common.time, 'sleep', sleeps.append)
    monkeypatch.setattr(f.time, 'sleep', sleeps.append)
    return {'tmp': tmp_path, 'sleeps': sleeps, 'monkeypatch': monkeypatch}


def run_fetch(bed: dict[str, Any], ids: list[int], script: dict[str, list[int | bytes]],
              *extra: str) -> tuple[int, Server]:  # fmt: skip
    tmp: Path = bed['tmp']
    pin = make_pin(ids)
    (tmp / 'pin.json').write_text(json.dumps(pin))
    server = Server(script)
    bed['monkeypatch'].setattr(_common.urllib.request, 'urlopen', server)
    code = f.main(['--out', str(tmp / 'set'), '--manifest', str(tmp / 'pin.json'), *extra])
    return code, server


def test_a_missing_file_in_the_middle_does_not_stop_the_rest(
    bed: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    script: dict[str, list[int | bytes]] = {
        url(1): [body(1)], url(2): [404], url(3): [body(3)], url(4): [body(4)]
    }  # fmt: skip
    code, server = run_fetch(bed, [1, 2, 3, 4], script, '--max-missing-share', '0.5')
    images = bed['tmp'] / 'set' / 'images'
    assert sorted(p.name for p in images.iterdir()) == ['1.jpg', '3.jpg', '4.jpg']
    assert server.asked.count(url(2)) == 1, 'a 404 is not retried'
    assert code == 0


def test_the_summary_names_the_missing_file_and_the_counts(
    bed: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level('INFO')
    run_fetch(bed, [1, 2, 3], {url(1): [body(1)], url(2): [404], url(3): [body(3)]},
              '--max-missing-share', '0.5')  # fmt: skip
    text = caplog.text
    assert '2.jpg' in text
    assert 'present 2' in text
    assert 'missing upstream 1' in text
    assert '2.jpg' in (bed['tmp'] / 'set' / 'missing_upstream.txt').read_text()


def test_strict_fails_on_any_missing_file(bed: dict[str, Any]) -> None:
    script: dict[str, list[int | bytes]] = {url(1): [body(1)], url(2): [404]}
    code, _ = run_fetch(bed, [1, 2], script, '--strict')
    assert code == 1
    assert (bed['tmp'] / 'set' / 'images' / '1.jpg').is_file()


def test_more_than_the_share_missing_fails_and_fewer_passes(bed: dict[str, Any]) -> None:
    script: dict[str, list[int | bytes]] = {url(1): [body(1)], url(2): [404], url(3): [404]}
    assert run_fetch(bed, [1, 2, 3], script, '--max-missing-share', '0.5')[0] == 1
    assert run_fetch(bed, [1, 2, 3], script, '--max-missing-share', '0.7')[0] == 0


def test_a_429_is_retried_after_the_servers_retry_after(bed: dict[str, Any]) -> None:
    code, server = run_fetch(bed, [1], {url(1): [429, 429, body(1)]})
    assert code == 0
    assert server.asked.count(url(1)) == 3
    assert bed['sleeps'] == [7.0, 7.0]


def test_a_retry_after_date_is_honoured(bed: dict[str, Any]) -> None:
    exc = urllib.error.HTTPError('u', 429, 'x', Message(), None)
    exc.headers['Retry-After'] = 'Wed, 21 Oct 2099 07:28:00 GMT'
    assert _common._retry_delay(exc, 1) == 300.0, 'capped, not the 15 s fallback'


def test_an_existing_local_copy_is_verified_not_downloaded(bed: dict[str, Any]) -> None:
    images = bed['tmp'] / 'set' / 'images'
    images.mkdir(parents=True)
    for i in (1, 2):
        (images / f'{i}.jpg').write_bytes(body(i))
    code, server = run_fetch(bed, [1, 2], {})
    assert code == 0
    assert server.asked == []


def test_verify_only_counts_present_missing_upstream_and_corrupt_apart(
    bed: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level('INFO')
    out = bed['tmp'] / 'set'
    (out / 'images').mkdir(parents=True)
    (out / 'images' / '1.jpg').write_bytes(body(1))
    (out / 'images' / '2.jpg').write_bytes(b'xxxxxxxx')  # corrupt
    (out / 'missing_upstream.txt').write_text('3.jpg\n')
    code, server = run_fetch(bed, [1, 2, 3, 4], {}, '--verify-only')
    assert server.asked == [], 'verify-only never touches the network'
    assert code == 1, 'a corrupt file always fails'
    assert 'present 1, missing upstream 1, not downloaded 1, corrupt 1' in caplog.text
