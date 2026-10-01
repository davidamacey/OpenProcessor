"""``import_labeled_dataset.py`` is a thin client of ``/datasets/imports``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from scripts.curation import import_labeled_dataset as cli


if TYPE_CHECKING:
    from pathlib import Path

PREVIEW = {
    'format': 'yolo',
    'totals': {'images': 2, 'boxes': 3},
    'classes': [
        {'dataset_class': 'Car', 'boxes': 2, 'suggestion': {'action': 'map', 'match': 'exact'}},
        {
            'dataset_class': 'person',
            'boxes': 1,
            'suggestion': {'action': 'create', 'match': 'none'},
        },
    ],
    'issues': [],
    'blocking': False,
}
JOB = {
    'import_id': 'imp_20260101T000000_aaaaaaaa',
    'status': 'completed',
    'reused': False,
    'progress': {'images_done': 2, 'images_total': 2},
    'report': {'items_created': 3},
}


class _Api:
    def __init__(self, *, start: httpx.Response | None = None) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.start = start

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        path = request.url.path
        if path.endswith('/datasets/preview'):
            return httpx.Response(200, json=PREVIEW)
        if path.endswith('/datasets/imports') and request.method == 'POST':
            return self.start or httpx.Response(202, json={**JOB, 'status': 'running'})
        if path.endswith('/resume'):
            return httpx.Response(202, json={**JOB, 'status': 'running'})
        if path.endswith('/entries'):
            return httpx.Response(
                200,
                json={
                    'total': 2,
                    'items': [
                        {
                            'rel_path': 'a.jpg',
                            'status': 'ok',
                            'split': 'train',
                            'image_id': 'i1',
                            'image_path': '/data/a.jpg',
                            'image_created': True,
                            'label_state': 'labeled',
                        },
                        {
                            'rel_path': 'b.jpg',
                            'status': 'ok',
                            'split': 'val',
                            'image_id': 'i2',
                            'image_path': '/data/b.jpg',
                            'image_created': False,
                            'label_state': 'negative',
                        },
                    ],
                },
            )
        return httpx.Response(200, json=JOB)


def _run(argv: list[str], api: _Api) -> int:
    args = cli.build_parser().parse_args(['--poll-interval', '0', *argv])
    with httpx.Client(transport=httpx.MockTransport(api)) as client:
        return cli.run(args, client)


def _start_body(api: _Api) -> dict:
    return next(b for m, p, b in api.calls if m == 'POST' and p.endswith('/datasets/imports'))


def test_images_only_skips_every_class_the_preview_found() -> None:
    api = _Api()
    assert _run(['--dataset', '/data/ds', '--images-only'], api) == 0
    body = _start_body(api)
    assert body['mapping'] == [
        {'dataset_class': 'Car', 'action': 'skip'},
        {'dataset_class': 'person', 'action': 'skip'},
    ]
    assert body['source']['path'] == '/data/ds'


def test_mapping_flags_become_entries_and_the_path_map_rewrites_the_dataset() -> None:
    api = _Api()
    argv = [
        '--dataset',
        '/local/ds',
        '--path-map',
        '/local=/srv',
        '--map',
        'Car=2',
        '--create',
        'person=human',
        '--accept-suggestions',
    ]
    assert _run(argv, api) == 0
    body = _start_body(api)
    assert body['source']['path'] == '/srv/ds'
    assert body['mapping'] == [
        {'dataset_class': 'Car', 'action': 'map', 'class_id': 2},
        {'dataset_class': 'person', 'action': 'create', 'new_class_name': 'human'},
    ]
    assert body['accept_suggestions'] is True


def test_dry_run_only_previews() -> None:
    api = _Api()
    assert _run(['--dataset', '/data/ds', '--dry-run'], api) == 0
    assert [p.rsplit('/', 1)[-1] for _m, p, _b in api.calls] == ['preview']


def test_a_resumable_import_is_resumed() -> None:
    api = _Api(
        start=httpx.Response(
            409, json={'detail': {'error': 'import_resumable', 'import_id': JOB['import_id']}}
        )
    )
    assert _run(['--dataset', '/data/ds', '--images-only'], api) == 0
    assert any(p.endswith(f'{JOB["import_id"]}/resume') for _m, p, _b in api.calls)


def test_a_class_given_twice_is_refused() -> None:
    with pytest.raises(cli.ImportCliError, match='more than once'):
        cli.build_mapping(
            cli.build_parser().parse_args(['--dataset', 'x', '--map', 'a=1', '--skip', 'a']), ['a']
        )


def test_server_errors_become_a_nonzero_exit(capsys: pytest.CaptureFixture[str]) -> None:
    api = _Api(start=httpx.Response(422, json={'detail': {'error': 'import_blocked'}}))
    with pytest.raises(cli.ImportCliError, match='import_blocked'):
        _run(['--dataset', '/data/ds', '--images-only'], api)


def test_state_dir_gets_the_per_split_ingested_cohort(tmp_path: Path) -> None:
    api = _Api()
    assert _run(['--dataset', '/data/ds', '--images-only', '--state-dir', str(tmp_path)], api) == 0
    train = [
        json.loads(line) for line in (tmp_path / 'ingested/train.jsonl').read_text().splitlines()
    ]
    assert train == [
        {
            'image': 'a.jpg',
            'server_path': '/data/a.jpg',
            'image_id': 'i1',
            'status': 'success',
            'positive': True,
        }
    ]
    val = json.loads((tmp_path / 'ingested/val.jsonl').read_text())
    assert (val['status'], val['positive']) == ('duplicate', False)
