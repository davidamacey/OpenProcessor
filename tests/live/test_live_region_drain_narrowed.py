"""Live: items seeded under a broad region profile stop counting once the profile is narrowed.

No detection worker runs here, so the uploaded items stay pending. Activating a profile whose
``parent_classes`` names no class an item has takes them out of the worker's scope: the region
drain route then reports nothing pending and, after its stable window, ``drained`` true (the
server's verdict, ``GET /ingest/region_drain``). Pending the wave run.
"""

from __future__ import annotations

from typing import Any

import pytest

from ._ingest_support import ingest_project, picture
from .conftest import wait_until


pytestmark = pytest.mark.live

DETECTOR = 'yolov11_small_trt_end2end'


def _profile(api: Any, prefix: str, name: str, **extra: Any) -> None:
    body = {
        'detector_model': DETECTOR,
        'region_class_name': 'thing',
        'display_name': 'Things',
        'display_name_singular': 'Thing',
        **extra,
    }
    created = api.post(f'{prefix}/region_profiles', json={'name': name, 'body': body})
    assert created.status_code == 201, created.text
    activated = api.post(f'{prefix}/region_profiles/{name}/activate', json={})
    assert activated.status_code == 200, activated.text


def test_narrowing_the_profile_drains_the_items_it_seeded(api: Any, opensearch: Any) -> None:
    with ingest_project(api, opensearch, 'narrow') as project:
        _profile(api, project.prefix, 'broad')
        uploaded = api.post(
            f'{project.prefix}/ingest/upload',
            files=[
                ('images', (f'n{i}.jpg', picture((50 * i, 80, 200 - 40 * i)), 'image/jpeg'))
                for i in range(1, 4)
            ],
        )
        assert uploaded.status_code == 200, uploaded.text
        assert uploaded.json()['summary']['successful'] == 3, uploaded.text

        def drain() -> dict[str, Any]:
            resp = api.get(f'{project.prefix}/ingest/region_drain')
            assert resp.status_code == 200, resp.text
            return resp.json()

        seeded = wait_until(lambda: drain()['pending_detection'] == 3, timeout=60)
        assert seeded, f'the broad profile left {drain()} instead of 3 pending'
        assert drain()['drained'] is False

        _profile(api, project.prefix, 'narrow', parent_classes=['no_such_class'])

        settled = wait_until(
            lambda: (d := drain())['drained'] and d['total_unfinished'] == 0,
            timeout=240,
            interval=3,
        )
        assert settled, f'narrowing did not drain: {drain()}'
