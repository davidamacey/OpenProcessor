"""Live: a new project takes the crop space of the deployment's size and its vectors land in it.

The harness runs the API with the default ``PE_CROP_SIZE`` (224: ``docker/test/compose.yml`` does
not set it); ``VERIFY_PE_CROP_SIZE`` says otherwise when a stack is started with another size. A
project of the OTHER size cannot be made over HTTP (a candidate space of another size never becomes
ready, and a project keeps the space it was created with), so the refusal across sizes is the
database-tier test ``tests/embedding_spaces/test_crop_space_across_sizes.py``.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from ._ingest_support import count, ingest_project, picture
from .conftest import API_PREFIX, refresh


pytestmark = pytest.mark.live

SIZE = int(os.environ.get('VERIFY_PE_CROP_SIZE', '224'))
CROP_MODEL = f'pe_core_l14_crop{SIZE}'
PICTURES = {'red': (200, 30, 30), 'green': (30, 200, 30), 'blue': (30, 30, 200)}


def test_a_new_project_is_on_the_deployments_crop_space_and_its_vectors_are_stored_there(
    api: Any, opensearch: Any
) -> None:
    with ingest_project(api, opensearch, 'cropspace') as project:
        spaces = api.get(f'{project.prefix}/embedding/spaces')
        assert spaces.status_code == 200, spaces.text
        primary = {
            s['target']: s
            for s in spaces.json()['spaces']
            if s['role'] == 'primary' and s['status'] == 'ready'
        }
        for target in ('item', 'box'):
            assert target in primary, f'no ready primary {target} space: {spaces.text}'
            assert primary[target]['model_key'] == CROP_MODEL, (
                f'{target} space is {primary[target]["model_key"]}, the deployment runs {CROP_MODEL}'
                f' (PE_CROP_SIZE={SIZE}): {spaces.text}'
            )

        record = api.get(f'{API_PREFIX}/projects/{project.slug}')
        assert record.status_code == 200, record.text
        assert record.json()['encoder_mismatch'] is None, record.text

        resp = api.post(
            f'{project.prefix}/ingest/upload',
            files=[
                ('images', (f'{name}.jpg', picture(rgb), 'image/jpeg'))
                for name, rgb in PICTURES.items()
            ],
        )
        assert resp.status_code == 200, f'ingest into the matching project: {resp.text}'
        assert [r['status'] for r in resp.json()['results']] == ['success'] * 3, resp.text
        items = count(opensearch, project.items_index)
        assert items >= 3, items

        refresh(opensearch, project.items_index)
        after = api.get(f'{project.prefix}/embedding/spaces')
        assert after.status_code == 200, after.text
        item_space = next(
            s for s in after.json()['spaces'] if s['target'] == 'item' and s['role'] == 'primary'
        )
        assert item_space['model_key'] == CROP_MODEL, after.text
        assert item_space['docs_with_vector'] == items, (
            f'{item_space["docs_with_vector"]} of {items} stored objects carry a vector in the '
            f'item space: {after.text}'
        )
