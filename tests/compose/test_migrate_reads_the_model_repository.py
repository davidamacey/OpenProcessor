"""db-migrate seeds the builtin engines from the shipped ``config.pbtxt`` files, so it mounts them.

``src/services/training/builtin_engines.py`` reads ``/app/models/<name>/config.pbtxt``; the api
image carries no model repository (``tests/test_dockerfile_hygiene.py``), so without the mount
the migration fails on a fresh stack.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('rel', ['docker-compose.yml', 'docker/test/compose.yml'])
def test_db_migrate_mounts_the_model_repository_read_only(rel: str) -> None:
    service = yaml.safe_load((ROOT / rel).read_text(encoding='utf-8'))['services']['db-migrate']
    targets = {v.split(':')[1]: v.split(':')[2:] for v in service['volumes'] if isinstance(v, str)}
    assert '/app/models' in targets, f'{rel}: db-migrate does not mount the model repository'
    assert targets['/app/models'] == ['ro'], f'{rel}: the model repository mount is not read-only'
