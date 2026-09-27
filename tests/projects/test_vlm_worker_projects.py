"""vlm_worker.py: no 'op_items' literal, and per-project URL/index
resolution goes through scripts/curation/_project_worker_utils.py
rather than a frozen module-level constant."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.curation import vlm_worker
from scripts.curation._project_worker_utils import project_items_index, scoped_url
from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new


pytestmark = pytest.mark.unbound


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


def test_no_op_items_literal_in_source() -> None:
    text = Path(vlm_worker.__file__).read_text(encoding='utf-8')
    assert 'op_items' not in text


def test_vlm_label_batch_url_is_project_scoped() -> None:
    url = scoped_url('http://api', '/curation', 'alpha', '/vlm/label_batch')
    assert url == 'http://api/curation/projects/alpha/vlm/label_batch'


def test_project_items_index_resolves_per_project() -> None:
    alpha = _record('alpha')
    beta = _record('beta')

    with bind_project(alpha):
        alpha_index = project_items_index(alpha)
    with bind_project(beta):
        beta_index = project_items_index(beta)

    assert alpha_index != beta_index
    # Resolving alpha's index while alpha is bound gives alpha's own
    # configured items index, not a shared literal.
    from src.config.curation import IndexRole

    assert alpha_index == alpha.resources.indexes[IndexRole.ITEMS]
    assert beta_index == beta.resources.indexes[IndexRole.ITEMS]
