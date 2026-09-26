"""P1 commit 2: CurationConfigView resolves project-scoped fields at
access time (docs/design/openprocessor_internal/projects_plan.md §3.3)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from src.config.curation import (
    PROJECT_SCOPED_FIELDS,
    CurationConfig,
    base_curation_config,
    get_curation_config,
)
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new


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


def test_every_field_is_classified_exactly_once() -> None:
    """Fails the moment a new CurationConfig field is added without being
    triaged into PROJECT_SCOPED_FIELDS (the rest are global by
    elimination) -- catches an unclassified field at test time rather
    than at a silent cross-project leak."""
    field_names = {f.name for f in dataclasses.fields(CurationConfig)}
    assert field_names >= PROJECT_SCOPED_FIELDS
    # every PROJECT_SCOPED_FIELDS name must be resolvable (no typos)
    for name in PROJECT_SCOPED_FIELDS:
        assert name in field_names


def test_scoped_field_resolves_per_bound_project() -> None:
    with bind_project(_record('alpha')):
        assert get_curation_config().items_index == 'op_prj_alpha__items'
    with bind_project(_record('beta')):
        assert get_curation_config().items_index == 'op_prj_beta__items'


def test_global_field_works_unbound() -> None:
    assert get_curation_config().api_prefix == base_curation_config().api_prefix


def test_unbound_scoped_field_falls_back_to_base() -> None:
    """Documented, temporary deviation from §3.3's "always raises unbound"
    rule -- see CurationConfigView's docstring."""
    assert get_curation_config().items_index == base_curation_config().items_index


def test_module_level_captured_view_follows_a_later_binding() -> None:
    config = get_curation_config()  # simulates `config = get_curation_config()` at import time
    assert config.items_index == base_curation_config().items_index
    with bind_project(_record('alpha')):
        assert config.items_index == 'op_prj_alpha__items'
    assert config.items_index == base_curation_config().items_index


def test_idx_helper_matches_index_name() -> None:
    from src.config import IndexRole, index_name
    from src.config.curation import idx

    with bind_project(_record('alpha')):
        assert idx(IndexRole.ITEMS) == index_name(get_curation_config(), IndexRole.ITEMS)
