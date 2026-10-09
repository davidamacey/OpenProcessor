"""Shard folding (owner D4, projects_plan.md §2.3): ``_configs_body()``'s
mapping is a conflict-free union with ``_settings_body()`` and
``_umap_viz_state_body()``, and every project -- ``default`` included, since
P1F made it an ordinary project (D-A) -- resolves 6 distinct index names via
``resources_for_new``, not 8."""

from __future__ import annotations

from src.clients.curation_opensearch.bodies_other import (
    _configs_body,
    _settings_body,
    _umap_viz_state_body,
)
from src.config.curation import IndexRole, base_curation_config
from src.config.projects import resources_for_new


def test_configs_mapping_union_has_no_type_conflicts() -> None:
    configs_props = _configs_body()['mappings']['properties']
    for other_body in (_settings_body(), _umap_viz_state_body()):
        for field, spec in other_body['mappings']['properties'].items():
            assert field in configs_props, f'{field!r} missing from _configs_body()'
            assert configs_props[field] == spec, f'{field!r} type mismatch'


def test_new_project_has_six_distinct_indexes() -> None:
    resources = resources_for_new('coco-wheels', base_curation_config())
    assert len(set(resources.indexes.values())) == 6
    assert resources.indexes[IndexRole.SETTINGS] == resources.indexes[IndexRole.CONFIGS]
    assert resources.indexes[IndexRole.UMAP_VIZ_STATE] == resources.indexes[IndexRole.CONFIGS]
    # Not folded: still their own names.
    assert resources.indexes[IndexRole.UMAP_STATE] != resources.indexes[IndexRole.CONFIGS]
    assert resources.indexes[IndexRole.CLASSES] != resources.indexes[IndexRole.CONFIGS]


def test_default_gets_the_same_folding_as_any_other_project() -> None:
    """``default`` is an ordinary project (P1F D-A): it goes through
    ``resources_for_new`` like any other slug, so it also folds."""
    resources = resources_for_new('default', base_curation_config())
    assert resources.indexes[IndexRole.SETTINGS] == resources.indexes[IndexRole.CONFIGS]
    assert resources.indexes[IndexRole.UMAP_VIZ_STATE] == resources.indexes[IndexRole.CONFIGS]
