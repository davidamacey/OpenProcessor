"""Pins for ``CurationConfig`` / ``IndexRole`` / ``index_name`` (Chunk 0).

See ``docs/design/curation_design_rationale.md`` §2.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.config import CurationConfig, IndexRole, index_name


if TYPE_CHECKING:
    import pytest


def test_defaults_are_generic_op_names() -> None:
    cfg = CurationConfig()
    assert cfg.images_index == 'op_images'
    assert cfg.items_index == 'op_items'
    assert cfg.labels_confirmed_index == 'op_labels_confirmed'
    assert cfg.classes_index == 'op_classes'
    assert cfg.clusters_index == 'op_clusters'
    assert cfg.api_prefix == '/curation'
    assert cfg.api_tag == 'Curation'


def test_defaults_use_path_types() -> None:
    cfg = CurationConfig()
    assert isinstance(cfg.class_registry_path, Path)
    assert isinstance(cfg.source_root, Path)
    assert isinstance(cfg.state_dir, Path)
    assert isinstance(cfg.crop_cache_dir, Path)


def test_is_frozen() -> None:
    cfg = CurationConfig()
    try:
        cfg.images_index = 'mutated'  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError('CurationConfig must be immutable (frozen dataclass)')


def test_index_name_resolves_each_role() -> None:
    cfg = CurationConfig(
        images_index='custom_images',
        items_index='custom_items',
        labels_confirmed_index='custom_labels',
        classes_index='custom_classes',
    )
    assert index_name(cfg, IndexRole.IMAGES) == 'custom_images'
    assert index_name(cfg, IndexRole.ITEMS) == 'custom_items'
    assert index_name(cfg, IndexRole.LABELS_CONFIRMED) == 'custom_labels'
    assert index_name(cfg, IndexRole.CLASSES) == 'custom_classes'


def test_from_env_overrides_only_set_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_ITEMS_INDEX', 'env_items')
    monkeypatch.setenv('OP_API_PREFIX', '/curate')
    cfg = CurationConfig.from_env()
    assert cfg.items_index == 'env_items'
    assert cfg.api_prefix == '/curate'
    # Unset vars fall back to the dataclass default.
    assert cfg.images_index == 'op_images'


def test_from_env_respects_custom_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MYAPP_CLASSES_INDEX', 'app_classes')
    cfg = CurationConfig.from_env(prefix='MYAPP_')
    assert cfg.classes_index == 'app_classes'
