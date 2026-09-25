"""Pins for ``CurationConfig`` / ``IndexRole`` / ``index_name`` (Chunk 0).

See ``docs/design/curation_design_rationale.md`` §2.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import CurationConfig, IndexRole, index_name


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
    assert isinstance(cfg.bakeoff_eval_root, Path)


def test_bakeoff_eval_root_default_is_relative_not_a_private_path() -> None:
    """CFG-6: the bake-off harness used to default to owner-private
    absolute paths (one of which named a licensed proprietary image
    corpus). The default must be a repo-relative path, never an
    absolute filesystem path baked into the source."""
    cfg = CurationConfig()
    assert not cfg.bakeoff_eval_root.is_absolute()


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


def test_from_env_overrides_bakeoff_eval_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_BAKEOFF_EVAL_ROOT', '/tmp/my_bakeoff_eval')
    cfg = CurationConfig.from_env()
    assert cfg.bakeoff_eval_root == Path('/tmp/my_bakeoff_eval')


def test_prompt_pack_path_defaults_to_none() -> None:
    """Unlike the other path fields, there is no generic on-disk default --
    most deployments never need a custom PromptPack (labeling-assist plan
    task a)."""
    cfg = CurationConfig()
    assert cfg.prompt_pack_path is None


def test_from_env_overrides_prompt_pack_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_PROMPT_PACK_PATH', '/tmp/my_pack.json')
    cfg = CurationConfig.from_env()
    assert cfg.prompt_pack_path == Path('/tmp/my_pack.json')


def test_from_env_prompt_pack_path_unset_stays_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_PROMPT_PACK_PATH', raising=False)
    cfg = CurationConfig.from_env()
    assert cfg.prompt_pack_path is None


# =============================================================================
# OP_MLFLOW_PUBLIC_URL (browser-reachable MLflow base for the served
# train/status + train/manifest mlflow_run_url)
# =============================================================================


def test_mlflow_public_url_defaults_to_none() -> None:
    assert CurationConfig().mlflow_public_url is None


def test_from_env_overrides_mlflow_public_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_MLFLOW_PUBLIC_URL', 'https://mlflow.example.com')
    assert CurationConfig.from_env().mlflow_public_url == 'https://mlflow.example.com'


def test_from_env_mlflow_public_url_unset_stays_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_MLFLOW_PUBLIC_URL', raising=False)
    assert CurationConfig.from_env().mlflow_public_url is None


# =============================================================================
# OP_SOURCE_PATH_ALIASES (named source roots served at /images/root/{alias})
# =============================================================================


def test_source_path_aliases_unset_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_SOURCE_PATH_ALIASES', raising=False)
    assert CurationConfig.from_env().source_path_aliases == {}


def test_source_path_aliases_from_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        'OP_SOURCE_PATH_ALIASES', '{"archive": "/data/archive", "nightly": "/data/nightly"}'
    )
    assert CurationConfig.from_env().source_path_aliases == {
        'archive': Path('/data/archive'),
        'nightly': Path('/data/nightly'),
    }


def test_source_path_aliases_from_pair_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_SOURCE_PATH_ALIASES', ' archive=/data/archive , nightly=/data/nightly,')
    assert CurationConfig.from_env().source_path_aliases == {
        'archive': Path('/data/archive'),
        'nightly': Path('/data/nightly'),
    }


@pytest.mark.parametrize(
    'raw',
    ['{not json', '["a"]', '{"a": 1}', 'archive', '=/data', 'archive=', 'a/b=/data'],
)
def test_source_path_aliases_malformed_raises(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv('OP_SOURCE_PATH_ALIASES', raw)
    with pytest.raises(ValueError, match='OP_SOURCE_PATH_ALIASES'):
        CurationConfig.from_env()


def test_prompt_pack_paths_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_PROMPT_PACK_PATHS', ' /packs/a.json, ,/packs/b.json ')
    assert CurationConfig.from_env().prompt_pack_paths == (
        Path('/packs/a.json'),
        Path('/packs/b.json'),
    )


def test_prompt_pack_paths_unset_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_PROMPT_PACK_PATHS', raising=False)
    assert CurationConfig.from_env().prompt_pack_paths == ()
