"""Pins for ``CurationConfig`` / ``IndexRole`` / ``index_name``.

See ``docs/design/curation_design_rationale.md`` §2.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import CurationConfig, IndexRole, index_name


def test_defaults_are_the_default_projects_names() -> None:
    """A bare ``CurationConfig`` carries the same names ``default`` is
    bootstrapped with (``resources_for_new('default', ...)``)."""
    from src.config.projects import resources_for_new

    cfg = CurationConfig()
    expected = resources_for_new('default', cfg)
    for role in IndexRole:
        assert index_name(cfg, role) == expected.indexes[role]
    assert cfg.items_index == 'op_prj_default__items'
    assert cfg.api_prefix == '/curation'
    assert cfg.api_tag == 'Curation'


def test_defaults_use_path_types() -> None:
    cfg = CurationConfig()
    assert isinstance(cfg.class_registry_path, Path)
    assert isinstance(cfg.source_root, Path)
    assert isinstance(cfg.state_dir, Path)
    assert isinstance(cfg.crop_cache_dir, Path)
    assert isinstance(cfg.bakeoff_eval_root, Path)


def test_bakeoff_eval_root_default_is_relative_not_a_private_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bake-off harness used to default to owner-private
    absolute paths (one of which named a licensed proprietary image
    corpus). The default must be a repo-relative path, never an
    absolute filesystem path baked into the source."""
    monkeypatch.delenv('OP_PROJECTS_DATA_ROOT', raising=False)
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
    monkeypatch.setenv('OP_API_PREFIX', '/curate')
    cfg = CurationConfig.from_env()
    assert cfg.api_prefix == '/curate'
    # Unset vars fall back to the dataclass default.
    assert cfg.api_tag == 'Curation'


def test_from_env_respects_custom_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MYAPP_API_TAG', 'App')
    cfg = CurationConfig.from_env(prefix='MYAPP_')
    assert cfg.api_tag == 'App'


@pytest.mark.parametrize(
    ('var', 'field'),
    [
        ('OP_ITEMS_INDEX', 'items_index'),
        ('OP_CLASSES_INDEX', 'classes_index'),
        ('OP_REGISTRY_PATH', 'class_registry_path'),
        ('OP_EXPORT_ROOT', 'export_root'),
        ('OP_UPLOAD_ROOT', 'upload_root'),
        ('OP_BAKEOFF_EVAL_ROOT', 'bakeoff_eval_root'),
    ],
)
def test_no_env_var_configures_a_project_resource(
    monkeypatch: pytest.MonkeyPatch, var: str, field: str
) -> None:
    """Index names and per-project paths come only from the project
    record; the retired env vars are ignored."""
    monkeypatch.setenv(var, '/tmp/retired-value')
    assert str(getattr(CurationConfig.from_env(), field)) != '/tmp/retired-value'


def test_prompt_pack_path_defaults_to_none() -> None:
    """Unlike the other path fields, there is no generic on-disk default --
    most deployments never need a custom PromptPack."""
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
# OP_PROBE_ACTIONABLE_MIN_CONFIDENCE (the floor probe_pred_confidence must
# clear, alongside in-scope + disagreeing, for the item wire's
# probe_actionable to be true -- see src.services.curation.wire.probe_actionable)
# =============================================================================


def test_probe_actionable_min_confidence_default_is_half() -> None:
    assert CurationConfig().probe_actionable_min_confidence == 0.5


def test_from_env_overrides_probe_actionable_min_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_PROBE_ACTIONABLE_MIN_CONFIDENCE', '0.8')
    assert CurationConfig.from_env().probe_actionable_min_confidence == 0.8


def test_from_env_probe_actionable_min_confidence_unset_stays_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('OP_PROBE_ACTIONABLE_MIN_CONFIDENCE', raising=False)
    assert CurationConfig.from_env().probe_actionable_min_confidence == 0.5


# =============================================================================
# OP_REGION_MAX_BOXES_PER_WRITE (W8: served abuse guard, human box lists are
# unbounded -- this caps the element count of one write request, not a
# labeling rule)
# =============================================================================


def test_region_max_boxes_per_write_default_is_500() -> None:
    assert CurationConfig().region_max_boxes_per_write == 500


def test_from_env_overrides_region_max_boxes_per_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_REGION_MAX_BOXES_PER_WRITE', '250')
    assert CurationConfig.from_env().region_max_boxes_per_write == 250


def test_from_env_region_max_boxes_per_write_unset_stays_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('OP_REGION_MAX_BOXES_PER_WRITE', raising=False)
    assert CurationConfig.from_env().region_max_boxes_per_write == 500


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


# =============================================================================
# OP_GRAFANA_URL / OP_PROMETHEUS_URL / OP_DASHBOARDS_URL (served resource_links)
# =============================================================================


def test_monitoring_urls_default_to_none_and_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('OP_GRAFANA_URL', 'OP_PROMETHEUS_URL', 'OP_DASHBOARDS_URL'):
        monkeypatch.delenv(name, raising=False)
    cfg = CurationConfig.from_env()
    assert (cfg.grafana_url, cfg.prometheus_url, cfg.dashboards_url) == (None, None, None)
    monkeypatch.setenv('OP_GRAFANA_URL', ' http://g:1 ')
    monkeypatch.setenv('OP_DASHBOARDS_URL', 'http://d:2')
    cfg = CurationConfig.from_env()
    assert (cfg.grafana_url, cfg.prometheus_url, cfg.dashboards_url) == (
        'http://g:1',
        None,
        'http://d:2',
    )
