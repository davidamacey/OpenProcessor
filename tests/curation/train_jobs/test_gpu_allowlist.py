"""The ``cuda_visible_devices`` allowlist and the default training GPU."""

from __future__ import annotations

import pytest

from src.config import GpuArbiterConfig
from src.services.training import job_models
from src.services.training.job_models import CampaignRunSpec, TrainCampaignSpec, TrainJobSpec


#
# The generic default (GpuArbiterConfig.allowed_gpu_ids == frozenset(), no
# restriction) accepts any GPU id combination. A deployment that pins
# training to specific GPUs (e.g. this deployment's two A6000s,
# {0, 2}, with GPU 1 -- a 3080 Ti -- reserved for an unrelated app)
# configures that allowlist and gets the restrictive behavior back --
# see the ``restricted_gpu_ids`` fixture.


def test_spec_accepts_gpu2_only() -> None:
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='2')
    assert spec.cuda_visible_devices == '2'


def test_spec_accepts_gpu0_only() -> None:
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='0')
    assert spec.cuda_visible_devices == '0'


def test_spec_accepts_any_gpu_id_by_default() -> None:
    """No allowlist configured -> permissive: an arbitrary id validates."""
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='7')
    assert spec.cuda_visible_devices == '7'


def test_spec_canonicalizes_dual_gpu_order() -> None:
    """'2,0' and '0,2' must compare equal downstream (arbiter lock, manifest)."""
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='2,0')
    assert spec.cuda_visible_devices == '0,2'


def test_default_cuda_visible_devices_is_neutral_single_gpu() -> None:
    """The generic default is a single neutral GPU id, not a
    deployment-specific dual-GPU pin."""
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x')
    assert spec.cuda_visible_devices == '0'


@pytest.mark.usefixtures('restricted_gpu_ids')
def test_spec_rejects_gpu1_when_allowlist_configured() -> None:
    """GPU 1 is the 3080 Ti reserved for an unrelated app in this
    deployment -- must never be selectable once the allowlist excludes it."""
    with pytest.raises(ValueError, match='allowed GPU id'):
        TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='1')


@pytest.mark.usefixtures('restricted_gpu_ids')
def test_spec_rejects_gpu1_mixed_with_valid_gpu_when_allowlist_configured() -> None:
    with pytest.raises(ValueError, match='allowed GPU id'):
        TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='0,1')


@pytest.mark.usefixtures('restricted_gpu_ids')
def test_spec_rejects_unknown_device_id_when_allowlist_configured() -> None:
    with pytest.raises(ValueError, match='allowed GPU id'):
        TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='7')


def test_spec_rejects_duplicate_device_id() -> None:
    """Duplicate-id rejection is unconditional -- it doesn't depend on an
    allowlist being configured."""
    with pytest.raises(ValueError, match='repeat'):
        TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='0,0')


@pytest.mark.usefixtures('restricted_gpu_ids')
def test_campaign_spec_rejects_gpu1_when_allowlist_configured() -> None:
    with pytest.raises(ValueError, match='allowed GPU id'):
        TrainCampaignSpec(
            dataset_export_dir='/data/exports/x',
            cuda_visible_devices='1',
            runs=[CampaignRunSpec(profile='probe')],
        )


def test_default_train_gpu_value_unrestricted_is_zero() -> None:
    assert job_models.default_train_gpu_value() == '0'


def test_default_train_gpu_value_unset_env_var_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_TRAIN_DEFAULT_GPUS', raising=False)
    assert job_models.default_train_gpu_value() == '0'


@pytest.mark.usefixtures('restricted_gpu_ids')
def test_default_train_gpu_value_uses_smallest_allowed_id() -> None:
    """OP_GPU_ALLOWED_IDS=0,2 (no OP_TRAIN_DEFAULT_GPUS) -> smallest id, '0'."""
    assert job_models.default_train_gpu_value() == '0'


def test_default_train_gpu_value_uses_smallest_allowed_id_gpu2_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OP_GPU_ALLOWED_IDS=2 -> default is '2', not '0' (which isn't allowed)."""
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    monkeypatch.setattr(
        gpu_arbiter_config_module,
        '_default_gpu_arbiter_config',
        GpuArbiterConfig(allowed_gpu_ids=frozenset({2})),
    )
    assert job_models.default_train_gpu_value() == '2'


def test_default_train_gpu_value_honors_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    monkeypatch.setattr(
        gpu_arbiter_config_module,
        '_default_gpu_arbiter_config',
        GpuArbiterConfig(allowed_gpu_ids=frozenset({0, 2}), default_train_gpus='2'),
    )
    assert job_models.default_train_gpu_value() == '2'


def test_default_train_gpu_value_rejects_env_outside_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misconfigured OP_TRAIN_DEFAULT_GPUS outside the allowlist must fail
    loudly (through the same validator specs use) rather than silently
    default to a disallowed GPU."""
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    monkeypatch.setattr(
        gpu_arbiter_config_module,
        '_default_gpu_arbiter_config',
        GpuArbiterConfig(allowed_gpu_ids=frozenset({0, 2}), default_train_gpus='1'),
    )
    with pytest.raises(ValueError, match='allowed GPU id'):
        job_models.default_train_gpu_value()


def test_spec_default_uses_default_train_gpu_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """TrainJobSpec()/TrainCampaignSpec() with no cuda_visible_devices pick up
    the configured default, not a hardcoded '0'."""
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    monkeypatch.setattr(
        gpu_arbiter_config_module,
        '_default_gpu_arbiter_config',
        GpuArbiterConfig(allowed_gpu_ids=frozenset({2})),
    )
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x')
    assert spec.cuda_visible_devices == '2'
    campaign = TrainCampaignSpec(
        dataset_export_dir='/data/exports/x', runs=[CampaignRunSpec(profile='probe')]
    )
    assert campaign.cuda_visible_devices == '2'
