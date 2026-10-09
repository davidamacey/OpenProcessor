"""Read-only catalogs under ``/train``: profiles, augmentation presets, class presets, GPUs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.config import get_gpu_arbiter_config
from src.services.training.augmentation_presets import (
    AUGMENTATION_PRESETS,
    DEFAULT_AUGMENTATION_PRESET,
)
from src.services.training.gpu_arbiter import containers_to_stop
from src.services.training.jobs import Profile
from src.services.training.preflight_checks import read_trainer_gpu_order
from src.services.training.profiles import get_class_subset_presets, get_profiles


router = APIRouter(default_response_class=ORJSONResponse)


class ProfilesResponse(BaseModel):
    profiles: list[Profile]


@router.get('/profiles', response_model=ProfilesResponse)
async def list_profiles() -> ProfilesResponse:
    """Return the YOLO26 profile table for the form picker."""
    rows = [Profile(**p) for p in get_profiles()]
    return ProfilesResponse(profiles=rows)


class AugmentationPresetOption(BaseModel):
    """One selectable ``augmentation.preset``."""

    id: str
    label: str
    description: str
    orientation_sensitive: bool = Field(
        description='Horizontal flip is disabled for the whole run with this preset.'
    )


class AugmentationPresetsResponse(BaseModel):
    presets: list[AugmentationPresetOption]
    default: str = Field(description='Preset used when a job omits augmentation.preset.')


@router.get('/augmentation_presets', response_model=AugmentationPresetsResponse)
async def list_augmentation_presets() -> AugmentationPresetsResponse:
    """The augmentation presets the trainer can build, for the form picker.

    Served from the catalog the trainer itself builds from
    (``src/services/training/augmentation_presets.py``); ``/preflight`` and
    ``/start`` reject any other id.
    """
    return AugmentationPresetsResponse(
        presets=[
            AugmentationPresetOption(
                id=p.id,
                label=p.label,
                description=p.description,
                orientation_sensitive=p.orientation_sensitive,
            )
            for p in AUGMENTATION_PRESETS
        ],
        default=DEFAULT_AUGMENTATION_PRESET,
    )


class PresetsResponse(BaseModel):
    class_subset_presets: list[dict[str, Any]] = Field(default_factory=list)


@router.get('/presets', response_model=PresetsResponse)
async def list_presets() -> PresetsResponse:
    """Return server-side class-subset presets."""
    return PresetsResponse(class_subset_presets=get_class_subset_presets())


# =============================================================================
# GET /gpus -- served training GPU picker (backend owns the decisions)
# =============================================================================


class TrainGpuOption(BaseModel):
    """One selectable ``cuda_visible_devices`` value for the train form."""

    value: str
    gpu_ids: list[int]
    label: str
    advisory: str
    stops_containers: list[str] = Field(default_factory=list)
    default: bool = False


class TrainGpuOptionsResponse(BaseModel):
    options: list[TrainGpuOption]
    allowed_ids: list[int]
    unrestricted: bool


def _train_gpu_option_label(gpu_ids: list[int], gpu_labels: dict[int, str]) -> str:
    ids_str = ','.join(str(i) for i in gpu_ids)
    if len(gpu_ids) == 1:
        gid = gpu_ids[0]
        card = gpu_labels.get(gid)
        return f'{card} (GPU {gid})' if card else f'GPU {gid}'
    names = {gpu_labels.get(gid) for gid in gpu_ids}
    if len(names) == 1 and (only := next(iter(names))):
        return f'{len(gpu_ids)}× {only} (GPUs {ids_str})'  # noqa: RUF001 - intentional display glyph
    return f'GPUs {ids_str}'


def _train_gpu_option_advisory(stopped: tuple[str, ...]) -> str:
    if stopped:
        return f'Stops {", ".join(stopped)} for the run; restarted when it ends.'
    return 'Shares GPU(s) with running services; background workers pause for the run.'


def _build_train_gpu_option(gpu_ids: list[int], gpu_labels: dict[int, str]) -> TrainGpuOption:
    value = ','.join(str(i) for i in gpu_ids)
    stopped = containers_to_stop(value)
    return TrainGpuOption(
        value=value,
        gpu_ids=gpu_ids,
        label=_train_gpu_option_label(gpu_ids, gpu_labels),
        advisory=_train_gpu_option_advisory(stopped),
        stops_containers=list(stopped),
    )


@router.get('/gpus', response_model=TrainGpuOptionsResponse)
async def list_train_gpu_options() -> TrainGpuOptionsResponse:
    """Serve the training GPU picker: values, labels, and stop advisories.

    Backend owns these decisions -- the frontend must never hardcode a
    deployment's GPU topology. Unrestricted installs (no
    ``OP_GPU_ALLOWED_IDS``) get exactly one option: the resolved default.

    When the trainer's published ``gpu_order`` (see ``_trainer_gpu_order``)
    is known and non-empty, the option list is intersected with it -- a
    trainer physically attached to only host GPU 2 must not offer GPU 0 as
    selectable, even if ``OP_GPU_ALLOWED_IDS`` (a policy allowlist, not a
    physical-attachment fact) says otherwise.
    """
    from src.services.training.jobs import default_train_gpu_value

    arbiter_cfg = get_gpu_arbiter_config()
    allowed_ids = sorted(arbiter_cfg.allowed_gpu_ids)
    trainer_gpu_order = read_trainer_gpu_order()
    if trainer_gpu_order:
        allowed_ids = (
            [i for i in allowed_ids if i in trainer_gpu_order]
            if allowed_ids
            else sorted(trainer_gpu_order)
        )
    default_value = default_train_gpu_value()

    if not allowed_ids:
        option = _build_train_gpu_option(
            [int(t) for t in default_value.split(',')], arbiter_cfg.gpu_labels
        )
        return TrainGpuOptionsResponse(
            options=[option.model_copy(update={'default': True})],
            allowed_ids=[],
            unrestricted=True,
        )

    options = [_build_train_gpu_option([gid], arbiter_cfg.gpu_labels) for gid in allowed_ids]
    if len(allowed_ids) > 1:
        options.append(_build_train_gpu_option(allowed_ids, arbiter_cfg.gpu_labels))

    default_idx = next(
        (i for i, o in enumerate(options) if o.value == default_value),
        0,
    )
    options[default_idx] = options[default_idx].model_copy(update={'default': True})
    return TrainGpuOptionsResponse(options=options, allowed_ids=allowed_ids, unrestricted=False)
