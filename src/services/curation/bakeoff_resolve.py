"""Bake-off request resolution: the shared error, profiles, baselines and finished training runs.

The lowest layer of the bake-off job builders:

* :class:`BakeoffRequestError`, the error a caller can fix, with the HTTP status
  the router should answer;
* a profile name -> :class:`BakeoffProfile`, the baseline registry of a profile;
* a training run -> checkpoint, imgsz, class names, the export it trained on
  (:func:`resolve_run_model`).

:mod:`~src.services.curation.bakeoff_mapping` builds on this for the class
mapping and :mod:`~src.services.curation.bakeoff_jobs` for the job spec.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.curation.bakeoff import class_map as cmap
from src.config import get_curation_config
from src.config.env import env_path
from src.core.logging import get_logger
from src.schemas.bakeoff.models import BaselineModel
from src.services.curation import eval_datasets as evd
from src.services.training import jobs as train_jobs
from src.services.training.class_remap import resolve_class_remap
from src.services.training.promote_errors import ClassRemapUnreadableError


if TYPE_CHECKING:
    from scripts.curation.bakeoff.profile import BakeoffProfile
    from src.services.training.class_remap import ClassRemapResult


logger = get_logger(__name__)
_config = get_curation_config()

# Runs record checkpoint_path as the trainer saw it. Older runs used a /runs/...
# container path; the API sees the same files under this root.
RUNS_HOST_ROOT = env_path('OP_TRAIN_RUNS_ROOT', _config.state_dir / 'training_runs')
# Baseline registry (external models). Empty by default; a profile may name
# its own via baselines_path, OP_BAKEOFF_BASELINES_PATH replaces the default.
_DEFAULT_BASELINES_PATH = (
    Path(__file__).resolve().parents[3] / 'scripts/curation/bakeoff/baselines.json'
)
BASELINES_PATH = env_path('OP_BAKEOFF_BASELINES_PATH', _DEFAULT_BASELINES_PATH)

JOB_ID_RE = re.compile(r'[A-Za-z0-9_.:-]{1,64}')


class BakeoffRequestError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def safe_job_id(job_id: str) -> str:
    if not JOB_ID_RE.fullmatch(job_id):
        raise BakeoffRequestError(400, f'invalid job_id: {job_id!r}')
    return job_id


def host_checkpoint(checkpoint_path: str) -> Path:
    """A run's checkpoint path as this process sees it (``/runs/...`` -> host root)."""
    if checkpoint_path.startswith('/runs/'):
        return RUNS_HOST_ROOT / checkpoint_path[len('/runs/') :]
    return Path(checkpoint_path)


def infer_model_size(job_id: str, model_size: str | None) -> str | None:
    """Size from the manifest/status, else parsed from a ``..._yolo26n`` job-id suffix."""
    if model_size:
        return model_size
    m = re.search(r'yolo\d+([nsmlx])$', job_id)
    return m.group(1) if m else None


# =============================================================================
# Profiles + baselines
# =============================================================================


def resolve_profile(spec: str | None) -> BakeoffProfile:
    from scripts.curation.bakeoff.profile import resolve_profile as _resolve

    try:
        return _resolve(spec)
    except (OSError, TypeError, ValueError) as exc:
        raise BakeoffRequestError(400, str(exc)) from exc


def baselines_path_for(profile_name: str | None) -> Path:
    """The baseline registry of a registered profile (``None`` = the default registry)."""
    if not profile_name:
        return BASELINES_PATH
    if '/' in profile_name or profile_name.endswith('.json'):
        raise BakeoffRequestError(400, 'profile must be a profile name')
    from scripts.curation.bakeoff.profile import resolve_baselines_path

    return resolve_baselines_path(resolve_profile(profile_name), BASELINES_PATH)


def load_baselines(path: Path) -> list[BaselineModel]:
    """Registry entries that match the 7.6 schema; others are logged and skipped."""
    try:
        reg = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning('bakeoff_baselines_read_failed', path=str(path), error=str(exc))
        return []
    out: list[BaselineModel] = []
    for entry in reg.get('baselines', []) if isinstance(reg, dict) else []:
        try:
            out.append(BaselineModel.model_validate(entry))
        except ValueError as exc:
            logger.warning('bakeoff_baseline_invalid', path=str(path), error=str(exc))
    return out


def resolve_baseline(name: str, profile: BakeoffProfile) -> BaselineModel:
    from scripts.curation.bakeoff.profile import resolve_baselines_path

    path = resolve_baselines_path(profile, BASELINES_PATH)
    for b in load_baselines(path):
        if b.name == name:
            return b
    raise BakeoffRequestError(400, f'unknown baseline {name!r} (registry {path})')


# =============================================================================
# Training runs
# =============================================================================


@dataclass(frozen=True)
class RunModel:
    """A finished training run, resolved for scoring."""

    run_id: str
    checkpoint_path: str
    imgsz: int
    model_family: str | None
    model_size: str | None
    finished_at: str | None
    campaign_id: str | None
    single_cls: bool
    include_classes: list[int] | None
    remap: ClassRemapResult
    class_names: dict[int, str]
    train_export_dir: Path | None
    train_export_id: str | None
    train_export_id_map: dict[int, int]
    dataset_sha: str | None
    frozen_test_sha: str | None
    trainer_map50: float | None
    trainer_map50_split: str | None


def _model_names(remap: ClassRemapResult, train_dir: Path | None) -> dict[int, str]:
    if remap.names:
        return dict(enumerate(remap.names))
    train_names = cmap.read_names(train_dir / 'data.yaml') if train_dir else {}
    if remap.mapping:
        id_map = cmap.read_export_id_map(train_dir) if train_dir else {}
        return {
            new: train_names.get(id_map.get(orig, -1), f'class_{orig}')
            for orig, new in remap.mapping.items()
        }
    return train_names


async def resolve_run_model(run_id: str) -> RunModel:
    """Status + manifest + class remap of a finished run with its checkpoint on disk."""
    try:
        status = await train_jobs.read_status(run_id)
    except ValueError:
        status = None
    if status is None:
        raise BakeoffRequestError(400, f'unknown training run {run_id!r}')
    if status.state != 'finished' or not status.checkpoint_path:
        raise BakeoffRequestError(400, f'training run {run_id!r} is {status.state}, not finished')
    ckpt = host_checkpoint(status.checkpoint_path)
    if not ckpt.is_file():
        raise BakeoffRequestError(400, f'training run {run_id!r} checkpoint is missing: {ckpt}')
    manifest = await train_jobs.read_manifest(run_id) or {}
    lineage = manifest.get('lineage') or {}
    spec = manifest.get('spec') or {}
    if not lineage.get('export_dir'):
        job = await train_jobs.read_job_spec(run_id) or {}
        lineage = {
            'export_dir': job.get('dataset_export_dir'),
            'single_cls': job.get('single_cls'),
            'include_classes': job.get('include_classes'),
            **lineage,
        }
    try:
        remap = resolve_class_remap(job_id=run_id, checkpoint_path=ckpt, manifest=manifest)
    except ClassRemapUnreadableError as exc:
        raise BakeoffRequestError(422, f'training run {run_id!r}: {exc}') from exc

    train_dir: Path | None = None
    train_id: str | None = None
    if lineage.get('export_dir'):
        try:
            train_id = evd.export_id_for_dir(lineage['export_dir'])
        except evd.UnknownDatasetError:
            train_id = None
        else:
            train_dir = evd.export_root() / train_id.split(':', 1)[1]
    run_eval = status.eval or {}
    hyper = spec.get('hyperparameters') or {}
    return RunModel(
        run_id=run_id,
        checkpoint_path=status.checkpoint_path,
        imgsz=int(hyper.get('imgsz') or 640),
        model_family=spec.get('model_family'),
        model_size=infer_model_size(run_id, spec.get('model_size')),
        finished_at=status.finished_at,
        campaign_id=status.campaign_id,
        single_cls=bool(lineage.get('single_cls') or remap.single_cls),
        include_classes=lineage.get('include_classes') or remap.include_classes,
        remap=remap,
        class_names=_model_names(remap, train_dir),
        train_export_dir=train_dir,
        train_export_id=train_id,
        train_export_id_map=cmap.read_export_id_map(train_dir) if train_dir else {},
        dataset_sha=lineage.get('dataset_sha'),
        frozen_test_sha=lineage.get('frozen_test_sha'),
        trainer_map50=run_eval.get('map50'),
        trainer_map50_split=run_eval.get('split'),
    )
