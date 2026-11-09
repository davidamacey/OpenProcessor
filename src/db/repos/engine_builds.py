"""The registry rows an engine build writes: a ``models`` version and its ``model_engines``.

The export job (lane 5B) is the writer. The start path inserts the model version and one
``building`` engine row per Triton model the job will build, in the same transaction as the job
row; the handler then fills the facts of each engine and moves it to ``ready`` or ``failed``.
Every function takes a sync ``Session`` the caller already scoped and never commits; an async
caller reaches them through ``AsyncSession.run_sync``.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.training import Model, ModelEngine


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy.orm import Session

UNBUILT = 'pending'
"""``trt_version`` and ``gpu_arch`` of a row whose build has not run: the worker knows them."""


@dataclass(frozen=True)
class ModelVersion:
    id: uuid.UUID
    name: str
    version: int


@dataclass(frozen=True)
class EngineFacts:
    """Everything the build records about one engine (research GPU S-1..S-4, S-13)."""

    input_size: int
    input_contract: Mapping[str, Any]
    output_contract: Mapping[str, Any]
    build_profile: str
    build_settings: Mapping[str, Any]
    build_stamp: str
    onnx_sha256: str | None
    engine_sha256: str
    device_mib_measured: int | None
    max_batch: int
    opt_batch: int | None
    precision: str
    trt_version: str
    gpu_arch: str
    engine_storage_key: str


def add_model_version(
    session: Session,
    *,
    name: str,
    labels: Sequence[str],
    family: str,
    artifact_storage_key: str,
    created_by: str,
) -> ModelVersion:
    """A new deployment-wide ``uploaded`` version of ``name`` (the next number after the last)."""
    last = session.scalar(
        select(func.max(Model.version)).where(Model.project_id.is_(None), Model.name == name)
    )
    row = Model(
        project_id=None,
        name=name,
        version=(last or 0) + 1,
        origin='uploaded',
        family=family,
        task='detect',
        labels=list(labels),
        artifact_storage_key=artifact_storage_key,
        created_by=created_by,
    )
    session.add(row)
    session.flush()
    return ModelVersion(row.id, row.name, row.version)


def latest_model_version(session: Session, name: str) -> ModelVersion | None:
    row = session.execute(
        select(Model.id, Model.name, Model.version)
        .where(Model.project_id.is_(None), Model.name == name)
        .order_by(Model.version.desc())
        .limit(1)
    ).first()
    return None if row is None else ModelVersion(row.id, row.name, row.version)


def begin_engine(
    session: Session,
    *,
    model_id: uuid.UUID,
    triton_model_name: str,
    max_batch: int,
    build_job_id: uuid.UUID,
) -> uuid.UUID:
    """The ``building`` row of one engine. A failed attempt that never learned its build key is
    reset by the same statement, so a retry does not collide with its own leftover."""
    placeholder: dict[str, Any] = {
        'precision': 'fp16',
        'input_size': 640,
        'input_contract': {},
        'output_contract': {},
        'build_profile': 'default',
        'build_settings': {},
        'build_stamp': UNBUILT,
        'trt_version': UNBUILT,
        'gpu_arch': UNBUILT,
    }
    insert_row = pg_insert(ModelEngine).values(
        model_id=model_id,
        triton_model_name=triton_model_name,
        max_batch=max_batch,
        build_job_id=build_job_id,
        status='building',
        **placeholder,
    )
    statement = insert_row.on_conflict_do_update(
        constraint='uq_model_engines_build_key',
        set_={
            'model_id': model_id,
            'max_batch': max_batch,
            'build_job_id': build_job_id,
            'status': 'building',
            'built_at': None,
        },
    ).returning(ModelEngine.id)
    return uuid.UUID(str(session.execute(statement).scalar_one()))


def finish_engine(session: Session, engine_id: uuid.UUID, facts: EngineFacts) -> None:
    """Record the facts and mark the row ``ready``.

    A row already holding this build key (the same model rebuilt with the same TensorRT on the
    same architecture and profile) is replaced: the Triton directory it described was
    overwritten by this build. A different profile or precision is a different key, so an
    opt-in INT8 or ``small`` build never overwrites the default one.
    """
    name = session.scalar(select(ModelEngine.triton_model_name).where(ModelEngine.id == engine_id))
    if name is None:
        raise LookupError(f'no engine row {engine_id}')
    session.execute(
        delete(ModelEngine).where(
            ModelEngine.id != engine_id,
            ModelEngine.triton_model_name == name,
            ModelEngine.trt_version == facts.trt_version,
            ModelEngine.gpu_arch == facts.gpu_arch,
            ModelEngine.build_profile == facts.build_profile,
        )
    )
    session.execute(
        update(ModelEngine)
        .where(ModelEngine.id == engine_id)
        .values(
            precision=facts.precision,
            input_size=facts.input_size,
            input_contract=dict(facts.input_contract),
            output_contract=dict(facts.output_contract),
            build_profile=facts.build_profile,
            build_settings=dict(facts.build_settings),
            build_stamp=facts.build_stamp,
            onnx_sha256=facts.onnx_sha256,
            engine_sha256=facts.engine_sha256,
            device_mib_measured=facts.device_mib_measured,
            max_batch=facts.max_batch,
            opt_batch=facts.opt_batch,
            trt_version=facts.trt_version,
            gpu_arch=facts.gpu_arch,
            engine_storage_key=facts.engine_storage_key,
            status='ready',
            built_at=dt.datetime.now(dt.UTC),
        )
    )


def fail_building(session: Session, build_job_id: uuid.UUID) -> int:
    """Mark every engine a job left ``building`` as ``failed``; the number of rows moved."""
    result = session.execute(
        update(ModelEngine)
        .where(ModelEngine.build_job_id == build_job_id, ModelEngine.status == 'building')
        .values(status='failed')
    )
    return int(result.rowcount)  # type: ignore[attr-defined]  # an UPDATE returns a CursorResult


def ready_names(session: Session) -> set[str]:
    """Triton model names that have a ``ready`` engine row."""
    return set(
        session.scalars(
            select(ModelEngine.triton_model_name).where(ModelEngine.status == 'ready').distinct()
        )
    )
