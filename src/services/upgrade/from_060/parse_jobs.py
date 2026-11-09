"""Job-shaped sources: ingest sources and their queues, dataset imports and their ledgers,
training runs, promotions, exports, combine runs, and the jobs that were alive at the upgrade.

Every 0.6.0 job family kept ``<job dir>/state.json`` (a ``FileJob``). A terminal job that a kept
row points at (an import, a training run, a combine run) becomes a terminal ``jobs`` row; a job
that was alive becomes an ``interrupted`` row with ``auto_resume = false`` (decision D-51) so
nothing restarts by itself; every other finished job history stays in the bundle only (D-52).
"""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.services.curation.bulk_ingest.source_queue import QueueEntry, file_key_for
from src.services.upgrade.from_060 import sources as legacy
from src.services.upgrade.from_060.ids import stable_id
from src.services.upgrade.from_060.model import (
    CombineSrc,
    ExportSrc,
    ImportSrc,
    IngestSourceSrc,
    JobRow,
    PromotionSrc,
    TrainingRunSrc,
)
from src.services.upgrade.from_060.parse_core import parse_ts


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from src.services.upgrade.from_060.model import ProjectSrc
    from src.services.upgrade.from_060.snapshot import BundleFile, Snapshot

INTERRUPTED_CODE = 'upgraded_from_0_6_0'
_TERMINAL = {
    'completed': 'completed',
    'completed_with_errors': 'completed_with_errors',
    'failed': 'failed',
    'cancelled': 'cancelled',
    'undone': 'completed',
    'finished': 'completed',
    'skipped': 'cancelled',
}
_FAMILY_QUEUE = 'jobs'
_JOB_DIR = re.compile(r'^(?P<job>[^/]+)/state\.json$')


def _status(raw: str | None) -> tuple[str, bool]:
    """``(0.7.0 status, was alive)``: anything not terminal arrives interrupted."""
    mapped = _TERMINAL.get(str(raw))
    return (mapped, False) if mapped else ('interrupted', True)


def _job_row(
    *,
    family: str,
    slug: str | None,
    legacy_id: str,
    state: Mapping[str, Any],
    params: dict[str, Any],
    fallback: dt.datetime,
    status_key: str = 'status',
    kind: str | None = None,
) -> JobRow:
    status, alive = _status(state.get(status_key))
    started = parse_ts(state.get('started_at'), fallback)
    finished = None
    if status != 'interrupted':
        finished = parse_ts(state.get('finished_at'), started)
    error = state.get('error')
    return JobRow(
        id=stable_id('job', family, slug or '', legacy_id),
        project_slug=slug,
        kind=kind or family,
        status=status,
        queue=_FAMILY_QUEUE,
        params={**params, 'legacy_id': legacy_id},
        progress={k: v for k, v in state.items() if k in {'images_done', 'images_total', 'phase'}},
        result=None,
        error_code=INTERRUPTED_CODE
        if alive
        else (None if status == 'completed' else 'failed_in_0_6_0'),
        error_message=(
            f'upgraded from 0.6.0 while the job was {state.get(status_key)}'
            if alive
            else (str(error) if error else None)
        ),
        created_at=started,
        started_at=started,
        finished_at=finished,
    )


def _job_dirs(snap: Snapshot, family: str, slug: str) -> Iterator[tuple[str, BundleFile]]:
    """``(job dir name, its state.json)`` of one family of one project."""
    for rel, file in sorted(snap.area_files(f'{family}:{slug}').items()):
        match = _JOB_DIR.match(rel)
        if match:
            yield match['job'], file


def _json(snap: Snapshot, area: str, rel: str) -> Any:
    file = snap.file(area, rel)
    if file is None:
        return None
    try:
        return file.json()
    except ValueError:
        return None


# --- dataset imports --------------------------------------------------------------------------


def parse_imports(snap: Snapshot, project: ProjectSrc) -> list[ImportSrc]:
    out: list[ImportSrc] = []
    area = f'dataset_import:{project.slug}'
    for legacy_id, file in _job_dirs(snap, 'dataset_import', project.slug):
        state: dict[str, Any] = file.json()
        request = _json(snap, area, f'{legacy_id}/request.json') or {}
        mapping = _json(snap, area, f'{legacy_id}/mapping.json') or {}
        scan = _json(snap, area, f'{legacy_id}/scan.json')
        job = _job_row(
            family='dataset_import',
            slug=project.slug,
            legacy_id=legacy_id,
            state=state,
            params={'import_key': state.get('import_key'), 'name': state.get('name')},
            fallback=project.updated_at,
        )
        root = str(state.get('source_root') or request.get('source', {}).get('path') or '')
        out.append(
            ImportSrc(
                id=stable_id('import', project.slug, legacy_id),
                legacy_id=legacy_id,
                project_slug=project.slug,
                job=job,
                format=str(state.get('source_format') or 'unknown'),
                source_key='source/' + root.strip('/') if root.strip('/') else 'source/unknown',
                mapping=mapping,
                options={**(request.get('options') or {}), 'legacy_source_root': root},
                scan=scan,
                report=state.get('report'),
                created_at=job.created_at,
                undone_at=parse_ts(state.get('finished_at'), job.created_at)
                if state.get('status') == 'undone'
                else None,
                undone=state.get('status') == 'undone',
            )
        )
    return out


_LEDGER_STATUS = {'pending': 'pending', 'ok': 'written', 'failed': 'failed', 'skipped': 'skipped'}


def import_ledger_rows(
    file: BundleFile, import_id: Any, *, undone: bool
) -> Iterator[tuple[Any, ...]]:
    """``(import_id, image_key, image_id, item_ids, status, error)`` per image: the latest event
    of each ``rel_path`` in the chunk file, as ``ImportStore.ledger_rows`` read it."""
    latest: dict[str, dict[str, Any]] = {}
    for row in file.jsonl():
        rel = row.get('rel_path')
        if isinstance(rel, str):
            latest[rel] = row
    for rel, row in latest.items():
        status = _LEDGER_STATUS.get(str(row.get('status')), 'failed')
        if undone and status == 'written':
            status = 'undone'
        item_ids = [str(i['crop_id']) for i in row.get('items') or [] if i.get('crop_id')]
        error = row.get('error') or row.get('error_kind')
        yield (
            import_id,
            rel,
            row.get('image_id'),
            item_ids,
            status,
            None if error is None else str(error),
        )


# --- ingest sources ---------------------------------------------------------------------------

_SOURCE_DIR = re.compile(r'^sources/(?P<id>[^/]+)/(?P<file>[^/]+)$')


def parse_ingest_sources(snap: Snapshot, project: ProjectSrc) -> list[IngestSourceSrc]:
    """Named sources. The queue database arrives as bytes; the ledger COPY reads its pending rows."""
    area = f'bulk_ingest:{project.slug}'
    by_id: dict[str, dict[str, BundleFile]] = {}
    for rel, file in snap.area_files(area).items():
        match = _SOURCE_DIR.match(rel)
        if match:
            by_id.setdefault(match['id'], {})[match['file']] = file
    out: list[IngestSourceSrc] = []
    for source_id, files in sorted(by_id.items()):
        if 'source.json' not in files:
            continue
        definition: dict[str, Any] = files['source.json'].json()
        state = files['state.json'].json() if 'state.json' in files else {}
        created = parse_ts(definition.get('created_at'), project.updated_at)
        poll = state.get('last_poll_started_at')
        queue = files.get('queue.sqlite')
        out.append(
            IngestSourceSrc(
                id=stable_id('ingest_source', project.slug, source_id),
                project_slug=project.slug,
                name=str(definition.get('name') or source_id),
                kind=str(definition.get('backend') or 'folder'),
                config={
                    k: definition.get(k)
                    for k in (
                        'bucket',
                        'prefix',
                        'path',
                        'extensions',
                        'source_tag',
                        'batch_size',
                        'concurrency',
                        'webhook_enabled',
                    )
                }
                | {'legacy_source_id': source_id},
                schedule={
                    'watch': definition.get('watch') or {},
                    'schedule': definition.get('schedule') or {},
                },
                enabled=bool(
                    (definition.get('watch') or {}).get('enabled')
                    or (definition.get('schedule') or {}).get('enabled')
                    or definition.get('webhook_enabled')
                ),
                created_at=created,
                updated_at=parse_ts(definition.get('updated_at'), created),
                scan_state={
                    k: v
                    for k, v in state.items()
                    if k in {'last_run_id', 'last_poll_error', 'failures'}
                }
                or None,
                last_poll_at=dt.datetime.fromtimestamp(float(poll), dt.UTC) if poll else None,
                queue_image=queue.data if queue is not None else None,
            )
        )
    return out


def pending_queue_rows(
    source_id: Any, image: bytes, *, backend: str, bucket: str = ''
) -> Iterator[tuple[Any, ...]]:
    """``(source_id, file_key, status, attempts, error, enqueued_at, updated_at)`` of the rows still
    to ingest (state ``pending``); done, failed and dead rows are history and are not loaded.

    The key is the one the 0.7.0 scanner writes for that file version (an S3 object reference with
    the source's bucket, or the folder path and its ``mtime_ns:size``), so the first scan after
    the upgrade finds the row already queued instead of ingesting the file again."""
    with tempfile.TemporaryDirectory(prefix='op-queue-') as tmp:
        path = Path(tmp) / 'queue.sqlite'
        path.write_bytes(image)
        conn = sqlite3.connect(path.as_posix())
        try:
            cursor = conn.execute(
                'SELECT key, etag, version, size, attempts, error, queued_at, updated_at '
                "FROM objects WHERE state = 'pending' ORDER BY queued_at, key"
            )
            seen: set[str] = set()
            for key, etag, version, size, attempts, error, queued, updated in cursor:
                if key in seen:  # one row per key (the queue keeps one per key and etag)
                    continue
                seen.add(key)
                entry = QueueEntry(key=key, etag=etag or '', version=version, size=size)
                yield (
                    source_id,
                    file_key_for(entry, backend=backend, bucket=bucket),
                    'pending',
                    int(attempts or 0),
                    error,
                    dt.datetime.fromtimestamp(float(queued), dt.UTC),
                    dt.datetime.fromtimestamp(float(updated), dt.UTC),
                )
        finally:
            conn.close()


# --- training runs, exports, promotions --------------------------------------------------------


def parse_training(snap: Snapshot, project: ProjectSrc) -> list[TrainingRunSrc]:
    area = f'train:{project.slug}'
    out: list[TrainingRunSrc] = []
    for rel, file in sorted(snap.area_files(area).items()):
        if not rel.endswith('.job.json'):
            continue
        run = rel.removesuffix('.job.json')
        spec: dict[str, Any] = file.json()
        status = _json(snap, area, f'{run}.status.json') or {}
        manifest = _json(snap, area, f'{run}.manifest.json')
        registry = _json(snap, area, f'{run}.registry_snapshot.json') or {}
        names = [
            str(c.get('class_name', c.get('name')))
            for c in sorted(
                registry.get('classes') or [], key=lambda c: c.get('class_id', c.get('id', 0))
            )
        ]
        job = _job_row(
            family='training',
            slug=project.slug,
            legacy_id=run,
            state={
                **status,
                'status': status.get('state'),
                'started_at': status.get('started_at') or spec.get('submitted_at'),
            },
            params={'spec': spec},
            fallback=project.updated_at,
            kind='train',
        )
        out.append(
            TrainingRunSrc(
                id=stable_id('training_run', project.slug, run),
                legacy_id=run,
                project_slug=project.slug,
                job=job,
                spec=spec,
                class_names=tuple(names),
                family=str(spec.get('model_family') or 'yolo26'),
                base_model=f'{spec.get("model_family", "yolo26")}{spec.get("model_size", "m")}',
                artifact_prefix=f'projects/{project.slug}/training/{run}',
                manifest=manifest,
                mlflow_run_id=status.get('mlflow_run_id'),
                created_at=job.created_at,
                finished_at=job.finished_at,
            )
        )
    return out


def parse_exports(snap: Snapshot, project: ProjectSrc) -> list[ExportSrc]:
    out: list[ExportSrc] = []
    area = f'exports:{project.slug}'
    for rel, file in sorted(snap.area_files(area).items()):
        directory = rel.removesuffix('/manifest.json')
        manifest: dict[str, Any] = file.json()
        names = manifest.get('class_names') or manifest.get('names') or []
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=int)]
        out.append(
            ExportSrc(
                id=stable_id('export', project.slug, directory),
                project_slug=project.slug,
                kind='single_class' if manifest.get('single_class') else 'yolo',
                class_names=tuple(str(n) for n in names),
                manifest_sha256=file.sha256,
                artifact_prefix=f'projects/{project.slug}/exports/{directory}',
                split_seed=int(manifest.get('seed') or manifest.get('split_seed') or 0),
                created_at=parse_ts(
                    manifest.get('created_at') or manifest.get('generated_at'), project.updated_at
                ),
            )
        )
    return out


def parse_promotions(
    snap: Snapshot, slugs: Mapping[str, ProjectSrc], runs: Mapping[tuple[str, str], TrainingRunSrc]
) -> list[PromotionSrc]:
    """Each ``<model repo>/<model>/promote.json`` is one live promotion of one model version."""
    out: list[PromotionSrc] = []
    for rel, file in sorted(snap.area_files('models').items()):
        if not rel.endswith('/promote.json'):
            continue
        triton = rel.split('/', 1)[0]
        promote: dict[str, Any] = file.json()
        slug = str(promote.get('project') or '')
        if slug not in slugs:
            continue
        remap = _json(snap, 'models', f'{triton}/class_remap.json')
        version = int(promote.get('version') or 1)
        labels = [
            c['name'] for c in sorted(promote.get('classes') or [], key=lambda c: c['model_id'])
        ]
        run_id = promote.get('job_id')
        out.append(
            PromotionSrc(
                model_id=stable_id('model', slug, triton, version),
                promotion_id=stable_id('promotion', slug, triton, version),
                project_slug=slug,
                triton_name=triton,
                triton_version=version,
                version=version,
                labels=tuple(labels),
                family='yolo26',
                class_remap=remap if remap is not None else promote.get('class_remap'),
                training_run_legacy_id=str(run_id)
                if run_id and (slug, str(run_id)) in runs
                else None,
                promoted_at=parse_ts(promote.get('promoted_at'), slugs[slug].updated_at),
                shared=bool(promote.get('shared')),
                artifact_key=f'models/{triton}/{version}',
            )
        )
    return out


# --- combine ----------------------------------------------------------------------------------


def parse_combine(snap: Snapshot, slugs: Mapping[str, ProjectSrc]) -> list[CombineSrc]:
    out: list[CombineSrc] = []
    for rel, file in sorted(snap.area_files('combine').items()):
        match = _JOB_DIR.match(rel)
        if not match:
            continue
        legacy_id = match['job']
        state: dict[str, Any] = file.json()
        target = str(state.get('target') or '')
        if target not in slugs:
            continue
        request = _json(snap, 'combine', f'{legacy_id}/request.json') or {}
        mapping = _json(snap, 'combine', f'{legacy_id}/mapping.json') or {}
        holdout = str(request.get('holdout') or request.get('holdout_mode') or 'union')
        job = _job_row(
            family='combine',
            slug=target,
            legacy_id=legacy_id,
            state=state,
            params={'sources': state.get('sources')},
            fallback=slugs[target].updated_at,
        )
        out.append(
            CombineSrc(
                job=job,
                target_slug=target,
                holdout_mode=holdout if holdout in {'union', 'recompute', 'drop'} else 'union',
                class_mapping=mapping.get('per_source', mapping),
                source_slugs=tuple(str(s) for s in state.get('sources') or []),
                created_at=job.created_at,
            )
        )
    return out


# --- jobs that were alive ----------------------------------------------------------------------

# Families whose jobs get their own typed rows above; the others are transient histories.
_TYPED = {'dataset_import', 'combine', 'training'}
_SKIP_FAMILIES = {'region_drain'}  # a project marker (load_project_state), not a job


def parse_interrupted(snap: Snapshot, project: ProjectSrc) -> list[JobRow]:
    """One ``interrupted`` row for each job a family left alive. The rest of the history stays
    in the bundle (D-52)."""
    out: list[JobRow] = []
    for family, _env, _default in legacy.JOB_FAMILIES:
        if family in _TYPED or family in _SKIP_FAMILIES:
            continue
        for legacy_id, file in _job_dirs(snap, family, project.slug):
            state: dict[str, Any] = file.json()
            if state.get('status') not in legacy.ACTIVE_JOB_STATUSES:
                continue
            out.append(
                _job_row(
                    family=family,
                    slug=project.slug,
                    legacy_id=legacy_id,
                    state=state,
                    params={},
                    fallback=project.updated_at,
                )
            )
    return out
