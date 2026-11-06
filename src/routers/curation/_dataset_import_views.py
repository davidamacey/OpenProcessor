"""Build the ``/datasets`` wire responses from service objects (kept out of
the route module: routes decide, this module shapes)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from src.routers.curation._dataset_import_models import (
    DatasetClassRow,
    DatasetEstimate,
    DatasetImportJob,
    DatasetImportProgress,
    DatasetImportReportWire,
    DatasetImportSource,
    DatasetPreview,
    DatasetRegionInfo,
    DatasetSplitRow,
    DatasetTotals,
    DatasetUndoReportWire,
    DisagreementsWire,
    IndexWouldHaveMapped,
    MappingSuggestionWire,
    NextStep,
    ResolvedMapTarget,
)
from src.routers.curation._dataset_issue_wire import issues_to_wire
from src.services.curation.dataset_import.actions import import_actions
from src.services.curation.dataset_import.options import DatasetImportOptions
from src.services.curation.dataset_import.prepare import mapping_from_dict
from src.services.curation.dataset_import.status import ACTIVE_STATUSES


if TYPE_CHECKING:
    from src.services.curation.dataset_import.control import ImportView
    from src.services.curation.dataset_import.prepare import PreparedImport

STATUS_LABELS: dict[str, str] = {
    'queued': 'Queued',
    'running': 'Importing',
    'paused_backpressure': 'Paused: waiting for the region worker',
    'completed': 'Completed',
    'completed_with_errors': 'Completed with errors',
    'failed': 'Failed',
    'cancelled': 'Cancelled',
    'interrupted': 'Interrupted',
    'undoing': 'Undoing',
    'undone': 'Undone',
}


def _splits(prepared: PreparedImport) -> list[DatasetSplitRow]:
    rows: dict[str, dict[str, int]] = {}
    for e in prepared.scan.entries:
        r = rows.setdefault(
            e.split or 'unsplit',
            {'images': 0, 'labeled': 0, 'negatives': 0, 'unlabeled': 0, 'boxes': 0},
        )
        r['images'] += 1
        r['boxes'] += len(e.boxes)
        r[
            {'labeled': 'labeled', 'negative': 'negatives', 'unlabeled': 'unlabeled'}[e.label_state]
        ] += 1
    return [DatasetSplitRow(split=k, **v) for k, v in sorted(rows.items())]


def _classes(prepared: PreparedImport) -> list[DatasetClassRow]:
    by_id = {c.class_id: c for c in prepared.view.registry_classes if not c.deprecated}
    images_per_class: dict[str, int] = {}
    for e in prepared.scan.entries:
        for name in {b.dataset_class for b in e.boxes}:
            images_per_class[name] = images_per_class.get(name, 0) + 1
    op = prepared.scan.op_export
    source_ids = dict(getattr(op, 'source_classes', None) or {})
    out = []
    for name, boxes in sorted(prepared.scan.class_box_counts.items()):
        ds_id = prepared.dataset_class_ids.get(name)
        would = by_id.get(ds_id) if ds_id is not None else None
        s = prepared.suggestions[name]
        target = prepared.resolved.targets.get(name)
        merged = []
        if target is not None and target.class_id is not None:
            merged = [
                d for d in prepared.resolved.merged_from.get(target.class_id, []) if d != name
            ]
        out.append(
            DatasetClassRow(
                dataset_class=name,
                dataset_id=ds_id,
                boxes=boxes,
                images=images_per_class.get(name, 0),
                source_class_id=source_ids.get(name),
                index_would_have_mapped_to=(
                    IndexWouldHaveMapped(class_id=would.class_id, class_name=would.class_name)
                    if would
                    else None
                ),
                suggestion=MappingSuggestionWire(
                    action=s.action, class_id=s.class_id, class_name=s.class_name, match=s.match
                ),
                resolved=(
                    ResolvedMapTarget(
                        dataset_class=name,
                        kind=target.kind,
                        class_id=target.class_id,
                        class_name=target.class_name,
                        created=name in prepared.resolved.created_classes,
                    )
                    if target
                    else None
                ),
                merged_from=merged,
            )
        )
    return out


def preview_wire(prepared: PreparedImport, *, already_indexed: int) -> DatasetPreview:
    entries = prepared.scan.entries
    view = prepared.view
    has_region = any(t.kind == 'region' for t in prepared.resolved.targets.values())
    region = None
    if has_region:
        profile = view.profile
        region = DatasetRegionInfo(
            profile=({'name': profile.name, 'revision': profile.revision} if profile else None),
            region_class_name=profile.region_class_name if profile else None,
            parent_classes=list(profile.parent_classes) if profile else [],
            parents_mode=prepared.parents,
            standalone_boxes=prepared.standalone_boxes,
        )
    op = prepared.scan.op_export
    uses_detector = prepared.parents == 'detect' and has_region
    return DatasetPreview(
        project=view.project,
        format=prepared.scan.format,
        root=str(prepared.scan.root),
        source_sha=prepared.source_sha,
        import_key=prepared.import_key,
        op_export=dataclasses.asdict(op)
        if op is not None and dataclasses.is_dataclass(op)
        else None,
        splits=_splits(prepared),
        totals=DatasetTotals(
            images=len(entries),
            boxes=sum(len(e.boxes) for e in entries),
            images_already_indexed=already_indexed,
            images_to_ingest=len(entries) - already_indexed,
        ),
        classes=_classes(prepared),
        region=region,
        issues=issues_to_wire(prepared.issues),
        blocking=prepared.blocking,
        force_allowed=prepared.force_allowed(),
        estimate=DatasetEstimate(
            detector_images=len(entries) if uses_detector else 0,
            embeddings=len(entries) + sum(len(e.boxes) for e in entries),
        ),
    )


def _report_wire(raw: dict[str, Any]) -> DatasetImportReportWire:
    data = {k: v for k, v in raw.items() if k in DatasetImportReportWire.model_fields}
    return DatasetImportReportWire(
        **data,
        disagreements=DisagreementsWire(
            counts=raw.get('disagreement_counts') or {},
            samples=raw.get('disagreement_samples') or [],
        ),
    )


def _iso(value: Any) -> str | None:
    return None if value is None else value.isoformat()


def job_wire(view: ImportView, *, project: str, reused: bool = False) -> DatasetImportJob:
    record = view.record
    job = view.undo if view.mode == 'undo' and view.undo is not None else view.run
    mapping_raw = record.mapping
    resolved = mapping_from_dict(mapping_raw) if mapping_raw else None
    targets = []
    if resolved is not None:
        for name, t in sorted(resolved.targets.items()):
            targets.append(
                ResolvedMapTarget(
                    dataset_class=name,
                    kind=t.kind,
                    class_id=t.class_id,
                    class_name=t.class_name,
                    created=name in resolved.created_classes,
                )
            )
    progress = dict((view.run.progress if view.run else None) or {})
    counters = progress.get('counters') or {}
    total = int(record.scan.get('images_total') or 0)
    done = int(counters.get('images_done') or 0)
    rate = counters.get('images_per_s')
    undo = (view.undo.result or {}).get('undo') if view.undo is not None else None
    status = view.status
    return DatasetImportJob(
        project=project,
        import_id=view.import_id,
        job_id=None if job is None else str(job.id),
        import_key=record.scan.get('import_key', ''),
        name=record.scan.get('name', ''),
        status=status,
        reused=reused,
        progress=DatasetImportProgress(
            images_total=total,
            images_done=done,
            images_failed=int(counters.get('images_failed') or 0),
            chunks_total=int(counters.get('chunks_total') or 0),
            chunks_done=int(counters.get('chunks_done') or 0),
            images_per_s=rate,
            eta_s=round((total - done) / rate, 1) if rate and total > done else None,
        ),
        waiting_for='region_worker' if status == 'paused_backpressure' else None,
        report=_report_wire(record.report),
        mapping=targets,
        options=DatasetImportOptions(**((record.request.get('options')) or {})),
        source=DatasetImportSource(
            format=record.scan.get('source_format'),
            root=record.scan.get('source_root'),
            source_sha=record.scan.get('source_sha'),
        ),
        issues_summary=record.scan.get('issues', []),
        undo=DatasetUndoReportWire(**undo) if undo else None,
        actions=import_actions(view.state, other_live_id=view.other_live_id),
        next_steps=[NextStep(**s) for s in (view.run.result or {}).get('next_steps') or []]
        if view.run is not None
        else [],
        error=job.error_message if job is not None else None,
        started_at=_iso(job.started_at or job.created_at) if job is not None else None,
        updated_at=_iso(job.updated_at) if job is not None else None,
        finished_at=_iso(job.finished_at) if job is not None else None,
        poll_after_s=2 if status in ACTIVE_STATUSES else None,
        labels={'status': STATUS_LABELS},
    )


__all__ = ['STATUS_LABELS', 'job_wire', 'preview_wire']
