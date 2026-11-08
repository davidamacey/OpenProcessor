"""Run a real import end to end against the in-memory fakes and a real database.

The harness builds exactly what the job builds (``prepare_import`` -> pinned mapping ->
``ImportContext`` -> ``run_import_job`` -> ``finish``) with the OpenSearch, Triton and
PE-encoder I/O boundary faked. The ``dataset_imports`` row and the ledger are real rows in the
test database (a ``db`` test: ``pytestmark = pytest.mark.db``), so a test exercises the
production importer and ledger, not a stand-in.
"""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path
from typing import Any

from _class_registry_fake import InMemoryRegistry
from fixtures.job_rows import sql
from integration.ingest_fakes import FakePEEncoder, FakeTritonPool
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch
from src.config import CurationConfig, DetectionProfile, get_curation_config
from src.config.project_context import current_project_id
from src.db.models.datasets import DatasetImport
from src.services.curation.dataset_import import rows as import_rows, runner
from src.services.curation.dataset_import.context import ImportContext
from src.services.curation.dataset_import.job import ImportRunError, finish
from src.services.curation.dataset_import.mapping import ClassMappingEntry
from src.services.curation.dataset_import.options import (
    DatasetImportOptions,
    DatasetImportRequest,
    DatasetSource,
)
from src.services.curation.dataset_import.paths import dataset_path_guard
from src.services.curation.dataset_import.prepare import (
    materialize_created_classes,
    pin_project_view,
    prepare_import,
)
from src.services.curation.ingest import CurationIngestService


PG: Any = None
"""The test database, set by the directory's conftest."""


class FakeRun:
    """The slice of the job context the loop uses: ``cancel_after`` cancels once that many
    progress snapshots were written."""

    def __init__(self, cancel_after: int | None = None) -> None:
        self.snapshots: list[dict[str, Any]] = []
        self._cancel_after = cancel_after

    def cancelled(self) -> bool:
        return self._cancel_after is not None and len(self.snapshots) >= self._cancel_after

    async def progress(
        self,
        done: int,
        total: int | None = None,
        stage: str | None = None,
        counters: Any = None,
        *,
        force: bool = False,  # noqa: ARG002
    ) -> None:
        self.snapshots.append(
            {'done': done, 'total': total, 'stage': stage, 'counters': dict(counters or {})}
        )


class _JobState:
    """What the old per-import state file served to a test: status, report, run facts."""

    def __init__(self, handle: ImportHandle) -> None:
        self._handle = handle

    def read(self) -> dict[str, Any]:
        [(report, scan)] = sql(
            PG,
            'SELECT report, scan FROM dataset_imports WHERE id = %s',
            uuid.UUID(self._handle.import_id),
        )
        return {
            **(scan or {}),
            'report': {k: v for k, v in (report or {}).items() if k != 'job_issues'} or None,
            'job_issues': (report or {}).get('job_issues') or [],
            'status': self._handle.status,
            'error': self._handle.error,
        }


class ImportHandle(import_rows.Ledger):
    """One import's ledger (what ``undo_import`` reads) with the facts a test asserts on."""

    def __init__(self, import_id: uuid.UUID) -> None:
        super().__init__(import_rows.project_sessions(), import_id)
        self.status = 'queued'
        self.error: str | None = None
        self.snapshots: list[dict[str, Any]] = []
        self.job = _JobState(self)

    def ledger_rows(self) -> list[dict[str, Any]]:
        found = sql(
            PG,
            'SELECT image_key, status, image_id, error, detail FROM dataset_import_images '
            "WHERE import_id = %s ORDER BY (detail->>'ord')::int",
            uuid.UUID(self.import_id),
        )
        return [import_rows.legacy_row(*r) for r in found]

    def read_mapping(self) -> dict[str, Any]:
        [(mapping,)] = sql(
            PG, 'SELECT mapping FROM dataset_imports WHERE id = %s', uuid.UUID(self.import_id)
        )
        return dict(mapping)


def write_image(path: Path, size: tuple[int, int] = (100, 100), color: str = 'red') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color=color).save(path, format='JPEG')


def write_yolo(
    root: Path,
    *,
    names: dict[int, str] | list[str],
    images: dict[str, list[str] | None],
    split: str = 'train',
    extra_yaml: str = '',
) -> None:
    """``images``: stem -> label rows (``None`` = no label file, ``[]`` = a
    reviewed negative)."""
    root.mkdir(parents=True, exist_ok=True)
    items = names.items() if isinstance(names, dict) else enumerate(names)
    names_yaml = '\n'.join(f'  {i}: {n}' for i, n in items)
    (root / 'data.yaml').write_text(f'{split}: images/{split}\n{extra_yaml}names:\n{names_yaml}\n')
    (root / f'labels/{split}').mkdir(parents=True, exist_ok=True)
    for i, (stem, rows) in enumerate(images.items()):
        write_image(
            root / f'images/{split}/{stem}.jpg', color=['red', 'green', 'blue', 'white'][i % 4]
        )
        if rows is not None:
            (root / f'labels/{split}/{stem}.txt').write_text(''.join(r + '\n' for r in rows))


def write_yolo_splits(
    root: Path, *, names: list[str], splits: dict[str, dict[str, list[str] | None]]
) -> None:
    """A YOLO dataset with several splits (``write_yolo`` writes one)."""
    root.mkdir(parents=True, exist_ok=True)
    names_yaml = '\n'.join(f'  {i}: {n}' for i, n in enumerate(names))
    keys = '\n'.join(f'{split}: images/{split}' for split in splits)
    (root / 'data.yaml').write_text(f'{keys}\nnames:\n{names_yaml}\n')
    index = 0
    for split, images in splits.items():
        (root / f'labels/{split}').mkdir(parents=True, exist_ok=True)
        for stem, rows in images.items():
            write_image(
                root / f'images/{split}/{stem}.jpg',
                size=(100 + index, 100),  # distinct bytes per image: no dedup
                color=['red', 'green', 'blue', 'white'][index % 4],
            )
            index += 1
            if rows is not None:
                (root / f'labels/{split}/{stem}.txt').write_text(''.join(r + '\n' for r in rows))


class Harness:
    def __init__(
        self, tmp_path: Path, monkeypatch: Any, *, detections: Any = None, root: Path | None = None
    ) -> None:
        from src.services.curation import image_serving

        self.tmp = tmp_path
        import src.config.curation_accessors as curation_config_mod

        monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
        monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
        monkeypatch.setattr(
            image_serving,
            '_configured_roots',
            lambda config=None: ((root or tmp_path).resolve(),),  # noqa: ARG005
        )
        self.os = QueryFakeOpenSearch()
        self.registry = InMemoryRegistry(tmp_path)
        self.triton = FakeTritonPool()
        if detections is not None:
            self.triton = detections
        self.pe = FakePEEncoder()
        cfg = get_curation_config()
        self.cfg = dataclasses.replace(
            cfg if isinstance(cfg, CurationConfig) else CurationConfig(),
            crop_cache_dir=tmp_path / 'crops',
        )
        self.profile = DetectionProfile(
            name='primary', detector_model='fake_item_detector', assigns_class=False
        )
        self.service = CurationIngestService(
            opensearch=self.os,
            triton_pool=self.triton,
            registry=self.registry,
            profile=self.profile,
            pe_encoder=self.pe,
            config=self.cfg,
        )
        self.guard = dataset_path_guard()
        self.last_store: ImportHandle | None = None
        self.pending: list[str] = []

    @property
    def items(self) -> dict[str, dict[str, Any]]:
        return self.os.docs(self.cfg.items_index)

    @property
    def images(self) -> dict[str, dict[str, Any]]:
        return self.os.docs(self.cfg.images_index)

    def request(
        self,
        root: Path,
        mapping: list[ClassMappingEntry] | None = None,
        *,
        accept: bool = False,
        **options: Any,
    ) -> DatasetImportRequest:
        return DatasetImportRequest(
            source=DatasetSource(path=str(root)),
            mapping=mapping or [],
            accept_suggestions=accept,
            options=DatasetImportOptions(**options),
        )

    def prepare(self, request: DatasetImportRequest):
        return prepare_import(
            request, pin_project_view(self.registry.load()), path_guard=self.guard
        )

    async def begin(self, request: DatasetImportRequest, prepared: Any) -> ImportHandle:
        """What a start persists: the import row (request, pinned mapping, scan facts)."""
        import_id = uuid.uuid4()
        async with import_rows.project_sessions()() as session:
            session.add(
                DatasetImport(
                    id=import_id,
                    project_id=current_project_id(),
                    source_key=request.source.path,
                    format=prepared.scan.format,
                    mapping=runner.pinned_mapping(prepared),
                    options=request.model_dump(mode='json'),
                    scan=runner.scan_document(
                        prepared, request, [], freeze_test=runner.freeze_default(prepared, request)
                    ),
                )
            )
        handle = ImportHandle(import_id)
        self.last_store = handle
        return handle

    def context(
        self, store: ImportHandle, request: DatasetImportRequest, prepared: Any
    ) -> ImportContext:
        return ImportContext(
            import_id=store.import_id,
            options=request.options,
            resolved=prepared.resolved,
            profile=prepared.view.profile,
            parents=prepared.parents,
            source_sha=prepared.source_sha,
            source_format=prepared.scan.format,
            source_root=prepared.scan.root,
            opensearch=self.os,
            service=self.service,
            images_index=self.cfg.images_index,
            items_index=self.cfg.items_index,
            crop_cache_dir=self.cfg.crop_cache_dir,
            upload_root=self.tmp / 'uploads',
            export_root=None,
            freeze_test=runner.freeze_default(prepared, request),
            ledger=store,
            enqueue_pending=self._record_pending,
        )

    async def _record_pending(self, image_ids: list[str]) -> None:
        self.pending.extend(image_ids)

    async def _execute(
        self,
        store: ImportHandle,
        ctx: ImportContext,
        entries: list[Any],
        *,
        cancel_after: int | None = None,
    ) -> None:
        """Run the loop and settle the way the job wrapper does; a crash (any exception out of
        the loop) leaves the import ``interrupted``, as the reaper would."""
        control = FakeRun(cancel_after)
        store.status = 'running'
        try:
            outcome = await runner.run_import_job(control, ctx, entries)
            record = await self._record(store)
            result = await finish(store.sessions, ctx, record, outcome, total=len(entries))
        except ImportRunError as exc:
            store.status, store.error = 'failed', str(exc)
            return
        except BaseException:
            store.status = 'interrupted'
            raise
        finally:
            store.snapshots = control.snapshots
        if outcome.cancelled:
            store.status = 'cancelled'
        else:
            with_errors = getattr(result, 'with_errors', False)
            store.status = 'completed_with_errors' if with_errors else 'completed'

    async def _record(self, store: ImportHandle) -> import_rows.ImportRecord:
        async with store.sessions() as session:
            record = await import_rows.get(session, uuid.UUID(store.import_id))
        assert record is not None
        return record

    async def run(
        self,
        request: DatasetImportRequest,
        *,
        after_chunk: Any = None,
        cancel_after: int | None = None,
    ) -> tuple[ImportHandle, Any]:
        prepared = self.prepare(request)
        assert not prepared.resolved.errors, prepared.resolved.errors
        await materialize_created_classes(prepared.resolved, self.registry)
        store = await self.begin(request, prepared)
        ctx = self.context(store, request, prepared)
        ctx.after_chunk = after_chunk
        await self._execute(
            store, ctx, runner.ordered_entries(prepared.scan.entries), cancel_after=cancel_after
        )
        return store, ctx

    async def resume(self, store: ImportHandle, request: DatasetImportRequest) -> ImportContext:
        """Resume exactly as the job does at the start of an attempt: the dataset is scanned
        again (and must be unchanged); everything else comes from what the START persisted."""
        record = await self._record(store)
        entries = runner.rescan(request, record.scan.get('source_sha'), path_guard=self.guard)
        resolved, profile, parents = runner.load_pinned(record.mapping)
        await materialize_created_classes(resolved, self.registry)
        ctx = ImportContext(
            import_id=store.import_id,
            options=request.options,
            resolved=resolved,
            profile=profile,
            parents=parents,
            source_sha=record.scan['source_sha'],
            source_format=record.scan['source_format'],
            source_root=Path(record.scan['source_root']),
            opensearch=self.os,
            service=self.service,
            images_index=self.cfg.images_index,
            items_index=self.cfg.items_index,
            crop_cache_dir=self.cfg.crop_cache_dir,
            upload_root=self.tmp / 'uploads',
            export_root=None,
            freeze_test=bool(record.scan.get('freeze_test')),
            ledger=store,
            enqueue_pending=self._record_pending,
        )
        await self._execute(store, ctx, entries)
        return ctx

    def undo_context(self, import_id: str):
        from src.config.region_fields import get_region_fields
        from src.services.curation.dataset_import.undo import UndoContext

        return UndoContext(
            import_id=import_id,
            opensearch=self.os,
            images_index=self.cfg.images_index,
            items_index=self.cfg.items_index,
            crop_cache_dir=self.cfg.crop_cache_dir,
            region_fields=get_region_fields(),
            registry=self.registry,
        )


def map_all(registry: InMemoryRegistry, *names: str) -> list[ClassMappingEntry]:
    """``map`` each name to its registry class, ``create`` the ones missing."""
    existing = {c.class_name: c.class_id for c in registry.load().classes}
    out = []
    for name in names:
        if name in existing:
            out.append(ClassMappingEntry(dataset_class=name, action='map', class_id=existing[name]))
        else:
            out.append(ClassMappingEntry(dataset_class=name, action='create', new_class_name=name))
    return out


def activate_region_profile(
    name: str = 'wheel_profile',
    region_class_name: str = 'wheel',
    parent_classes: tuple[str, ...] = (),
) -> None:
    """Register ``name`` as the active region profile (the conftest resets the
    registry after each test)."""
    from src.services.detection.profile_registry import register_profile

    register_profile(
        DetectionProfile(
            name=name,
            region_class_name=region_class_name,
            parent_classes=frozenset(parent_classes),
        ),
        default=True,
    )
