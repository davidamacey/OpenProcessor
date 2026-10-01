"""Run a real import end to end against the in-memory fakes.

The harness builds exactly what the route builds (``prepare_import`` ->
claim -> pinned mapping -> ``ImportContext`` -> ``run_import_job``) with the
OpenSearch, Triton and PE-encoder I/O boundary faked, so a test exercises
the production importer, not a stand-in.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from integration.ingest_fakes import FakePEEncoder, FakeTritonPool
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import CurationConfig, DetectionProfile, get_curation_config
from src.services.curation.dataset_import import runner
from src.services.curation.dataset_import.context import ImportContext
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
from src.services.curation.dataset_import.store import ImportStore, imports_root, new_import_id
from src.services.curation.ingest import CurationIngestService


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
        import src.config.curation as curation_config_mod

        monkeypatch.setenv('OP_DATASET_IMPORTS_DIR', str(tmp_path / 'imports'))
        monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
        monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
        monkeypatch.setattr(
            image_serving,
            '_configured_roots',
            lambda config=None: ((root or tmp_path).resolve(),),  # noqa: ARG005
        )
        self.os = QueryFakeOpenSearch()
        self.registry = ClassRegistry(path=tmp_path / 'class_registry.json')
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
        self.last_store: ImportStore | None = None

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
        return prepare_import(request, pin_project_view(self.registry), path_guard=self.guard)

    def context(
        self, store: ImportStore, request: DatasetImportRequest, prepared: Any
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
        )

    async def run(
        self, request: DatasetImportRequest, *, after_chunk: Any = None
    ) -> tuple[ImportStore, Any]:
        prepared = self.prepare(request)
        assert not prepared.resolved.errors, prepared.resolved.errors
        store, reused = runner.claim_import(request, prepared)
        assert not reused
        self.last_store = store
        materialize_created_classes(prepared.resolved, self.registry)
        runner.persist_pinned(store, prepared)
        runner.persist_scan_summary(store, prepared)
        store.job.update(freeze_test=runner.freeze_default(prepared, request))
        ctx = self.context(store, request, prepared)
        ctx.after_chunk = after_chunk
        await runner.run_import_job(ctx, store, runner.ordered_entries(prepared.scan.entries))
        return store, ctx

    async def resume(self, store: ImportStore, request: DatasetImportRequest) -> ImportContext:
        """Resume exactly as ``POST /imports/{id}/resume`` does: everything
        but the dataset path comes from what the START persisted."""
        runner.check_resumable(store)
        entries = runner.rescan_for_resume(store, request, path_guard=self.guard)
        runner.prepare_resume(store, self.registry)
        resolved, profile, parents = runner.load_pinned(store)
        state = store.job.read()
        ctx = ImportContext(
            import_id=store.import_id,
            options=request.options,
            resolved=resolved,
            profile=profile,
            parents=parents,
            source_sha=state['source_sha'],
            source_format=state['source_format'],
            source_root=Path(state['source_root']),
            opensearch=self.os,
            service=self.service,
            images_index=self.cfg.images_index,
            items_index=self.cfg.items_index,
            crop_cache_dir=self.cfg.crop_cache_dir,
            upload_root=self.tmp / 'uploads',
            export_root=None,
            freeze_test=bool(state.get('freeze_test')),
        )
        store.job.clear_signals()
        store.job.update(status='queued', error=None, finished_at=None)
        await runner.run_import_job(ctx, store, entries)
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

    def fresh_id(self) -> str:
        return new_import_id('0' * 64)

    def imports_dir(self) -> Path:
        return imports_root()


def map_all(registry: ClassRegistry, *names: str) -> list[ClassMappingEntry]:
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
