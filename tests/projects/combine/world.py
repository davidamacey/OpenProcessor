"""A small multi-project world for combine tests: several projects whose
indexes live in one query-evaluating in-memory OpenSearch, real image files
under each project's own upload root, and per-project class registries.

Production-shaped: images are real JPEGs with distinct bytes, image ids and
crop ids come from the production functions, registries are real files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
from typing import TYPE_CHECKING, Any

from curation.query_fakes import QueryFakeOpenSearch
from PIL import Image

from src.clients.curation_opensearch import ClassRegistry
from src.config.curation import IndexRole, base_curation_config
from src.config.projects import ProjectRecord, new_project_record
from src.services.curation.ingest_index import image_id_for
from src.services.detection.geometry import crop_id as make_crop_id
from src.services.projects import registry as registry_module
from src.services.projects.combine import service as combine_service
from src.services.projects.combine.models import CombineRequest
from src.services.projects.registry import ProjectRegistry


if TYPE_CHECKING:
    from pathlib import Path


NOW = '2026-10-01T00:00:00+00:00'
DIM = 4


def jpeg_bytes(seed: int) -> bytes:
    """A decodable JPEG whose bytes differ per ``seed``."""
    buf = io.BytesIO()
    Image.new('RGB', (32 + seed, 32), color=(seed * 7 % 256, seed * 13 % 256, 90)).save(
        buf, format='JPEG'
    )
    return buf.getvalue()


class StubRegistry(ProjectRegistry):
    """The project registry over a fixed set of records (no OpenSearch)."""

    def __init__(self, records: dict[str, ProjectRecord]) -> None:
        super().__init__(lambda: None)
        self.records = records

    async def ensure_fresh(self) -> None:
        return None

    def get(self, slug: str) -> ProjectRecord | None:
        return self.records.get(slug)

    def snapshot(self) -> dict[str, ProjectRecord]:  # type: ignore[override]
        return dict(self.records)


class World:
    def __init__(self, tmp_path: Path, monkeypatch: Any) -> None:
        import src.config.curation as curation_mod

        self.tmp = tmp_path
        monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
        monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
        monkeypatch.setenv('OP_COMBINE_JOBS_DIR', str(tmp_path / 'combine_jobs'))
        monkeypatch.setattr(curation_mod, '_default_curation_config', None)
        self.fake = QueryFakeOpenSearch()
        self.records: dict[str, ProjectRecord] = {}
        self.registry = StubRegistry(self.records)
        monkeypatch.setattr(registry_module, '_REGISTRY', self.registry)
        self._seed = 0
        monkeypatch.setattr(combine_service, 'target_embedding_dim', lambda: DIM)
        self.settled: list[bool] = []

    # ------------------------------------------------------------ projects

    def project(self, slug: str, classes: list[str], *, status: str = 'active') -> ProjectRecord:
        record = new_project_record(slug, base_curation_config(), status=status)  # type: ignore[arg-type]
        res = record.resources
        data = self.tmp / 'projects_data' / slug
        record = dataclasses.replace(
            record,
            resources=dataclasses.replace(
                res,
                class_registry_path=data / 'class_registry.json',
                upload_root=self.tmp / 'uploads' / slug,
                project_state_dir=self.tmp / 'state' / 'projects' / slug,
                export_root=data / 'exports',
            ),
        )
        record.resources.upload_root.mkdir(parents=True, exist_ok=True)
        record.resources.project_state_dir.mkdir(parents=True, exist_ok=True)
        registry = ClassRegistry(path=record.resources.class_registry_path)
        for name in classes:
            registry.add_class(name)
        self.records[slug] = record
        return record

    def items_index(self, slug: str) -> str:
        return self.records[slug].resources.indexes[IndexRole.ITEMS]

    def images_index(self, slug: str) -> str:
        return self.records[slug].resources.indexes[IndexRole.IMAGES]

    def items(self, slug: str) -> dict[str, dict[str, Any]]:
        return self.fake.docs(self.items_index(slug))

    def images(self, slug: str) -> dict[str, dict[str, Any]]:
        return self.fake.docs(self.images_index(slug))

    def registry_ids(self, slug: str) -> dict[str, int]:
        reg = ClassRegistry(path=self.records[slug].resources.class_registry_path)
        return {c.class_name: c.class_id for c in reg.load().classes}

    # -------------------------------------------------------------- images

    def next_seed(self) -> int:
        self._seed += 1
        return self._seed

    def add_image(
        self,
        slug: str,
        *,
        seed: int | None = None,
        items: list[dict[str, Any]] | None = None,
        split: str | None = None,
        negative: bool = False,
        vector: bool = False,
    ) -> tuple[str, list[str]]:
        """A real file under the project's upload root plus its docs.
        ``seed`` fixes the bytes (the same seed in two projects is a
        byte-identical image at a different path). Each item dict takes
        ``cls`` (class name), ``bbox`` and any item-doc field."""
        seed = self.next_seed() if seed is None else seed
        data = jpeg_bytes(seed)
        imohash = hashlib.sha256(data).hexdigest()[:32]
        path = self.records[slug].resources.upload_root / imohash[:2] / f'{imohash}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        image_id = image_id_for(str(path), imohash)
        names = self.registry_ids(slug)
        image_doc: dict[str, Any] = {
            'image_id': image_id,
            'image_path': str(path),
            'source': 'test',
            'width': 32 + seed,
            'height': 32,
            'imohash': imohash,
            'indexed_at': NOW,
        }
        if split:
            image_doc['dataset_split'] = split
        if negative:
            image_doc['import_label_state'] = 'negative'
            image_doc['negative_for'] = list(names)
        if vector:
            image_doc['pe_embedding'] = [0.1] * DIM
        self.images(slug)[image_id] = image_doc
        crop_ids = [self._item(slug, image_id, path, names, vector, s) for s in items or []]
        return image_id, crop_ids

    def _item(
        self,
        slug: str,
        image_id: str,
        path: Path,
        names: dict[str, int],
        vector: bool,
        spec: dict[str, Any],
    ) -> str:
        spec = dict(spec)
        bbox = list(spec.pop('bbox'))
        cls = spec.pop('cls', None)
        crop_id = make_crop_id(image_id, bbox)
        doc: dict[str, Any] = {
            'crop_id': crop_id,
            'image_id': image_id,
            'image_path': str(path),
            'bbox_norm': bbox,
            'source': 'test',
            'class_source': 'ingest',
            'label_source': 'detector',
            'class_validated': False,
            'test_holdout': False,
            'created_at': NOW,
            'updated_at': NOW,
            'cluster_id': 7,
            'cluster_subid': '7a',
            'class_id_history': [{'class_id': 99, 'at': NOW}],
        }
        if cls is not None:
            doc['class_id'] = names[cls]
            doc['class_name'] = cls
        if vector:
            doc['pe_embedding'] = [0.2] * DIM
        doc.update(spec)
        self.items(slug)[crop_id] = doc
        return crop_id

    # ------------------------------------------------------------- request

    def request(
        self,
        sources: list[str],
        mapping: dict[str, list[dict[str, Any]]],
        *,
        target: str = 'combined',
        **extra: Any,
    ) -> CombineRequest:
        include = extra.pop('include', {})
        return CombineRequest.model_validate(
            {
                'target': {'slug': target, 'display_name': target},
                'sources': [
                    {'project': s, 'include': {'label_states': include.get(s, 'all')}}
                    for s in sources
                ],
                'class_mapping': mapping,
                **extra,
            }
        )


async def run_job(
    world: World, request: CombineRequest, *, target_slug: str | None = None, resume: Any = None
) -> tuple[Any, ProjectRecord]:
    """Preview, persist the plan and run the job body exactly as ``start``
    does after creating the target (the lifecycle writes are not part of the
    job). ``resume``: an existing store to resume instead of starting."""
    from src.services.projects.combine import service
    from src.services.projects.combine.execute import load_plan, persist_plan, run_combine
    from src.services.projects.combine.store import create_job, new_job_id

    slug = target_slug or request.target.slug
    if resume is None:
        result, analysis = await service.preview(world.fake, request)
        assert result.ok, result.errors
        assert analysis is not None
        store = create_job(new_job_id())
        persist_plan(store, analysis, result.preview_sha)
        sources = analysis.records
        target = world.project(slug, [], status='building')
        store.job.write(
            {
                'status': 'queued',
                'target': slug,
                'sources': [r.slug for r in sources],
                'total': sum(s.images for s in analysis.stats),
            }
        )
    else:
        store = resume
        sources = [world.records[s.project] for s in load_plan(store).request.sources]
        target = world.records[slug]
        store.job.clear_signals()

    async def settle(ok: bool) -> None:
        world.settled.append(ok)

    await run_combine(
        world.fake,
        store=store,
        plan=load_plan(store),
        sources=sources,
        target=target,
        embedding_dim=DIM,
        settle=settle,
    )
    return store, target


def tree_hash(root: Path) -> str:
    """Names and bytes of every file under ``root`` (a missing root hashes
    like an empty one)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def snapshot_indexes(world: World, slugs: list[str]) -> dict[str, Any]:
    """Docs and per-doc versions of every index of ``slugs``."""
    out: dict[str, Any] = {}
    for slug in slugs:
        for idx in (world.items_index(slug), world.images_index(slug)):
            out[idx] = (
                {k: dict(v) for k, v in world.fake.docs(idx).items()},
                {k: v for k, v in world.fake.seq.items() if k[0] == idx},
            )
    return out
