"""A small multi-project world for combine tests: several projects whose
indexes live in one query-evaluating in-memory OpenSearch, real image files
under each project's own upload root, and per-project class registries.

Production-shaped: images are real JPEGs with distinct bytes, image ids and
crop ids come from the production functions, registries are documents in the project's settings index.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import uuid
from typing import TYPE_CHECKING, Any

from _class_registry_fake import registry_of
from curation.query_fakes import QueryFakeOpenSearch
from PIL import Image

from src.clients.curation_opensearch.base import default_knn_spec
from src.config.curation_accessors import base_curation_config
from src.config.curation_roles import IndexRole
from src.config.project_context import current_project
from src.config.projects import ProjectRecord, new_project_record
from src.domain.embedding_spaces import EmbeddingSpaceError, ResolvedSpace, Target, legacy_names
from src.services.detection.geometry import crop_id as make_crop_id
from src.services.projects import registry as registry_module
from src.services.projects.combine import service as combine_service, spaces as combine_spaces
from src.services.projects.combine.models import CombineRequest
from src.services.projects.registry import ProjectRegistry


if TYPE_CHECKING:
    from pathlib import Path


NOW = '2026-10-01T00:00:00+00:00'
DIM = 4
DEFAULT_KEY = 'pe_core_l14_336_v1'


def legacy_space(target: Target, key: str = DEFAULT_KEY, dim: int = DIM) -> ResolvedSpace:
    """A project's primary space of ``target`` with the 0.6.0 physical names, as the registry
    serves it."""
    names = legacy_names(target)
    return ResolvedSpace(
        space_key=key,
        target=target,
        vector_field=names.vector_field,
        state_field=names.state_field,
        nested_path=names.nested_path,
        dim=dim,
        index_role='images' if target is Target.FRAME else 'items',
        model_key=key.rsplit('_v', 1)[0],
        model_version=1,
        triton_image_model=None,
        triton_text_model=None,
        knn_method=default_knn_spec(),
    )


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

    async def lookup(self, slug: str) -> ProjectRecord | None:
        return self.records.get(slug)

    def snapshot(self) -> dict[str, ProjectRecord]:  # type: ignore[override]
        return dict(self.records)


class World:
    def __init__(self, tmp_path: Path, monkeypatch: Any) -> None:
        import src.config.curation_accessors as curation_mod

        self.tmp = tmp_path
        monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
        monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
        monkeypatch.setattr(curation_mod, '_default_curation_config', None)
        self.fake = QueryFakeOpenSearch()
        self.records: dict[str, ProjectRecord] = {}
        self.registry = StubRegistry(self.records)
        monkeypatch.setattr(registry_module, '_REGISTRY', self.registry)
        self._seed = 0
        monkeypatch.setattr(combine_service, 'target_space_keys', lambda: frozenset({DEFAULT_KEY}))
        monkeypatch.setattr(combine_spaces, 'embedding_space', self._embedding_space)
        self.space_keys: dict[str, str] = {}
        """slug -> the space key of that project's primary spaces (default: ``DEFAULT_KEY``)."""
        self.settled: list[bool] = []
        self.pending: list[tuple[str, list[str]]] = []
        # set by a db test: every project then gets a real projects row under its own slug (the
        # test starts from an empty projects table, as pg_registry leaves it)
        self.pg: Any = None

    async def _embedding_space(self, target: Target, key: str | None = None) -> ResolvedSpace:
        """``embedding_space`` for the bound project: its primary space, or the live one named
        ``key`` (there is exactly one per target)."""
        held = self.space_keys.get(current_project().record.slug, DEFAULT_KEY)
        if key is not None and key != held:
            raise EmbeddingSpaceError('embedding_space_not_found', f'no live space {key}')
        return legacy_space(target, held)

    async def _client(self) -> QueryFakeOpenSearch:
        return self.fake

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
        if self.pg is not None:
            from _records_world import make_project_row

            record = dataclasses.replace(record, id=make_project_row(self.pg, slug))
        registry_of(record).seed(list(classes))
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

    def registry_json(self, slug: str) -> str:
        """The project's registry document as stable text, to compare before and after."""
        import json

        doc = registry_of(self.records[slug]).load().model_dump(exclude={'updated_at'})
        return json.dumps(doc, sort_keys=True, default=str)

    def registry_ids(self, slug: str) -> dict[str, int]:
        return {c.class_name: c.class_id for c in registry_of(self.records[slug]).load().classes}

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
        # A path-and-hash id, as ingest wrote before #358: one content at two paths is two ids.
        image_id = hashlib.sha256(f'{path}|{imohash}'.encode()).hexdigest()[:32]
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
            'label_op_id': 'lop_20260101T000000000000_abcdef',
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


class FakeJobContext:
    """The slice of ``JobContext`` the combine body uses, recording what it writes.

    ``cancel_after`` cancels once that many checkpoints were written (a cancel request the
    heartbeat would have delivered); ``resume_from`` is the cursor a previous attempt left.
    """

    def __init__(
        self,
        params: dict[str, Any],
        *,
        resume_from: Any = None,
        job_id: uuid.UUID | None = None,
        cancel_after: int | None = None,
    ) -> None:
        self.job_id = job_id or uuid.uuid4()
        self.project_id = uuid.uuid4()
        self.project_slug = 'combined'
        self.params = params
        self.resume_from = resume_from
        self.cursor: Any = resume_from
        self.checkpoints: list[Any] = []
        self.stages: list[str | None] = []
        self.lease_lost = False
        self._cancel_after = cancel_after

    def cancelled(self) -> bool:
        return self._cancel_after is not None and len(self.checkpoints) >= self._cancel_after

    def lost(self) -> bool:
        return self.lease_lost

    async def progress(
        self,
        done: int,
        total: int | None = None,
        stage: str | None = None,
        counters: Any = None,
        *,
        force: bool = False,
    ) -> None:
        del done, total, counters, force
        self.stages.append(stage)

    async def checkpoint(self, cursor: Any) -> None:
        self.cursor = cursor
        self.checkpoints.append(cursor)


@dataclasses.dataclass
class Run:
    """How one run of the body ended: its context, its result (``None`` when it stopped) and the
    error it raised, if any."""

    ctx: FakeJobContext
    params: dict[str, Any]
    document: dict[str, Any]
    result: dict[str, Any] | None
    error: BaseException | None = None

    @property
    def job_id(self) -> str:
        return str(self.ctx.job_id)

    @property
    def status(self) -> str:
        if self.error is not None:
            return 'failed'
        if self.result is None:
            return 'cancelled' if self.ctx.cancelled() else 'stopped'
        return 'completed'

    @property
    def report(self) -> dict[str, Any]:
        return (self.result or {}).get('report', {})


async def run_job(
    world: World,
    request: CombineRequest,
    *,
    target_slug: str | None = None,
    resume: Run | None = None,
    cancel_after: int | None = None,
    lease_lost: bool = False,
    reraise: bool = False,
) -> tuple[Run, ProjectRecord]:
    """Preview, plan and run the job body exactly as a worker does after ``start`` created the
    target (the lifecycle writes and the queueing are not part of the body). ``resume``: the
    earlier run to continue from its checkpoint, on its own plan."""
    from src.services.projects.combine import service
    from src.services.projects.combine.execute import load_plan, plan_documents, run_combine

    slug = target_slug or request.target.slug
    if resume is None:
        result, analysis = await service.preview(world.fake, request)
        assert result.ok, result.errors
        assert analysis is not None
        params, document = plan_documents(analysis, result.preview_sha)
        sources = analysis.records
        target = world.project(slug, [], status='building')
        ctx = FakeJobContext(params, cancel_after=cancel_after)
    else:
        params, document = resume.params, resume.document
        sources = [world.records[s.project] for s in load_plan(params, document).request.sources]
        target = world.records[slug]
        ctx = FakeJobContext(
            params,
            resume_from=resume.ctx.cursor,
            job_id=resume.ctx.job_id,
            cancel_after=cancel_after,
        )

    async def settle(ok: bool) -> None:
        world.settled.append(ok)

    async def enqueue(level: str, member_ids: list[str]) -> None:
        world.pending.append((level, list(member_ids)))

    ctx.lease_lost = lease_lost
    run = Run(ctx, params, document, None)
    try:
        run.result = await run_combine(
            ctx,
            world.fake,
            plan=load_plan(params, document),
            sources=sources,
            target=target,
            settle=settle,
            enqueue_pending=enqueue,
        )
    except Exception as exc:
        if reraise:
            raise
        run.error = exc
    return run, target


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
