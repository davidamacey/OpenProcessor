"""Fakes for the auto-label loop tests: a world of clustered items with a ground-truth class
each, a deterministic VLM whose accuracy is set per class, and a helper that runs one loop
over them with the real driver, evaluator and writer, and the loop's record in PostgreSQL.

Nothing here is the code under test: the item store is the query-capable fake OpenSearch and
the VLM is a labeler object answering from the ground truth.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import sys
import uuid
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import psycopg
from _class_registry_fake import InMemoryRegistry, use_registry
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.storage.roots import RootedStore
from src.config.curation_accessors import base_curation_config
from src.db import session as db
from src.db.repos import assist_loop as budget_repo, assist_loop_record as rows
from src.services.curation.assist.slot_write import RunMeta
from src.services.curation.assist_loop.audit import AuditJudge, Auditor, AuditParams
from src.services.curation.assist_loop.budget import BudgetLimits, LoopBudget
from src.services.curation.assist_loop.driver import (
    LoopDeps,
    LoopOutcome,
    LoopParams,
    finish,
    prepare,
    schedule,
)
from src.services.curation.assist_loop.evaluate import VlmAsker
from src.services.curation.assist_loop.node_step import RunControl
from src.services.curation.assist_loop.partition import (
    ClusterFieldPartition,
    PartitionScope,
    RootReader,
)
from src.services.curation.assist_loop.record import LoopRecord
from src.services.curation.assist_loop.snapshots import SnapshotStore
from src.services.curation.class_write_guard import ClassWriteGuard
from src.services.curation.vlm_class_batch import class_context
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_models import VlmClassPrediction
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import pytest
    from fixtures.postgres import PgDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'db'))
from _research_seed import make_loop_run, make_project


ITEMS = base_curation_config().items_index
CLASSES = ('car', 'truck', 'bus', 'van', 'tractor')
CREATED_AT = '2026-10-28T00:00:00+00:00'
BOXES = [{'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.5, 0.5], 'state': 'proposed'}]


def unit(*parts: object) -> float:
    """A uniform number in [0, 1) fixed by ``parts``."""
    digest = hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()
    return int(digest[:12], 16) / float(1 << 48)


class FakeVlm:
    """Answers each crop with its true class with probability ``accuracy[class]`` (one fixed
    draw per crop, so a crop always gets the same answer), else with ``confusion[class]``;
    ``fail_rate`` of the crops get a failed request. ``answers`` pins individual crops."""

    def __init__(
        self,
        truth: Mapping[str, str],
        *,
        accuracy: float | Mapping[str, float] = 1.0,
        confusion: Mapping[str, str] | None = None,
        fail_rate: float = 0.0,
        seed: int = 0,
        answers: Mapping[str, str | None] | None = None,
        model: str = 'fake-vlm',
        usage_per_image: tuple[int, int] | None = None,
    ) -> None:
        self.truth = truth
        self.accuracy = accuracy
        self.confusion = dict(confusion or {})
        self.fail_rate = fail_rate
        self.seed = seed
        self.answers = dict(answers or {})
        self._pack = GENERIC_ITEM_PACK
        self.identity = VlmIdentity(endpoint_ref=f'{model}@1', model=model)
        self.asked: list[str] = []
        self.requests = 0
        self.before_answer: Callable[[list[str]], None] | None = None
        self.usage_per_image = usage_per_image
        """``(prompt, completion)`` tokens the fake endpoint reports per image; ``None``: no usage."""

    def _accuracy(self, cls: str) -> float:
        if isinstance(self.accuracy, Mapping):
            return float(self.accuracy.get(cls, 1.0))
        return float(self.accuracy)

    def _wrong(self, cls: str) -> str:
        if cls in self.confusion:
            return self.confusion[cls]
        others = [c for c in CLASSES if c != cls]
        return others[0]

    def answer_for(self, crop_id: str) -> str | None:
        """The class this VLM names for ``crop_id``; ``None`` for a failed request."""
        if crop_id in self.answers:
            return self.answers[crop_id]
        if unit(self.seed, 'fail', crop_id) < self.fail_rate:
            return None
        cls = self.truth[crop_id]
        return cls if unit(self.seed, 'right', crop_id) < self._accuracy(cls) else self._wrong(cls)

    async def label_or_propose_batch(
        self, crops: list[Any], _class_names: list[str], *, class_catalog: Any = None
    ) -> list[VlmClassPrediction]:
        del class_catalog
        ids = [c.img_id for c in crops]
        if self.usage_per_image is not None:
            from src.services.labeling.vlm_usage import record_call

            prompt, completion = self.usage_per_image
            await record_call(
                endpoint_name='fake',
                source=None,
                payload={},
                response={
                    'usage': {
                        'prompt_tokens': prompt * len(ids),
                        'completion_tokens': completion * len(ids),
                    }
                },
            )
        self.asked.extend(ids)
        self.requests += 1
        if self.before_answer is not None:
            self.before_answer(ids)
        await asyncio.sleep(0)
        out = []
        for crop_id in ids:
            name = self.answer_for(crop_id)
            if name is None:
                out.append(
                    VlmClassPrediction(
                        img_id=crop_id, class_name='', confidence='low', failure='request_failed'
                    )
                )
            else:
                out.append(VlmClassPrediction(img_id=crop_id, class_name=name, confidence='high'))
        return out


@dataclass
class World:
    """Items in candidate clusters, each with a true class and an embedding near its class."""

    registry: InMemoryRegistry
    os: QueryFakeOpenSearch = field(default_factory=lambda: QueryFakeOpenSearch({ITEMS: {}}))
    truth: dict[str, str] = field(default_factory=dict)

    def add_item(self, crop_id: str, cluster_id: int, truth: str, **fields: Any) -> None:
        axis = CLASSES.index(truth)
        rng = np.random.default_rng(int(unit('emb', crop_id) * 2**31))
        vector = np.zeros(8)
        vector[axis] = 1.0
        vector = vector + rng.normal(0, 0.03, 8)
        doc: dict[str, Any] = {
            'crop_id': crop_id,
            'image_id': f'img-{crop_id}',
            'image_path': '/data/frame.jpg',
            'bbox_norm': [0.1, 0.1, 0.9, 0.9],
            'cluster_id': cluster_id,
            'cluster_subid': None,
            'class_id': None,
            'class_name': None,
            'class_source': 'item_proposal',
            'label_source': None,
            'class_validated': False,
            'region_boxes': copy.deepcopy(BOXES),
            'pe_embedding': vector.tolist(),
            'updated_at': '2026-10-01T00:00:00+00:00',
        }
        doc.update(fields)
        self.os.docs(ITEMS)[crop_id] = doc
        self.truth[crop_id] = truth

    def add_cluster(self, cluster_id: int, mix: Mapping[str, int], *, prefix: str) -> list[str]:
        ids = []
        for cls, count in mix.items():
            for i in range(count):
                crop_id = f'{prefix}-{cls}-{i:04d}'
                self.add_item(crop_id, cluster_id, cls)
                ids.append(crop_id)
        return ids

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self.os.docs(ITEMS))

    def restore(self, docs: dict[str, dict[str, Any]]) -> None:
        self.os.store[ITEMS] = copy.deepcopy(docs)
        self.os.seq.clear()

    def doc(self, crop_id: str) -> dict[str, Any]:
        return self.os.docs(ITEMS)[crop_id]

    def class_id(self, name: str) -> int:
        return next(c.class_id for c in self.registry.load().classes if c.class_name == name)


def make_world(monkeypatch: pytest.MonkeyPatch) -> World:
    """A world whose registry holds :data:`CLASSES`, installed as the bound project's. The
    label-operation store is its own module's test: here a propagation is the same merge,
    written without a record."""
    from src.services.curation.assist_loop import propagate
    from src.services.curation.crop_bulk import occ_bulk_human_relabel

    registry = InMemoryRegistry()
    for name in CLASSES:
        registry.add(name)
    use_registry(registry)

    async def recorded(opensearch: Any, ids: list[str], *, op: Any, merger_for: Any) -> Any:
        out = await occ_bulk_human_relabel(
            opensearch, ids, merger_for(str(uuid.uuid4())), writer_id=op.writer
        )
        return {**out, 'op_id': None}

    monkeypatch.setattr(propagate, 'recorded_relabel', recorded)
    return World(registry)


SLUG = 'loop-test'


@dataclass
class LoopEnv:
    """A project, and where a run's blobs go; runs are made with :meth:`new_run`."""

    pg: PgDatabase
    project: uuid.UUID
    store: RootedStore

    @classmethod
    def make(cls, pg: PgDatabase, tmp_path: Path) -> LoopEnv:
        with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
            project = make_project(conn, f'loop-{uuid.uuid4().hex[:8]}')
        return cls(pg, project, RootedStore({'projects': tmp_path}))

    def new_run(
        self,
        *,
        max_crops: int = 100_000,
        max_requests: int | None = None,
        max_wall_s: float | None = None,
        images_per_call: int = 4,
        vlm_concurrency: int = 4,
        audit_max_crops: int = 500,
    ) -> RunRef:
        """A running job (fencing token 1) with its run and budget rows."""
        limits = BudgetLimits.from_request(
            max_vlm_crops=max_crops,
            max_vlm_requests=max_requests,
            max_wall_s=max_wall_s,
            images_per_call=images_per_call,
            vlm_concurrency=vlm_concurrency,
            audit_max_crops=audit_max_crops,
        )
        with psycopg.connect(self.pg.conninfo('super'), autocommit=True) as conn:
            run = make_loop_run(conn, self.project)
            job = conn.execute(
                'SELECT job_id FROM assist_loop_runs WHERE id = %s', (run,)
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO assist_loop_iterations (run_id, ordinal, status) VALUES (%s, 1, 'open')",
                (run,),
            )
            conn.execute(
                "UPDATE jobs SET status = 'running', fencing_token = 1, attempt = 1, "
                "owner = 'test', started_at = now(), heartbeat_at = now(), "
                "lease_expires_at = now() + interval '1 hour' WHERE id = %s",
                (job,),
            )
        return RunRef(run, job, 1, limits)

    async def sessions_for(self, engine: Any) -> Callable[[], Any]:
        @asynccontextmanager
        async def sessions() -> AsyncIterator[AsyncSession]:
            async with (
                AsyncSession(engine, expire_on_commit=False) as session,
                db.project_tx(session, self.project),
            ):
                yield session

        return sessions


@dataclass
class RunRef:
    run_id: uuid.UUID
    job_id: uuid.UUID
    token: int
    limits: BudgetLimits
    cancel: bool = False
    pause: bool = False
    seeded: bool = False


@dataclass
class Run:
    outcome: LoopOutcome
    record: LoopRecord
    budget: LoopBudget
    ref: RunRef


def loop_params(**over: Any) -> LoopParams:
    options: dict[str, Any] = {
        'seed': 0,
        'theta': 0.9,
        'alpha': 0.05,
        'max_dissent': 1,
        'max_depth': 3,
        'branching': 4,
        'min_node_size': 40,
        'node_concurrency': 4,
        'images_per_call': 4,
        'on_budget': 'complete',
    }
    options.update(over)
    return LoopParams(**options)


async def noop_progress(*_a: Any) -> None:
    return None


async def build_deps(
    world: World,
    vlm: FakeVlm,
    lp: LoopParams,
    record: LoopRecord,
    budget: LoopBudget,
    *,
    vlm_concurrency: int = 4,
    reader: RootReader | None = None,
    audit: AuditParams | None = None,
    judge: FakeVlm | None = None,
) -> LoopDeps:
    """The loop's dependencies over the fake world and the fake VLMs."""
    ctx = await class_context(vlm, ClassWriteGuard('assist_run'), revision=None)
    meta = RunMeta(
        run_id=record.loop_id,
        created_at=CREATED_AT,
        vlm_endpoint=vlm.identity.endpoint_ref,
        vlm_model=vlm.identity.model,
        prompt_pack=GENERIC_ITEM_PACK.name,
        prompt_pack_revision=None,
    )
    deps = LoopDeps(
        opensearch=world.os,  # type: ignore[arg-type]
        reader=reader or ClusterFieldPartition(world.os),  # type: ignore[arg-type]
        asker=VlmAsker(
            world.os,  # type: ignore[arg-type]
            ctx,
            meta,
            images_per_call=lp.images_per_call,
            semaphore=asyncio.Semaphore(vlm_concurrency),
        ),
    )
    if audit is not None and judge is not None:
        deps.auditor = Auditor(
            record,
            budget,
            AuditJudge(
                world.os,  # type: ignore[arg-type]
                await class_context(judge, ClassWriteGuard('assist_loop_audit'), revision=None),
                images_per_call=lp.images_per_call,
                semaphore=asyncio.Semaphore(vlm_concurrency),
            ),
            audit,
            seed=lp.seed,
        )
    return deps


async def drive(
    world: World,
    env: LoopEnv,
    vlm: FakeVlm,
    *,
    ref: RunRef | None = None,
    max_crops: int = 100_000,
    max_requests: int | None = None,
    max_wall_s: float | None = None,
    vlm_concurrency: int = 4,
    scope: PartitionScope | None = None,
    clock: Callable[[], float] | None = None,
    reader: RootReader | None = None,
    audit: AuditParams | None = None,
    judge: FakeVlm | None = None,
    audit_max_crops: int = 500,
    slice_due: Callable[[], bool] | None = None,
    arm: Callable[[LoopRecord], None] | None = None,
    **params: Any,
) -> Run:
    """Run one slice of a loop (a new one, or ``ref``'s) to its stop."""
    lp = loop_params(**params)
    ref = ref or env.new_run(
        max_crops=max_crops,
        max_requests=max_requests,
        max_wall_s=max_wall_s,
        images_per_call=lp.images_per_call,
        vlm_concurrency=vlm_concurrency,
        audit_max_crops=audit_max_crops,
    )
    engine = create_async_engine(env.pg.url('app'), poolclass=NullPool)
    try:
        sessions = await env.sessions_for(engine)
        async with sessions() as session:
            row = await session.run_sync(lambda s: rows.run_by_id(s, ref.run_id))
            if not ref.seeded:
                await session.run_sync(
                    lambda s: budget_repo.seed_budgets(s, ref.run_id, ref.limits.seed())
                )
                ref.seeded = True
        assert row is not None
        row = replace_run(row, seed=lp.seed)
        record = LoopRecord(
            sessions,
            row,
            token=ref.token,
            snapshots=SnapshotStore(SLUG, ref.run_id, store=env.store),
        )
        if arm is not None:
            arm(record)
        budget = LoopBudget(record, **({'clock': clock} if clock else {}))
        await budget.start_slice()
        deps = await build_deps(
            world,
            vlm,
            lp,
            record,
            budget,
            vlm_concurrency=vlm_concurrency,
            reader=reader,
            audit=audit,
            judge=judge,
        )
        control = RunControl(
            cancelled=lambda: ref.cancel,
            paused=lambda: ref.pause,
            slice_due=slice_due or (lambda: False),
            progress=noop_progress,
        )
        loop_run = await prepare(
            record,
            lp,
            deps,
            budget,
            scope or PartitionScope(cluster_ids=None, min_root_size=40),
            control,
        )
        outcome = await schedule(loop_run)
        await finish(loop_run, outcome)
        return Run(outcome, record, budget, ref)
    finally:
        await engine.dispose()


def replace_run(row: rows.RunRow, **over: Any) -> rows.RunRow:
    import dataclasses

    return dataclasses.replace(row, **over)


def decisions(record: LoopRecord) -> dict[str, str]:
    """Final decision kind per node path."""
    return {
        path: node.decision['kind']
        for path, node in record.tree.nodes.items()
        if node.decision is not None and node.step == 'done'
    }


async def reload(env: LoopEnv, ref: RunRef) -> LoopRecord:
    """A fresh record over the run's rows, as a new slice would load them."""
    engine = create_async_engine(env.pg.url('app'), poolclass=NullPool)
    try:
        sessions = await env.sessions_for(engine)
        async with sessions() as session:
            row = await session.run_sync(lambda s: rows.run_by_id(s, ref.run_id))
        assert row is not None
        record = LoopRecord(
            sessions,
            row,
            token=ref.token,
            snapshots=SnapshotStore(SLUG, ref.run_id, store=env.store),
        )
        await record.load()
        return record
    finally:
        await engine.dispose()


def propagated_ids(record: LoopRecord) -> dict[str, list[str]]:
    import asyncio as _a

    return {
        path: sorted(_a.run(record.written_ids(node)))
        for path, node in record.tree.nodes.items()
        if node.propagated is not None
    }


class Killed(BaseException):
    """The process died: nothing after this point reaches the tables or the item store."""


class KillSwitch:
    """Kills the loop at the ``occurrence``-th call of the record method ``event`` (``when`` it
    is made: just ``before`` or just ``after`` its rows are committed). From then on every
    write to the tables and the item store raises :class:`Killed`, like a dead process."""

    WRITES = (
        'admit',
        'select',
        'record_answers',
        'forget_failed',
        'decide',
        'gate_pass',
        'begin_propagate',
        'finish_propagate',
        'split',
        'record_audit_chunk',
        'set_audit',
        'save_counts',
        'pin_roots',
    )

    def __init__(self, event: str, occurrence: int, when: str = 'after') -> None:
        self.event = event
        self.occurrence = occurrence
        self.when = when
        self.seen = 0
        self.tripped = False

    def _check(self) -> None:
        if self.tripped:
            raise Killed

    def _count(self, kind: str, when: str) -> None:
        if kind == self.event and self.when == when:
            self.seen += 1
            if self.seen == self.occurrence:
                self.tripped = True
                raise Killed

    def arm_record(self, record: LoopRecord) -> None:
        for name in self.WRITES:
            real = getattr(record, name)

            def wrapped(*a: Any, _real: Any = real, _name: str = name, **kw: Any) -> Any:
                async def run() -> Any:
                    self._check()
                    self._count(_name, 'before')
                    out = await _real(*a, **kw)
                    self._count(_name, 'after')
                    return out

                return run()

            setattr(record, name, wrapped)

    def arm_os(self, os_fake: QueryFakeOpenSearch) -> None:
        real_bulk = os_fake.bulk
        real_update = os_fake.update

        async def bulk(**kw: Any) -> dict[str, Any]:
            self._check()
            return await real_bulk(**kw)

        async def update(**kw: Any) -> dict[str, Any]:
            self._check()
            return await real_update(**kw)

        os_fake.bulk = bulk  # type: ignore[method-assign]
        os_fake.update = update  # type: ignore[method-assign]
        self._restore = (os_fake, real_bulk, real_update)

    def disarm(self) -> None:
        """The process restarts: the item store accepts writes again."""
        os_fake, real_bulk, real_update = self._restore
        os_fake.bulk = real_bulk  # type: ignore[method-assign]
        os_fake.update = real_update  # type: ignore[method-assign]


def without_times(value: Any) -> Any:
    """``value`` with every timestamp key (``at``, ``*_at``, ``wall_s``) dropped, recursively."""
    if isinstance(value, dict):
        return {
            k: without_times(v)
            for k, v in value.items()
            if k not in ('at', 'wall_s') and not k.endswith('_at')
        }
    if isinstance(value, list):
        return [without_times(v) for v in value]
    return value


def counts_of(env: LoopEnv, run: Run) -> dict[str, Any]:
    """The iteration row's counts (the per-level rows and totals the status routes serve)."""
    import psycopg

    with psycopg.connect(env.pg.conninfo('super'), autocommit=True) as conn:
        found = conn.execute(
            'SELECT counts FROM assist_loop_iterations WHERE run_id = %s', (run.ref.run_id,)
        ).fetchone()
    assert found is not None
    return dict(found[0])
