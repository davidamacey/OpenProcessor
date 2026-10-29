"""Fakes for the auto-label loop tests: a world of clustered items with a ground-truth class
each, a deterministic VLM whose accuracy is set per class, and a helper that runs one loop
over them with the real driver, evaluator, writer and job directory.

Nothing here is the code under test: the item store is the query-capable fake OpenSearch and
the VLM is a labeler object answering from the ground truth.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch.registry import ClassRegistry
from src.config.curation_accessors import base_curation_config
from src.services.curation.assist.slot_write import RunMeta
from src.services.curation.assist_loop.budget import BudgetLimits, LoopBudget
from src.services.curation.assist_loop.driver import LoopDeps, LoopOutcome, LoopParams, run_loop
from src.services.curation.assist_loop.evaluate import VlmAsker
from src.services.curation.assist_loop.partition import (
    ClusterFieldPartition,
    PartitionReader,
    PartitionScope,
)
from src.services.curation.assist_loop.store import LoopStore
from src.services.curation.class_write_guard import ClassWriteGuard
from src.services.curation.vlm_class_batch import class_context
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_models import VlmClassPrediction
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


if TYPE_CHECKING:
    from pathlib import Path

    import pytest

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

    registry: ClassRegistry
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


def make_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    """A world whose registry holds :data:`CLASSES`, installed as the bound project's. The
    job files skip ``fsync``: durability is the store's own test, not these."""
    registry = ClassRegistry(path=tmp_path / 'class_registry.json')
    for name in CLASSES:
        registry.add_class(name)
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.get_class_registry', lambda: registry
    )
    monkeypatch.setattr(os, 'fsync', lambda _fd: None)
    return World(registry)


@dataclass
class Run:
    outcome: LoopOutcome
    store: LoopStore
    budget: LoopBudget


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
    }
    options.update(over)
    return LoopParams(**options)


async def drive(
    world: World,
    directory: Path,
    vlm: FakeVlm,
    *,
    max_crops: int = 100_000,
    max_requests: int | None = None,
    max_wall_s: float | None = None,
    vlm_concurrency: int = 4,
    scope: PartitionScope | None = None,
    clock: Callable[[], float] | None = None,
    store: LoopStore | None = None,
    reader: PartitionReader | None = None,
    **params: Any,
) -> Run:
    """Run one loop (or resume the one in ``directory``) to its end."""
    store = store or LoopStore(directory)
    lp = loop_params(**params)
    limits = BudgetLimits.from_request(
        max_vlm_crops=max_crops,
        max_vlm_requests=max_requests,
        max_wall_s=max_wall_s,
        images_per_call=lp.images_per_call,
        vlm_concurrency=vlm_concurrency,
        audit_max_crops=500,
    )
    budget = LoopBudget(store, limits, **({'clock': clock} if clock else {}))
    ctx = class_context(vlm, ClassWriteGuard('assist_run'), revision=None)
    meta = RunMeta(
        run_id=store.loop_id,
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
    outcome = await run_loop(
        store, lp, deps, budget, scope or PartitionScope(cluster_ids=None, min_root_size=40)
    )
    return Run(outcome, store, budget)


def decisions(store: LoopStore) -> dict[str, str]:
    """Final decision kind per node path."""
    out: dict[str, str] = {}
    for event in store.events():
        if event['event'] == 'decision':
            out[event['path']] = event['kind']
    return out


def propagated_ids(store: LoopStore) -> dict[str, list[str]]:
    return {
        e['path']: sorted(e['written_ids']) for e in store.events() if e['event'] == 'propagated'
    }


class Killed(BaseException):
    """The process died: nothing after this point reaches the disk or the item store."""


class KillSwitch:
    """Kills the loop at the ``occurrence``-th ``event`` (``when`` it is appended: just
    ``before`` or just ``after``). From then on every write to the job directory and the
    item store raises :class:`Killed`, like a dead process; reads still work so the test
    can look."""

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

    def arm(self, store: LoopStore, os_fake: QueryFakeOpenSearch) -> None:
        real_append = store.append
        real_json = store.write_json
        real_members = store.write_node_members
        real_bulk = os_fake.bulk
        real_update = os_fake.update

        def append(kind: str, /, **fields: Any) -> dict[str, Any]:
            self._check()
            self._count(kind, 'before')
            event = real_append(kind, **fields)
            self._count(kind, 'after')
            return event

        def write_json(name: str, value: Any) -> None:
            self._check()
            real_json(name, value)

        def write_members(path: str, ids: Any) -> str:
            self._check()
            return real_members(path, ids)

        async def bulk(**kw: Any) -> dict[str, Any]:
            self._check()
            return await real_bulk(**kw)

        async def update(**kw: Any) -> dict[str, Any]:
            self._check()
            return await real_update(**kw)

        store.append = append  # type: ignore[method-assign]
        store.write_json = write_json  # type: ignore[method-assign]
        store.write_node_members = write_members  # type: ignore[method-assign]
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
