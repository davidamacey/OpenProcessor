"""One node of the auto-label loop: draw a seeded sample, ask the VLM, apply the gate.

The sample of a node is its parent's answers that fall in it plus a seeded draw from its
other eligible members, up to the size the exact bound needs with no dissent
(:func:`~src.domain.loop_stats.required_sample`). Failed answers are re-drawn once. With
one to ``max_dissent`` dissenting answers and too few of them, the sample is topped up
once to the size that dissent needs (a second look, which is why each look is tested at
``alpha / (max_dissent + 1)``). Every draw depends only on the members, the seed, the
node path and the answers already on record, so a resumed node draws the same ids.

The VLM is asked through the assist core (:func:`~src.services.curation.assist.sample_eval.ask_sample`),
one chunk per call, under the semaphore every node shares. Each chunk is reserved in the
budget when it gets its turn on that semaphore, before anything is sent: the chunk's ids go
on record as selections, the call is made, the answers become proposals and the reservation is
settled at the usage the endpoint reported. The per-item writes are the assist run's own: the
sampled item gets the VLM's class (never a locked one) and an ``assist`` slot whose run id is
``<loop_id>/<path>``, which is how a resume finds an answer that never reached the tables.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.clients.curation_opensearch.crops import mget_crops
from src.domain.loop_stats import homogeneity_lower_bound, required_sample
from src.domain.loop_tree import child_sample
from src.services.curation.assist.proposal import ClassAnswer, compute_proposal
from src.services.curation.assist.sample_eval import ask_sample
from src.services.curation.assist_loop.concurrent import gather_all
from src.services.curation.assist_loop.tree import VLM
from src.services.curation.assist_loop.vlm_failure import VlmFailedError
from src.services.labeling.vlm_usage import observe_usage, total_usage
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

    from opensearchpy import AsyncOpenSearch

    from src.domain.vlm_usage import VlmUsage
    from src.services.curation.assist.slot_write import RunMeta
    from src.services.curation.assist_loop.budget import LoopBudget, Reservation
    from src.services.curation.assist_loop.record import LoopRecord
    from src.services.curation.assist_loop.tree import NodeRecord
    from src.services.curation.vlm_class_batch import VlmContext

ASK_ACTIONS = ('classify',)
VETO_MIN_TRUTH = 3
_RECOVERED = 'recovered'
_STAGE_TAG = {'node_draw': '1', 'node_topup': '2'}
MIN_CONSECUTIVE_FAILURES = 5
"""Failed crops in a row, with no reply in between, that count as an outage rather than a
flaky request. A node whose own crops all failed is only stopped on when the VLM has been
silent this long, so one lost request in a tiny node is still just a re-drawn failure."""
"""Ground-truth members a node needs before they can veto its class."""


# ---------------------------------------------------------------------- the gate


@dataclass(frozen=True)
class GateParams:
    theta: float
    alpha: float
    max_dissent: int

    @property
    def alpha_stage(self) -> float:
        """The level each look is tested at (Bonferroni over the dissent stages)."""
        return self.alpha / (self.max_dissent + 1)


@dataclass(frozen=True)
class Verdict:
    passed: bool
    class_name: str | None
    """The top voted class (registry or not)."""
    agree: int
    answered: int
    dissent: int
    need: int
    """Answers the bound needs at this dissent."""
    bound: float
    """Exact lower bound on the node's agreement with ``class_name``."""
    vetoed: bool
    may_top_up: bool
    """Too few answers, but a second stage could still pass."""
    distribution: list[tuple[str, int]]
    new_class_proposals: list[tuple[str, int]]


def _vetoed(top: str | None, truth: Sequence[str], theta: float) -> bool:
    if top is None or len(truth) < VETO_MIN_TRUTH:
        return False
    wanted = normalize_class_name(top)
    carrying = sum(1 for name in truth if normalize_class_name(name) == wanted)
    return carrying < theta * len(truth)


def judge(
    answers: Iterable[ClassAnswer],
    *,
    n_eligible: int,
    gate: GateParams,
    registry: frozenset[str],
    truth: Sequence[str],
) -> Verdict:
    """The gate over a node's answers. ``truth`` are the classes its ground-truth members
    carry (a person's or a validated import's)."""
    proposal = compute_proposal(list(answers), min_agreement=0.0, registry_names=registry)
    distribution = [(v.class_name, v.votes) for v in proposal.distribution]
    top, agree = distribution[0] if distribution else (None, 0)
    answered = proposal.answered
    dissent = answered - agree
    if n_eligible < 1:
        need, bound = 0, 0.0
    else:
        need = required_sample(n_eligible, gate.theta, gate.alpha_stage, dissent)
        bound = homogeneity_lower_bound(
            agree, min(answered, n_eligible), n_eligible, gate.alpha_stage
        )
    vetoed = _vetoed(top, truth, gate.theta)
    eligible_class = top is not None and top in registry and not vetoed
    within = dissent <= gate.max_dissent
    enough = n_eligible > 0 and answered >= need
    return Verdict(
        passed=eligible_class and within and enough,
        class_name=top,
        agree=agree,
        answered=answered,
        dissent=dissent,
        need=need,
        bound=round(bound, 4),
        vetoed=vetoed,
        may_top_up=eligible_class and within and not enough and gate.max_dissent >= 1,
        distribution=distribution,
        new_class_proposals=[(p.name, p.votes) for p in proposal.new_class_proposals],
    )


# ---------------------------------------------------------------------- asking


def slot_run_id(loop_id: str, path: str) -> str:
    """The ``assist.run_id`` the loop writes on the items it asks about in ``path``."""
    return f'{loop_id}/{path}'


@dataclass(frozen=True)
class Asked:
    answers: dict[str, ClassAnswer]
    """The usable answers by crop id; a crop missing from it got none."""
    request_failed: frozenset[str]
    """Crops whose call failed (no reply at all), not crops the VLM answered with nothing usable."""
    reservation: Reservation
    usage: VlmUsage
    """What the endpoint reported for the call (``NO_USAGE`` when it reported none)."""


class VlmAsker:
    """Asks the loop's VLM about crops through the assist core."""

    def __init__(
        self,
        opensearch: AsyncOpenSearch,
        ctx: VlmContext,
        meta: RunMeta,
        *,
        images_per_call: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self._os = opensearch
        self._ctx = ctx
        self._meta = meta
        self.images_per_call = max(1, images_per_call)
        self._semaphore = semaphore
        self.silent_crops = 0
        """Crops in a row (across nodes) whose call failed, with no reply since."""

    @property
    def registry(self) -> frozenset[str]:
        return frozenset(self._ctx.class_names)

    async def ask(
        self, path: str, chunk: list[str], reserve: Callable[[], Awaitable[Reservation | None]]
    ) -> Asked | None:
        """Ask about one chunk; ``None`` when ``reserve`` (awaited once the chunk has its turn
        on the shared semaphore, before anything is sent) refused it."""
        async with self._semaphore:
            reservation = await reserve()
            if reservation is None:
                return None
            with observe_usage() as seen:
                out = await ask_sample(
                    self._os,
                    self._ctx,
                    chunk,
                    actions=ASK_ACTIONS,
                    box_ids=None,
                    images_per_call=len(chunk),
                    meta=dataclasses.replace(
                        self._meta, run_id=slot_run_id(self._meta.run_id, path)
                    ),
                    semaphore=asyncio.Semaphore(1),
                )
        if out.request_failed_ids and out.predicted == len(out.request_failed_ids):
            self.silent_crops += len(out.request_failed_ids)
        elif out.predicted:
            self.silent_crops = 0
        return Asked(
            answers={
                a.crop_id: a for a in out.answers if a.crop_id and (a.class_name or a.proposed)
            },
            request_failed=frozenset(out.request_failed_ids),
            reservation=reservation,
            usage=total_usage(seen),
        )

    async def recover(self, path: str, ids: Sequence[str]) -> dict[str, ClassAnswer | None]:
        """Answers the items carry from an earlier ask of this node whose log line was
        lost: ``{crop_id: answer}``, ``None`` for an item asked with no usable answer.
        Items with no trace of the ask are absent (they are asked again)."""
        if not ids:
            return {}
        run_id = slot_run_id(self._meta.run_id, path)
        docs = await mget_crops(self._os, list(ids), source_includes=['assist'])
        out: dict[str, ClassAnswer | None] = {}
        for crop_id, doc in docs.items():
            slot = (doc.get('_source') or {}).get('assist') or {}
            if slot.get('run_id') != run_id:
                continue
            name = slot.get('class_name')
            if not name:
                out[crop_id] = None
            elif name in self.registry:
                out[crop_id] = ClassAnswer(class_name=name)
            else:
                out[crop_id] = ClassAnswer(class_name=None, proposed=name)
        return out


# ---------------------------------------------------------------------- one node


@dataclass
class EvalContext:
    record: LoopRecord
    budget: LoopBudget
    asker: VlmAsker
    gate: GateParams
    seed: int
    should_stop: Callable[[], bool]
    """Cancel or pause: no new reservation, the node is left open."""


@dataclass(frozen=True)
class NodeInput:
    record: NodeRecord
    """The node's live record (answers logged so far)."""
    eligible: list[str]
    truth: list[str]
    reused: dict[str, ClassAnswer]
    """Answers ancestors got for this node's eligible members."""
    known_failed: frozenset[str]
    """Members an ancestor asked without a usable answer: never drawn again."""


@dataclass(frozen=True)
class Evaluation:
    verdict: Verdict | None
    """``None`` when the node was stopped or could not be afforded."""
    unvisited: str | None = None
    """The budget that could not cover the draw."""
    sampled: int = 0
    """Members asked or reused in the node (answered or not)."""
    answers: dict[str, ClassAnswer] = dataclasses.field(default_factory=dict)


class _Stop(Exception):  # noqa: N818  # a control-flow signal, not an error
    def __init__(self, unvisited: str | None) -> None:
        super().__init__(unvisited or 'stopped')
        self.unvisited = unvisited


class _Node:
    def __init__(self, ec: EvalContext, node: NodeInput) -> None:
        self.ec = ec
        self.node = node
        self.path = node.record.path
        self.drawable = [i for i in node.eligible if i not in node.known_failed]
        self.request_failed: set[str] = set()
        """Crops whose call failed in this run of the node (not logged as ``failed``)."""

    def answers(self) -> dict[str, ClassAnswer]:
        return {**self.node.reused, **self.node.record.answers}

    def verdict(self) -> Verdict:
        return judge(
            self.answers().values(),
            n_eligible=len(self.node.eligible),
            gate=self.ec.gate,
            registry=self.ec.asker.registry,
            truth=self.node.truth,
        )

    def draw(self, before: list[str], count: int, tag: str) -> list[str]:
        """``count`` more ids, none of ``before``, seeded by the node path and ``tag``."""
        if count < 1:
            return []
        return child_sample(
            before, self.drawable, len(before) + count, seed=self.ec.seed, path=self.path + tag
        ).draw

    async def recover(self) -> None:
        """Answers an earlier ask of this node left on the items but not in the tables."""
        record = self.node.record
        if not record.pending:
            return
        found = await self.ec.asker.recover(self.path, sorted(record.pending))
        got = {i: a for i, a in found.items() if a is not None}
        gone = [i for i, a in found.items() if a is None]
        if got:
            await self.ec.record.record_answers(
                record, uuid.uuid4(), sorted(got), got, proposer_ref=_RECOVERED
            )
        if gone:
            await self.ec.record.record_answers(
                record, uuid.uuid4(), gone, {}, proposer_ref=_RECOVERED
            )

    async def _chunk(self, chunk: list[str], purpose: str) -> bool:
        record = self.node.record

        async def reserve() -> Reservation | None:
            if self.ec.should_stop():
                return None
            res = await self.ec.budget.reserve(VLM, self.path, chunk)
            if res is not None:
                await self.ec.record.select(record, purpose, chunk)
            return res

        asked = await self.ec.asker.ask(self.path, chunk, reserve)
        if asked is None:
            return False
        self.request_failed |= asked.request_failed
        await self.ec.record.record_answers(
            record,
            asked.reservation.request_id,
            chunk,
            asked.answers,
            request_failed=asked.request_failed,
        )
        await self.ec.budget.settle(asked.reservation, asked.usage)
        return True

    async def ask(self, ids: list[str], purpose: str) -> None:
        record = self.node.record
        missing = [i for i in ids if i not in record.answers and i not in record.failed]
        if not missing:
            return
        size = self.ec.asker.images_per_call
        chunks = [missing[i : i + size] for i in range(0, len(missing), size)]
        reason = self.ec.budget.would_refuse(VLM, crops=len(missing), requests=len(chunks))
        if reason is not None:
            raise _Stop(reason)
        done = await gather_all([self._chunk(c, purpose) for c in chunks])
        if not all(done):
            raise _Stop(None if self.ec.should_stop() else self.ec.budget.refused)
        await self._check_outage(missing)

    async def _check_outage(self, asked: list[str]) -> None:
        """Raise when this ask got no answer and every miss is a failed call: the VLM said
        nothing, so the gate has no evidence to decide on. Misses that are answers with no
        usable class (or unreadable crops) mean the VLM is up and keep the gate's say."""
        answers = self.node.record.answers
        misses = [i for i in asked if i not in answers]
        if len(misses) < len(asked) or not misses or not self.request_failed.issuperset(misses):
            return
        if self.ec.asker.silent_crops < MIN_CONSECUTIVE_FAILURES:
            return
        await self.ec.record.forget_failed(self.node.record, sorted(misses))
        msg = (
            f'every VLM call for node {self.path} failed ({len(misses)} crops): '
            'check the endpoint and the model; the node is left open for a resume'
        )
        raise VlmFailedError(msg)

    async def stage(self, before: list[str], ids: list[str], purpose: str) -> list[str]:
        """Ask ``ids`` and re-draw their failures once; every id the stage asked."""
        await self.ask(ids, purpose)
        failed = sum(1 for i in ids if i in self.node.record.failed or i in self.request_failed)
        again = self.draw([*before, *ids], failed, f'#redraw{_STAGE_TAG[purpose]}')
        await self.ask(again, 'redraw')
        return [*ids, *again]

    async def run(self) -> Verdict:
        await self.recover()
        reused = sorted(self.node.reused)
        n = len(self.node.eligible)
        k1 = required_sample(n, self.ec.gate.theta, self.ec.gate.alpha_stage, 0) if n else 0
        first = child_sample(
            reused, self.drawable, max(k1, 1), seed=self.ec.seed, path=self.path
        ).draw
        asked = [*reused, *await self.stage(reused, first, 'node_draw')]
        verdict = self.verdict()
        if verdict.may_top_up:
            more = self.draw(asked, verdict.need - verdict.answered, '#topup')
            if more:
                await self.stage(asked, more, 'node_topup')
                verdict = self.verdict()
        return verdict


async def evaluate_node(ec: EvalContext, node: NodeInput) -> Evaluation:
    """Draw, ask and judge one node. A node with no eligible member asks nothing."""
    worker = _Node(ec, node)
    if not node.eligible:
        return Evaluation(verdict=worker.verdict())
    try:
        verdict = await worker.run()
    except _Stop as stop:
        return Evaluation(verdict=None, unvisited=stop.unvisited)
    record = node.record
    sampled = set(worker.answers()) | ((record.failed | worker.request_failed) & set(node.eligible))
    return Evaluation(verdict=verdict, sampled=len(sampled), answers=worker.answers())


__all__ = [
    'EvalContext',
    'Evaluation',
    'GateParams',
    'NodeInput',
    'Verdict',
    'VlmAsker',
    'evaluate_node',
    'judge',
    'slot_run_id',
]
