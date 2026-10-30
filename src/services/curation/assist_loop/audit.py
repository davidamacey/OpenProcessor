"""Second-opinion audit of the labels an auto-label loop propagated.

After every ``batch_items`` propagated items, and once more at the end, a seeded sample of
the batch's propagated items (``ceil(rate x written)``, at least ``min_per_batch``,
spread over the batch's nodes in proportion to what each wrote) is shown to a second
model with the same pack's classify call, through the labeler like any VLM request. Its
class is compared with the propagated one by name. The answers are recorded in
``audit.jsonl`` and the ``audit`` event only: an audit never writes a label, so an
external judge cannot change data.

The run's agreement so far and its one-sided Wilson lower bound are reported; once at
least ``min_per_batch`` answers are in and the bound is under ``floor``, the batch fails
and the driver pauses the loop (``on_fail: rollback`` also restores the batch). Audit
requests draw on their own budget pool; when it runs out the audit stops and says so.
A remote audit endpoint goes through the same external-images consent as any run:
:func:`resolve_audit_vlm` refuses it without the deployment policy, the endpoint's
``allow_external`` and the request's acknowledgement.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from src.clients.curation_opensearch.crops import mget_crops
from src.domain.loop_stats import wilson_lower_bound
from src.services.curation.assist_loop.budget import AUDIT
from src.services.curation.autolabel.run_vlm import resolve_run_vlm
from src.services.curation.vlm_class_attempt import NEW_CLASS_SENTINEL
from src.services.curation.vlm_class_batch import chunk_crops
from src.services.labeling.detector_hint import DETECTOR_HINT_FIELDS
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from opensearchpy import AsyncOpenSearch

    from src.services.curation.assist_loop.budget import LoopBudget
    from src.services.curation.assist_loop.store import LoopStore, LoopTree
    from src.services.curation.autolabel.run_vlm import RunVlm
    from src.services.curation.vlm_class_batch import VlmContext

AUDIT_FILE = 'audit.jsonl'
WILSON_Z = 1.645
"""One-sided 95 %."""
MIN_AUDITED_TO_PAUSE = 30
"""Answers the run's audit needs before its bound may pause the loop."""


@dataclass(frozen=True)
class AuditParams:
    rate: float = 0.01
    min_per_batch: int = 30
    floor: float = 0.85
    on_fail: Literal['pause', 'rollback'] = 'pause'
    batch_items: int = 2000


async def resolve_audit_vlm(
    opensearch: AsyncOpenSearch, vlm: str | None, *, pack: Any, acknowledge_external: bool
) -> RunVlm | None:
    """Pin the audit endpoint through the run gate; ``None`` (no endpoint named) is an
    audit that is off."""
    if vlm is None:
        return None
    return await resolve_run_vlm(
        opensearch, vlm, pack=pack, acknowledge_external=acknowledge_external
    )


class AuditJudge:
    """Asks the audit model for a class; reads items, writes nothing."""

    def __init__(
        self,
        opensearch: AsyncOpenSearch,
        ctx: VlmContext,
        *,
        images_per_call: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self._os = opensearch
        self._ctx = ctx
        self.images_per_call = max(1, images_per_call)
        self._semaphore = semaphore

    def _name(self, pred: Any) -> str | None:
        if pred.failure is not None or not pred.class_name.strip():
            return None
        raw = pred.proposed_class if pred.class_name == NEW_CLASS_SENTINEL else pred.class_name
        resolved = self._ctx.resolve_class_name(raw, confidence=pred.confidence)
        return resolved or raw.strip().lower()

    async def judge(
        self, chunk: list[str], admit: Callable[[], bool]
    ) -> dict[str, str | None] | None:
        """``{crop_id: class or None}`` for one chunk; ``None`` when ``admit`` refused it
        (called once the chunk has its turn, before anything is sent)."""
        async with self._semaphore:
            if not admit():
                return None
            docs = await mget_crops(
                self._os,
                chunk,
                source_includes=['image_path', 'bbox_norm', 'test_holdout', *DETECTOR_HINT_FIELDS],
            )
            crops = chunk_crops(docs, self._ctx)
            preds = (
                await self._ctx.labeler.label_or_propose_batch(
                    crops, self._ctx.class_names, class_catalog=self._ctx.class_catalog
                )
                if crops
                else []
            )
        out: dict[str, str | None] = dict.fromkeys(chunk)
        for pred in preds:
            out[pred.img_id] = self._name(pred)
        return out


@dataclass(frozen=True)
class AuditResult:
    failing: bool
    nodes: list[str] = field(default_factory=list)
    """The nodes the batch covered (what ``on_fail: rollback`` restores)."""


def _allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` over nodes in proportion to ``sizes`` (largest remainder)."""
    whole = sum(sizes.values())
    if whole == 0:
        return dict.fromkeys(sizes, 0)
    exact = {p: total * n / whole for p, n in sizes.items()}
    out = {p: math.floor(v) for p, v in exact.items()}
    left = total - sum(out.values())
    for path in sorted(exact, key=lambda p: (-(exact[p] - out[p]), p))[:left]:
        out[path] += 1
    return out


class Auditor:
    """The audit of one loop, resumed from its event log."""

    def __init__(
        self,
        store: LoopStore,
        budget: LoopBudget,
        judge: AuditJudge,
        params: AuditParams,
        *,
        seed: int,
    ) -> None:
        self._store = store
        self._budget = budget
        self._judge = judge
        self.params = params
        self._seed = seed

    # ------------------------------------------------------------------ state

    def _batch(self, tree: LoopTree) -> list[tuple[str, str, list[str]]]:
        """Propagated nodes no audit covered yet: ``(path, class, written ids)``."""
        covered = {p for event in tree.audits for p in event.get('nodes') or []}
        return [
            (path, str(node.propagated['class_name']), list(node.propagated['written_ids']))
            for path, node in sorted(tree.nodes.items())
            if node.propagated is not None and path not in covered
        ]

    def pending(self, tree: LoopTree) -> bool:
        return any(ids for _, _, ids in self._batch(tree))

    def due(self, tree: LoopTree) -> bool:
        return sum(len(ids) for _, _, ids in self._batch(tree)) >= self.params.batch_items

    def exhausted(self, tree: LoopTree) -> bool:
        return any(e.get('status') == 'budget_exhausted' for e in tree.audits)

    def summary(self, tree: LoopTree) -> dict[str, Any]:
        last = tree.audits[-1] if tree.audits else {}
        sampled = int(last.get('total_sampled', 0))
        agreed = int(last.get('total_agreed', 0))
        spent = self._budget.snapshot().get(AUDIT, {})
        return {
            'sampled': sampled,
            'agreed': agreed,
            'agreement': round(agreed / sampled, 4) if sampled else None,
            'lower_bound': round(wilson_lower_bound(agreed, sampled, WILSON_Z), 4),
            'crops_spent': spent.get('crops_spent', 0) + spent.get('crops_reserved', 0),
            'status': last.get('status', 'pending'),
        }

    # ------------------------------------------------------------------ one batch

    def _sample(
        self, batch: list[tuple[str, str, list[str]]], number: int
    ) -> list[tuple[str, str, str]]:
        written = sum(len(ids) for _, _, ids in batch)
        want = min(written, max(self.params.min_per_batch, math.ceil(self.params.rate * written)))
        quota = _allocate({path: len(ids) for path, _, ids in batch}, want)
        picked: list[tuple[str, str, str]] = []
        for path, class_name, ids in batch:
            rng = random.Random(f'{self._seed}:audit:{number}:{path}')
            drawn = sorted(rng.sample(sorted(ids), quota[path]))
            picked.extend((crop_id, path, class_name) for crop_id in drawn)
        return picked

    async def _ask(self, picked: list[tuple[str, str, str]]) -> tuple[dict[str, str | None], bool]:
        ids = [crop_id for crop_id, _, _ in picked]
        size = self._judge.images_per_call
        chunks = [ids[i : i + size] for i in range(0, len(ids), size)]
        refused = False

        async def one(chunk: list[str]) -> dict[str, str | None]:
            nonlocal refused
            held = []

            def admit() -> bool:
                res = self._budget.reserve(AUDIT, 'audit', chunk)
                if res is not None:
                    held.append(res)
                return res is not None

            got = await self._judge.judge(chunk, admit)
            if got is None:
                refused = True
                return {}
            self._budget.settle(held[0])
            return got

        answers: dict[str, str | None] = {}
        for part in await asyncio.gather(*[one(c) for c in chunks]):
            answers.update(part)
        return answers, refused

    async def run_batch(self, tree: LoopTree) -> AuditResult:
        """Audit every propagated node not audited yet; log and return the outcome."""
        batch = self._batch(tree)
        number = len(tree.audits) + 1
        picked = self._sample(batch, number)
        answers, refused = await self._ask(picked)
        rows = [
            {
                'batch': number,
                'crop_id': crop_id,
                'node': path,
                'propagated_class': class_name,
                'audit_answer': answers[crop_id],
                'agree': normalize_class_name(answers[crop_id] or '')
                == normalize_class_name(class_name),
            }
            for crop_id, path, class_name in picked
            if answers.get(crop_id) is not None
        ]
        self._write_rows(rows)
        last = tree.audits[-1] if tree.audits else {}
        total_sampled = int(last.get('total_sampled', 0)) + len(rows)
        total_agreed = int(last.get('total_agreed', 0)) + sum(1 for r in rows if r['agree'])
        lower = wilson_lower_bound(total_agreed, total_sampled, WILSON_Z)
        failing = total_sampled >= MIN_AUDITED_TO_PAUSE and lower < self.params.floor
        status = 'fail' if failing else 'budget_exhausted' if refused else 'ok'
        nodes = [path for path, _, _ in batch]
        self._store.append(
            'audit',
            batch=number,
            nodes=nodes,
            sampled=len(rows),
            agreed=sum(1 for r in rows if r['agree']),
            total_sampled=total_sampled,
            total_agreed=total_agreed,
            lower_bound=round(lower, 4),
            status=status,
        )
        return AuditResult(failing=failing, nodes=nodes)

    def _write_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        target = self._store.directory / AUDIT_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('a', encoding='utf-8') as fh:
            fh.write(''.join(json.dumps(r) + '\n' for r in rows))


__all__ = [
    'AUDIT_FILE',
    'AuditJudge',
    'AuditParams',
    'AuditResult',
    'Auditor',
    'resolve_audit_vlm',
]
