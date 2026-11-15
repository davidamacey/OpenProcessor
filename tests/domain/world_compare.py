"""Run one act sequence through today's document and through the claim fold, and diff every field
the wire serializer reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.clients import occ_locks
from src.config import get_region_fields
from src.domain.claim_kinds import ActorKind
from src.domain.claim_layers import split_claims
from src.domain.fold_locks import box_is_locked, has_import_geometry, is_locked_class
from tests.domain.doc_world import DocWorld, RefusedError
from tests.domain.fold_world import FoldWorld
from tests.domain.world_acts import Act, random_sequence


def doc_predicates(world: DocWorld) -> dict[str, Any]:
    """Today's lock predicates (``src/clients/occ_locks.py``) on the document."""
    doc = world.doc
    boxes = world.boxes()
    return {
        'p_is_locked_class': occ_locks.is_locked_class(doc),
        'p_is_human_deleted': occ_locks.is_human_deleted(doc),
        'p_has_human_geometry': occ_locks.has_human_geometry(doc),
        'p_has_import_geometry': occ_locks.has_import_geometry(doc),
        'p_is_locked_geometry': occ_locks.is_locked_geometry(doc),
        'p_is_locked_item': occ_locks.is_locked_item(doc, get_region_fields()),
        'p_is_human_owned_item': occ_locks.is_human_owned_item(doc, get_region_fields()),
        'p_is_locked_box': tuple(occ_locks.is_locked_box(b) for b in boxes),
    }


def fold_predicates(world: FoldWorld) -> dict[str, Any]:
    """The same predicates, read off the folded state and its claims (15a 4.4)."""
    f = world.fold()
    s = f.state
    live = split_claims(world.log.claims).live
    return {
        'p_is_locked_class': is_locked_class(
            class_source=s.class_source, class_validated=s.class_validated, holdout=s.holdout
        ),
        'p_is_human_deleted': s.excluded_by_human and s.excluded_reason == 'not_an_object',
        'p_has_human_geometry': s.geometry_source == 'human',
        'p_has_import_geometry': has_import_geometry(
            import_ids=s.import_ids, class_source=s.class_source, class_validated=s.class_validated
        ),
        'p_is_locked_geometry': s.geometry_locked or s.geometry_source == 'human',
        'p_is_locked_item': s.label_locked,
        'p_is_human_owned_item': any(c.actor_kind is ActorKind.HUMAN for c in live),
        'p_is_locked_box': tuple(
            box_is_locked(
                source=b.source,
                rejection_reason=b.rejection_reason,
                text_source=b.text_source,
                state=b.state,
            )
            for b in f.boxes
        ),
    }


def doc_view(world: DocWorld, *, predicates: bool = False) -> dict[str, Any]:
    w = world.wire()
    boxes = [
        (
            b['box_id'],
            tuple(b['bbox_norm']),
            b['state'],
            b['rejection_reason'],
            b['source'],
            b['text'],
            b['text_source'],
            b['locked'],
        )
        for b in w['region_boxes']
    ]
    return {
        'class_id': w['class_id'],
        'class_name': w['class_name'] or None,
        'class_source': w['class_source'] or None,
        'class_validated': w['class_validated'],
        'label_validated': w['label_validated'],
        'label_source': w['label_source'] or None,
        'class_excluded': w['class_excluded'],
        'excluded_reason': w['excluded_reason'],
        'test_holdout': w['test_holdout'],
        'review_dismissed': bool(w['review_dismissed_at']),
        'needs_new_class': w['needs_new_class'],
        'vlm_suggestion': (w['vlm_proposed_class_id'], w['vlm_proposed_class_name']),
        'bbox_norm': tuple(w['bbox_norm']),
        'geometry_source': w['geometry_source'],
        'geometry_locked': w['geometry_locked'],
        'label_locked': w['label_locked'],
        'region_status': w['region_status'],
        'region_validated': bool(w['region_validated']),
        'region_count': w['region_count'],
        'region_rejected_count': w['region_rejected_count'],
        'region_set_complete': w['region_set_complete'],
        'region_boxes': boxes,
        'import_ids': frozenset(w['import_ids']),
        'combine_conflict': w['combine_conflict'],
        **(doc_predicates(world) if predicates else {}),
    }


def fold_view(world: FoldWorld, *, predicates: bool = False) -> dict[str, Any]:
    f = world.fold()
    s = f.state
    boxes = [
        (
            b.box_id,
            tuple(b.bbox),
            b.state,
            b.rejection_reason,
            b.source,
            b.text,
            b.text_source,
            b.locked,
        )
        for b in f.boxes
    ]
    suggestion = (
        (None, None)
        if s.vlm_suggestion_dismissed
        else (s.vlm_suggestion_class_id, s.vlm_suggestion_name)
    )
    return {
        'class_id': s.class_id,
        'class_name': s.class_name,
        'class_source': s.class_source,
        'class_validated': s.class_validated,
        'label_validated': s.class_validated or s.region_validated,
        'label_source': s.label_source,
        'class_excluded': s.excluded,
        'excluded_reason': s.excluded_reason,
        'test_holdout': s.holdout,
        'review_dismissed': s.review_dismissed_at is not None,
        'needs_new_class': s.needs_new_class,
        'vlm_suggestion': suggestion,
        'bbox_norm': tuple(s.bbox),
        'geometry_source': s.geometry_source,
        'geometry_locked': s.geometry_locked,
        'label_locked': s.label_locked,
        'region_status': s.region_status,
        'region_validated': s.region_validated,
        'region_count': s.region_count,
        'region_rejected_count': s.region_rejected_count,
        'region_set_complete': s.region_set_complete,
        'region_boxes': boxes,
        'import_ids': frozenset(s.import_ids),
        'combine_conflict': s.combine_conflict,
        **(fold_predicates(world) if predicates else {}),
    }


@dataclass
class Ctx:
    """Everything an allow-list entry may look at to explain a difference."""

    acts: list[Act]
    doc: DocWorld
    fold: FoldWorld
    doc_view: dict[str, Any]
    fold_view: dict[str, Any]
    diffs: dict[str, tuple[Any, Any]]


@dataclass
class Divergence:
    seed: int
    step: int
    acts: list[Act]
    field: str
    doc: Any
    fold: Any


@dataclass
class Outcome:
    divergences: list[Divergence] = field(default_factory=list)
    explained: dict[str, int] = field(default_factory=dict)
    steps: int = 0


def diff(doc: dict[str, Any], fold: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    return {k: (doc[k], fold[k]) for k in doc if doc[k] != fold[k]}


def run_seed(seed: int, allowed: list[Any], length: int = 8) -> Outcome:
    classifier, acts = random_sequence(seed, length)
    return run_acts(classifier, acts, allowed, seed=seed)


def run_acts(
    classifier: str | None,
    acts: list[Act],
    allowed: list[Any],
    *,
    seed: int = 0,
    predicates: bool = False,
) -> Outcome:
    doc, fold = DocWorld(classifier), FoldWorld(classifier)
    out = Outcome()
    done: list[Act] = []
    for step, act in enumerate(acts):
        done.append(act)
        try:
            doc.apply(act) if act[0] != 'region_write' else _region_write(doc, fold, act)
        except RefusedError:
            doc_refused = True
        else:
            doc_refused = False
        try:
            if act[0] != 'region_write':
                fold.apply(act)
            fold_refused = False
        except RefusedError:
            fold_refused = True
        if doc_refused != fold_refused:
            ctx = Ctx(
                done,
                doc,
                fold,
                doc_view(doc, predicates=predicates),
                fold_view(fold, predicates=predicates),
                {},
            )
            hit = next((a for a in allowed if a.applies(ctx, 'refusal')), None)
            if hit is None:
                out.divergences.append(
                    Divergence(seed, step, list(done), 'refusal', doc_refused, fold_refused)
                )
            else:
                out.explained[hit.name] = out.explained.get(hit.name, 0) + 1
            return out
        out.steps += 1
        ctx = Ctx(
            done,
            doc,
            fold,
            doc_view(doc, predicates=predicates),
            fold_view(fold, predicates=predicates),
            {},
        )
        if fold.fold().state.class_claim_id is None:
            # label_source of a machine class belongs to the writer, not to the claim model.
            ctx.doc_view['label_source'] = ctx.fold_view['label_source'] = None
        ctx.diffs = diff(ctx.doc_view, ctx.fold_view)
        for name, (d, f) in ctx.diffs.items():
            hit = next((a for a in allowed if a.applies(ctx, name)), None)
            if hit is None:
                out.divergences.append(Divergence(seed, step, list(done), name, d, f))
            else:
                out.explained[hit.name] = out.explained.get(hit.name, 0) + 1
    return out


def _region_write(doc: DocWorld, fold: FoldWorld, act: Act) -> None:
    before = list(doc.landed_machine_sources)
    doc.act_region_write(*act[1:])
    if len(doc.landed_machine_sources) > len(before):
        fold.region_write_with(doc.last_machine_boxes)
    else:
        from src.domain.region_boxes import RegionBox

        boxes = [
            RegionBox(box_id=f'b9{i}', bbox_norm=b, state=act[2], score=0.8, source='segmenter')
            for i, b in enumerate(act[1])
        ]
        fold.region_write_with(boxes, bump_seq=False)
