"""Resolve an assist run's target to members, then to the sample the VLM sees.

A cluster or sub-cluster target is a filter on stored fields (``cluster_id``,
``cluster_subid``); nothing here depends on how items were clustered. The
run records the resolved member ids and a digest, so staleness is measured as
membership overlap (:func:`membership_overlap`), never by trusting a sub-id
string across a re-refine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import items_index
from src.domain.assist_sampling import MAX_ASSIST_ITEMS, representative_size, stratified_sample
from src.services.curation.assist.models import (
    AssistRunRequest,
    AssistTarget,
    BoxesTarget,
    ClusterTarget,
    CropsTarget,
    FilterTarget,
    SkippedCounts,
    StratumInfo,
)
from src.services.curation.class_write_guard import class_write_locked
from src.services.curation.item_filter import ItemFilter, item_filter_clauses, visibility_clauses
from src.services.curation.item_selection import MAX_SELECTION_SCAN
from src.services.curation.reprocess_targets import items_by_terms, scan_items
from src.services.rejections import rejection


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

_INCLUDES = ['test_holdout', 'class_excluded', 'cluster_subid', 'class_source', 'class_validated']


@dataclass
class ScopeResolution:
    members: list[str]
    """Visible members of the target, in a stable order."""
    digest: str
    skipped: SkippedCounts
    sampled_ids: list[str]
    strata: list[StratumInfo]
    seed: int
    box_ids: dict[str, list[str]] | None = None
    """For a boxes target: the valid named box ids per item."""
    class_locked_ids: set[str] = field(default_factory=set)
    """Sampled members whose class an automated writer may not touch."""


def membership_digest(ids: list[str]) -> str:
    """``sha256:`` over the sorted ids: equal members, equal digest."""
    return 'sha256:' + hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def membership_overlap(before: list[str], now: list[str]) -> dict[str, Any]:
    """How much two member lists share: ``{count, overlap, jaccard, changed}``.
    ``changed`` is any difference at all."""
    a, b = set(before), set(now)
    union = len(a | b)
    shared = len(a & b)
    return {
        'count': len(b),
        'overlap': shared,
        'jaccard': (shared / union) if union else 1.0,
        'changed': a != b,
    }


def scope_filter(target: ClusterTarget | FilterTarget) -> ItemFilter:
    if isinstance(target, ClusterTarget):
        return ItemFilter(cluster_id=target.cluster_id, cluster_subid=target.cluster_subid)
    return target.filter


def proposal_scope(target: AssistTarget) -> ItemFilter | None:
    """The (sub-)cluster a run's proposal is about: the cluster target itself, or the
    context a selection of crops was made in; ``None`` for a bare selection or a filter."""
    if isinstance(target, ClusterTarget):
        return scope_filter(target)
    if isinstance(target, CropsTarget) and target.context is not None:
        return ItemFilter(
            cluster_id=target.context.cluster_id, cluster_subid=target.context.cluster_subid
        )
    return None


def _base_query(flt: ItemFilter, *extra: dict[str, Any]) -> dict[str, Any]:
    try:
        clauses = [*item_filter_clauses(flt), *extra]
    except ValueError as exc:
        raise rejection(422, 'invalid_request', str(exc)) from exc
    return {'bool': {'filter': clauses}} if clauses else {'match_all': {}}


async def _count(opensearch: AsyncOpenSearch, query: dict[str, Any]) -> int:
    resp = await opensearch.count(index=items_index(), body={'query': query})
    return int(resp.get('count', 0))


async def filter_members(
    opensearch: AsyncOpenSearch, flt: ItemFilter, *, label: str
) -> tuple[list[tuple[str, dict[str, Any]]], SkippedCounts]:
    total = await _count(opensearch, _base_query(flt))
    if total == 0:
        raise rejection(404, 'not_found', f'{label} selects no items')
    holdout = await _count(opensearch, _base_query(flt, {'term': {'test_holdout': True}}))
    excluded = await _count(
        opensearch,
        _base_query(
            flt,
            {'term': {'class_excluded': True}},
            {'bool': {'must_not': {'term': {'test_holdout': True}}}},
        ),
    )
    visible = _base_query(flt, *visibility_clauses(include_test=False, include_excluded=False))
    found = await scan_items(
        opensearch,
        visible,
        index=items_index(),
        includes=_INCLUDES,
        max_docs=MAX_SELECTION_SCAN + 1,
    )
    if len(found) > MAX_SELECTION_SCAN:
        raise rejection(
            422, 'assist_too_large', f'{label} matches more than {MAX_SELECTION_SCAN} items'
        )
    return found, SkippedCounts(holdout=holdout, excluded=excluded)


async def _crop_members(
    opensearch: AsyncOpenSearch, target: CropsTarget
) -> tuple[list[tuple[str, dict[str, Any]]], SkippedCounts]:
    wanted = list(dict.fromkeys(target.crop_ids))
    found = await items_by_terms(
        opensearch, 'crop_id', wanted, index=items_index(), includes=_INCLUDES
    )
    by_id = dict(found)
    ordered = [(cid, by_id[cid]) for cid in wanted if cid in by_id]
    holdout = sum(1 for _, s in ordered if s.get('test_holdout'))
    excluded = sum(1 for _, s in ordered if s.get('class_excluded') and not s.get('test_holdout'))
    visible = [
        (cid, s) for cid, s in ordered if not s.get('test_holdout') and not s.get('class_excluded')
    ]
    return visible, SkippedCounts(
        holdout=holdout, excluded=excluded, not_found=len(wanted) - len(ordered)
    )


async def _box_members(
    opensearch: AsyncOpenSearch, target: BoxesTarget
) -> tuple[list[tuple[str, dict[str, Any]]], SkippedCounts, dict[str, list[str]]]:
    from src.clients.occ_locks import is_locked_box
    from src.config.region_fields import get_region_fields
    from src.domain.region_boxes import read_boxes

    F = get_region_fields()
    wanted: dict[str, list[str]] = {}
    for ref in target.boxes:
        ids = wanted.setdefault(ref.crop_id, [])
        if ref.box_id not in ids:
            ids.append(ref.box_id)
    found = await items_by_terms(
        opensearch, 'crop_id', list(wanted), index=items_index(), includes=[*_INCLUDES, F.boxes]
    )
    by_id = dict(found)
    holdout = excluded = box_not_found = box_locked = 0
    members: list[tuple[str, dict[str, Any]]] = []
    valid: dict[str, list[str]] = {}
    for cid, named in wanted.items():
        source = by_id.get(cid)
        if source is None:
            continue
        if source.get('test_holdout'):
            holdout += 1
            continue
        if source.get('class_excluded'):
            excluded += 1
            continue
        stored = {b.box_id: b for b in read_boxes(source, F)}
        valid[cid] = [b for b in named if b in stored]
        box_not_found += len(named) - len(valid[cid])
        box_locked += sum(1 for b in valid[cid] if is_locked_box(stored[b]))
        members.append((cid, source))
    skipped = SkippedCounts(
        not_found=len(wanted) - len(by_id),
        holdout=holdout,
        excluded=excluded,
        box_not_found=box_not_found,
        box_locked=box_locked,
    )
    return members, skipped, valid


def _stratum(source: dict[str, Any]) -> str:
    return str(source.get('cluster_subid') or '')


def _sample(
    request: AssistRunRequest, members: list[tuple[str, dict[str, Any]]]
) -> tuple[list[str], list[StratumInfo]]:
    sample = request.sample
    ids = [cid for cid, _ in members]
    if sample.mode == 'all':
        if len(ids) > MAX_ASSIST_ITEMS:
            raise rejection(
                422,
                'assist_too_large',
                f'{len(ids)} items exceed the {MAX_ASSIST_ITEMS} an assist run sends to the VLM; '
                'use a representative sample, a narrower scope, or the auto-label job',
                limit=MAX_ASSIST_ITEMS,
            )
        return ids, []
    k = representative_size(len(ids), size=sample.size)
    by_stratum: dict[str, list[str]] = {}
    for cid, source in members:
        by_stratum.setdefault(_stratum(source), []).append(cid)
    stratify = sample.stratify_by == 'cluster_subid' or (
        sample.stratify_by == 'auto' and len(by_stratum) > 1
    )
    if not stratify:
        return stratified_sample({'': ids}, k, seed=sample.seed), []
    picked = stratified_sample(by_stratum, k, seed=sample.seed)
    picked_set = set(picked)
    strata = [
        StratumInfo(key=key, size=len(group), sampled=sum(1 for i in group if i in picked_set))
        for key, group in sorted(by_stratum.items())
    ]
    return picked, strata


async def resolve_scope(opensearch: AsyncOpenSearch, request: AssistRunRequest) -> ScopeResolution:
    """The members, skipped counts, digest and sample of ``request``'s target.
    Raises a structured refusal: 404 ``not_found`` (an empty cluster), 422
    ``assist_too_large`` or ``invalid_request``."""
    target = request.target
    box_ids: dict[str, list[str]] | None = None
    if isinstance(target, BoxesTarget):
        members, skipped, box_ids = await _box_members(opensearch, target)
    elif isinstance(target, CropsTarget):
        members, skipped = await _crop_members(opensearch, target)
    else:
        label = (
            f'cluster {target.cluster_id}'
            + (f' sub-cluster {target.cluster_subid}' if target.cluster_subid else '')
            if isinstance(target, ClusterTarget)
            else 'the filter'
        )
        members, skipped = await filter_members(opensearch, scope_filter(target), label=label)
    sampled, strata = _sample(request, members)
    sampled_set = set(sampled)
    locked = {cid for cid, s in members if cid in sampled_set and class_write_locked(s)}
    skipped.class_locked = len(locked)
    ids = [cid for cid, _ in members]
    return ScopeResolution(
        members=ids,
        digest=membership_digest(ids),
        skipped=skipped,
        sampled_ids=sampled,
        strata=strata,
        seed=request.sample.seed,
        box_ids=box_ids,
        class_locked_ids=locked,
    )


__all__ = [
    'ScopeResolution',
    'filter_members',
    'membership_digest',
    'membership_overlap',
    'proposal_scope',
    'resolve_scope',
    'scope_filter',
]
