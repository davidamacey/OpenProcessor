"""One page of source images into the target (projects plan section 6).

Every write is an ``index`` of a whole doc under a deterministic id (target
image id from the target path and hash, crop id from image id and box), so a
chunk redone after a crash rewrites the same docs and a merge into a
duplicate is recomputed to the same result.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.curation import IndexRole
from src.config.project_context import bind_project
from src.services.curation.dataset_import.mapping import norm_class_name
from src.services.curation.dataset_import.project_source import item_class_name
from src.services.curation.ingest_index import image_id_for
from src.services.projects.combine.copy_docs import (
    link_or_copy,
    target_image_path,
    transform_image,
    transform_item,
)
from src.services.projects.combine.dedup import Probe, decide_merges, trust_rank
from src.services.projects.combine.plan import class_target
from src.services.projects.combine.regions import attach_regions


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord
    from src.config.region_fields import RegionFields
    from src.services.curation.dataset_import.project_source import SourceImage
    from src.services.projects.combine.mapping import CombineMapping
    from src.services.projects.combine.models import CombineRequest

_BULK_BATCH = 500


@dataclass
class CopyContext:
    client: Any
    job_id: str
    request: CombineRequest
    sources: list[ProjectRecord]
    target: ProjectRecord
    mapping: CombineMapping
    target_ids: dict[str, int]
    names: list[dict[int, str]]
    fields: RegionFields
    embedding_dim: int
    now: str
    duplicates: dict[tuple[int, str], tuple[int, str]]
    originals: dict[tuple[int, str], tuple[str, str]]
    """``(source index, image id) -> (path, imohash)`` of every duplicate's priority image."""
    containment: float = 0.9
    counts: Counter[str] = field(default_factory=Counter)

    @property
    def items_index(self) -> str:
        return self.target.resources.indexes[IndexRole.ITEMS]

    @property
    def images_index(self) -> str:
        return self.target.resources.indexes[IndexRole.IMAGES]


def _target_location(
    ctx: CopyContext, source_index: int, path: str, imohash: str
) -> tuple[str, str, Path | None]:
    """``(target path, target image id, file to link)``. Without dedup the id
    also names the source, so the same file arriving from two sources stays
    two images instead of overwriting one."""
    source = ctx.sources[source_index]
    tpath, link_src = target_image_path(
        path,
        source_upload_root=source.resources.upload_root,
        target_upload_root=ctx.target.resources.upload_root,
    )
    key = tpath if ctx.request.dedup == 'content_hash' else f'{source.slug}:{tpath}'
    return tpath, image_id_for(key, imohash), link_src


def map_negative_for(names: list[str], ctx: CopyContext, project: str) -> list[str]:
    """A reviewed negative's class names as target class names; a name the
    mapping skipped or turned into a region class does not carry over."""
    per_source = ctx.mapping.per_source.get(project, {})
    by_norm = {norm_class_name(n): n for n in ctx.target_ids}
    out: list[str] = []
    for name in names:
        target = per_source.get(name)
        resolved = (
            target.class_name
            if target and target.kind == 'item'
            else by_norm.get(norm_class_name(name))
        )
        if resolved and resolved not in out and (target is None or target.kind == 'item'):
            out.append(resolved)
    return out


def _holdout_flag(ctx: CopyContext, image: SourceImage) -> bool:
    return ctx.request.holdout == 'preserve_union' and image.is_holdout


def _class_of(
    ctx: CopyContext, source_index: int, item: dict[str, Any]
) -> tuple[str, tuple[int, str] | None]:
    """``(kind, (target id, target name))`` of a source item."""
    name = item_class_name(item, ctx.names[source_index])
    kind, target = class_target(ctx.mapping, ctx.sources[source_index].slug, name)
    if kind == 'item' and target is not None:
        return kind, (ctx.target_ids[target], target)
    return kind, None


def _item_docs(
    ctx: CopyContext, source_index: int, image: SourceImage, tid: str, tpath: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(item docs, region-class source items)`` of one image."""
    docs: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    project = ctx.sources[source_index].slug
    for item in image.items:
        kind, target = _class_of(ctx, source_index, item)
        if kind == 'skip' or len(item.get('bbox_norm') or ()) != 4:
            ctx.counts['items_skipped'] += 1
        elif kind == 'region':
            regions.append(item)
        else:
            doc, dropped = transform_item(
                item,
                target_image_id=tid,
                target_image_path_=tpath,
                target_class=target,
                job_id=ctx.job_id,
                origin_project=project,
                now=ctx.now,
                embedding_dim=ctx.embedding_dim,
                fields=ctx.fields,
            )
            ctx.counts['embeddings_dropped'] += dropped
            docs.append(doc)
    return docs, regions


def _apply_holdout(docs: list[dict[str, Any]], flag: bool, mode: str) -> None:
    for doc in docs:
        if mode == 'preserve_union':
            doc['test_holdout'] = bool(doc.get('test_holdout')) or flag
        else:
            doc['test_holdout'] = False


async def _bulk(ctx: CopyContext, ops: list[dict[str, Any]]) -> None:
    """``ops``: ``{'index': name, 'id': ..., 'doc': ...}`` rows, written to
    the target in batches."""
    with bind_project(ctx.target):
        for start in range(0, len(ops), _BULK_BATCH):
            body: list[dict[str, Any]] = []
            for op in ops[start : start + _BULK_BATCH]:
                body.append({'index': {'_index': op['index'], '_id': op['id']}})
                body.append(op['doc'])
            resp = await ctx.client.bulk(body=body, refresh=False)
            if resp.get('errors'):
                failed = [r for r in resp.get('items', []) if (r.get('index') or {}).get('error')]
                raise RuntimeError(f'target bulk write failed: {failed[:3]}')


async def _target_items(ctx: CopyContext, image_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor: list[Any] | None = None
    with bind_project(ctx.target):
        while True:
            body: dict[str, Any] = {
                'size': 500,
                'query': {'bool': {'filter': [{'term': {'image_id': image_id}}]}},
                'sort': [{'crop_id': 'asc'}],
            }
            if cursor is not None:
                body['search_after'] = cursor
            hits = (
                (await ctx.client.search(index=ctx.items_index, body=body)).get('hits') or {}
            ).get('hits') or []
            out.extend(h.get('_source') or {} for h in hits)
            if len(hits) < 500:
                return out
            cursor = hits[-1].get('sort')


def _origin(doc: dict[str, Any]) -> str:
    return f'{doc.get("origin_project")}:{doc.get("origin_item_id")}'


async def _copy_image(
    ctx: CopyContext, source_index: int, image: SourceImage
) -> list[dict[str, Any]]:
    project = ctx.sources[source_index].slug
    tpath, tid, link_src = _target_location(ctx, source_index, image.path, image.imohash)
    if link_src is not None:
        ctx.counts[f'files_{link_or_copy(link_src, Path(tpath))}'] += 1
    negative_for = (
        map_negative_for(list(image.doc.get('negative_for') or []), ctx, project)
        if image.is_negative
        else None
    )
    ops = [
        {
            'index': ctx.images_index,
            'id': tid,
            'doc': transform_image(
                image.doc,
                target_image_id=tid,
                target_image_path_=tpath,
                job_id=ctx.job_id,
                origin_project=project,
                now=ctx.now,
                embedding_dim=ctx.embedding_dim,
                negative_for=negative_for,
            ),
        }
    ]
    docs, regions = _item_docs(ctx, source_index, image, tid, tpath)
    attached, standalone = attach_regions(
        docs,
        regions,
        image_id=tid,
        image_path=tpath,
        fields=ctx.fields,
        job_id=ctx.job_id,
        origin_project=project,
        containment=ctx.containment,
        now=ctx.now,
    )
    ctx.counts['regions_attached'] += attached
    ctx.counts['regions_standalone'] += len(standalone)
    all_docs = [*docs, *standalone]
    _apply_holdout(all_docs, _holdout_flag(ctx, image), ctx.request.holdout)
    ops.extend({'index': ctx.items_index, 'id': d['crop_id'], 'doc': d} for d in all_docs)
    ctx.counts['images_copied'] += 1
    ctx.counts['items_copied'] += len(all_docs)
    return ops


async def _merge_duplicate(
    ctx: CopyContext, source_index: int, image: SourceImage
) -> list[dict[str, Any]]:
    """Attach a duplicate image's boxes to the priority copy already in the
    target (see :func:`decide_merges`)."""
    pri_index, pri_id = ctx.duplicates[(source_index, image.image_id)]
    path, imohash = ctx.originals[(pri_index, pri_id)]
    tpath, tid, _ = _target_location(ctx, pri_index, path, imohash)
    existing = await _target_items(ctx, tid)
    probes = [
        Probe(tuple(d['bbox_norm']), d.get('class_name'), trust_rank(d))  # type: ignore[arg-type]
        for d in existing
    ]
    docs, _regions = _item_docs(ctx, source_index, image, tid, tpath)
    _apply_holdout(docs, _holdout_flag(ctx, image), ctx.request.holdout)
    changed: dict[str, dict[str, Any]] = {}
    added: list[dict[str, Any]] = []
    incoming = [Probe(tuple(d['bbox_norm']), d.get('class_name'), trust_rank(d)) for d in docs]  # type: ignore[arg-type]
    for decision in decide_merges(probes, incoming, iou_min=ctx.request.dedup_iou):
        new = docs[decision.incoming]
        if decision.kind == 'union':
            added.append(new)
            continue
        base = existing[decision.existing or 0]
        keep = changed.setdefault(base['crop_id'], dict(base))
        if decision.kind == 'conflict':
            keep['combine_conflict'] = True
            keep['combine_conflict_origins'] = sorted(
                {*keep.get('combine_conflict_origins', [_origin(keep)]), _origin(new)}
            )
            ctx.counts['items_conflicts'] += 1
            continue
        merged_from = sorted({*keep.get('combine_merged_origins', [_origin(keep)]), _origin(new)})
        if decision.take_incoming_label:
            flags = {
                k: keep[k] for k in ('combine_conflict', 'combine_conflict_origins') if k in keep
            }
            keep.clear()
            keep.update({**new, 'crop_id': base['crop_id'], 'bbox_norm': base['bbox_norm']})
            keep.update(flags)
        keep['combine_merged_origins'] = merged_from
        ctx.counts['items_merged'] += 1
    flag = _holdout_flag(ctx, image)
    final = [*changed.values(), *added]
    untouched = [d for d in existing if d['crop_id'] not in changed]
    if flag:
        for doc in (*final, *untouched):
            doc['test_holdout'] = True
        final = [*final, *untouched]
    ops = [{'index': ctx.items_index, 'id': d['crop_id'], 'doc': d} for d in final]
    ctx.counts['images_duplicate'] += 1
    ctx.counts['items_copied'] += len(added)
    return ops


async def copy_page(ctx: CopyContext, source_index: int, images: list[SourceImage]) -> Counter[str]:
    """Copy (or merge) one page; returns this page's counters."""
    before = Counter(ctx.counts)
    ops: list[dict[str, Any]] = []
    if any((source_index, i.image_id) in ctx.duplicates for i in images):
        with bind_project(ctx.target):
            await ctx.client.indices.refresh(index=ctx.items_index)
    for image in images:
        if (source_index, image.image_id) in ctx.duplicates:
            ops.extend(await _merge_duplicate(ctx, source_index, image))
        else:
            ops.extend(await _copy_image(ctx, source_index, image))
    await _bulk(ctx, ops)
    return ctx.counts - before


__all__ = ['CopyContext', 'copy_page', 'map_negative_for']
