"""Combine preview and plan (projects plan section 6): validate the request,
read every source through W10's project reader, resolve the name mapping,
find duplicates, and report exactly what a run would do. Writes nothing.

:func:`analyze` is the single computation behind ``POST /projects/combine/
preview`` and the job's persisted plan, so the two cannot disagree.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from src.config.curation import IndexRole
from src.config.project_context import bind_project
from src.config.projects import is_valid_slug
from src.services.curation.dataset_import.project_source import (
    class_names_by_id,
    fetch_items,
    item_class_name,
    iter_source_pages,
)
from src.services.projects.combine.dedup import (
    Fingerprint,
    Probe,
    decide_merges,
    find_duplicates,
    trust_rank,
)
from src.services.projects.combine.mapping import (
    CombineMapping,
    resolve_combine_mapping,
    suggest_combine_mapping,
)
from src.services.projects.combine.models import (
    MAX_SOURCES,
    CombineIssue,
    CombinePreview,
    CombineRequest,
)
from src.services.projects.combine.warnings import embedding_model_mismatch, region_profiles_differ


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord

_SOURCE_STATUSES = frozenset({'active', 'archived'})
_CONFLICT_SAMPLES = 20


@dataclass
class SourceStats:
    project: str
    images: int = 0
    items: int = 0
    labeled_items: int = 0
    holdout_images: int = 0
    classes: dict[str, int] = field(default_factory=dict)


@dataclass
class Analysis:
    request: CombineRequest
    records: list[ProjectRecord]
    mapping: CombineMapping
    stats: list[SourceStats]
    fingerprints: list[Fingerprint]
    duplicates: dict[tuple[int, str], tuple[int, str]]
    holdout_images: int
    merged_items: int
    conflicts: int
    conflict_samples: list[dict[str, Any]]
    to_link_bytes: int
    to_copy_bytes: int
    errors: list[CombineIssue]
    warnings: list[CombineIssue]


# --------------------------------------------------------------- environment


async def resolve_sources(
    request: CombineRequest, *, ignore_job: str | None = None
) -> tuple[list[ProjectRecord], list[CombineIssue]]:
    """The source records, plus every environment error (unknown, not ready,
    busy, duplicated, too many, or the target slug itself). Preview, start
    and resume all gate on this one function; ``ignore_job`` is the combine
    job being resumed, which is not "busy" with itself."""
    from src.services.projects import busy
    from src.services.projects.registry import get_project_registry

    errors: list[CombineIssue] = []
    if len(request.sources) > MAX_SOURCES:
        errors.append(
            CombineIssue(
                code='too_many_sources',
                message=f'a combine takes at most {MAX_SOURCES} sources',
                detail={'max': MAX_SOURCES, 'given': len(request.sources)},
            )
        )
    registry = get_project_registry()
    await registry.ensure_fresh()
    records: list[ProjectRecord] = []
    seen: set[str] = set()
    for source in request.sources:
        slug = source.project
        if slug in seen:
            errors.append(CombineIssue(code='duplicate_source', project=slug))
            continue
        seen.add(slug)
        if slug == request.target.slug:
            errors.append(CombineIssue(code='target_is_source', project=slug))
            continue
        record = registry.get(slug)
        if record is None or record.status == 'deleted':
            errors.append(CombineIssue(code='source_not_found', project=slug))
        elif record.status not in _SOURCE_STATUSES:
            errors.append(
                CombineIssue(
                    code='source_not_ready',
                    project=slug,
                    message=f"'{slug}' is {record.status}",
                )
            )
        elif jobs := [
            j
            for j in busy.running_jobs(record)
            if not (j.kind == 'combine' and j.job_id == ignore_job)
        ]:
            errors.append(
                CombineIssue(
                    code='source_busy',
                    project=slug,
                    detail={'jobs': [f'{j.kind}:{j.job_id}' for j in jobs]},
                )
            )
        else:
            records.append(record)
    return records, errors


async def check_target(
    client: Any, request: CombineRequest
) -> tuple[list[CombineIssue], list[CombineIssue]]:
    """Target-side errors and warnings: slug, and the shard budget."""
    from src.services.projects import lifecycle
    from src.services.projects.registry import get_project_registry

    errors: list[CombineIssue] = []
    warnings: list[CombineIssue] = []
    slug = request.target.slug
    if not is_valid_slug(slug):
        errors.append(CombineIssue(code='slug_invalid', message=f"'{slug}' is not a valid slug"))
        return errors, warnings
    registry = get_project_registry()
    await registry.ensure_fresh()
    existing = registry.get(slug)
    if existing is not None:
        code = 'slug_retired' if existing.status == 'deleted' else 'slug_taken'
        errors.append(CombineIssue(code=code, message=f"'{slug}' cannot be used"))
        return errors, warnings
    try:
        warnings.extend(
            CombineIssue(
                code='shard_budget_high',
                severity='warning',
                message=w.get('message', ''),
                detail=w.get('detail') or {},
            )
            for w in await lifecycle._capacity_error_or_warning(client)
        )
    except HTTPException as exc:
        detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {}
        if detail.get('error') != 'shard_budget_exceeded':
            raise
        errors.append(
            CombineIssue(
                code='shard_budget_exceeded',
                message=str(detail.get('message', '')),
                detail={'capacity': detail.get('capacity')},
            )
        )
    return errors, warnings


# ------------------------------------------------------------------ analysis


def class_target(
    mapping: CombineMapping, project: str, class_name: str | None
) -> tuple[str, str | None]:
    """``(kind, target class name)`` of one source class name. An item with
    no class is ``('item', None)``: it is copied unclassed."""
    if class_name is None:
        return 'item', None
    target = mapping.per_source.get(project, {}).get(class_name)
    if target is None:
        return 'skip', None
    return target.kind, target.class_name


def probes_of(
    items: list[dict[str, Any]], mapping: CombineMapping, project: str, names: dict[int, str]
) -> list[Probe]:
    """The mergeable (``item`` kind) items of one image as merge probes."""
    probes: list[Probe] = []
    for item in items:
        kind, target = class_target(mapping, project, item_class_name(item, names))
        if kind == 'item' and len(item.get('bbox_norm') or ()) == 4:
            probes.append(Probe(tuple(item['bbox_norm']), target, trust_rank(item)))  # type: ignore[arg-type]
    return probes


async def _source_pass(
    client: Any, request: CombineRequest, records: list[ProjectRecord]
) -> tuple[list[SourceStats], list[Fingerprint], dict[str, dict[str, int]], set[tuple[int, str]]]:
    stats: list[SourceStats] = []
    prints: list[Fingerprint] = []
    holdout: set[tuple[int, str]] = set()
    for index, (source, record) in enumerate(zip(request.sources, records, strict=False)):
        names = class_names_by_id(record)
        st = SourceStats(project=record.slug)
        async for page in iter_source_pages(
            client, record, validated_only=source.include.label_states == 'validated_only'
        ):
            for image in page:
                st.images += 1
                prints.append(Fingerprint(index, image.image_id, image.imohash, image.path))
                if image.is_holdout:
                    st.holdout_images += 1
                    holdout.add((index, image.image_id))
                for item in image.items:
                    st.items += 1
                    st.labeled_items += bool(item.get('class_validated'))
                    if (name := item_class_name(item, names)) is not None:
                        st.classes[name] = st.classes.get(name, 0) + 1
        stats.append(st)
    return stats, prints, {s.project: s.classes for s in stats}, holdout


async def _dedup_stats(
    client: Any,
    request: CombineRequest,
    records: list[ProjectRecord],
    mapping: CombineMapping,
    duplicates: dict[tuple[int, str], tuple[int, str]],
) -> tuple[int, int, list[dict[str, Any]]]:
    """``(merged_items, conflicts, samples)`` of the boxes of duplicate images,
    decided by the same rules the executor applies."""
    merged = conflicts = 0
    samples: list[dict[str, Any]] = []
    names = [class_names_by_id(r) for r in records]
    validated = [s.include.label_states == 'validated_only' for s in request.sources]
    for (src_i, image_id), (pri_i, pri_id) in sorted(duplicates.items()):
        mine = (await fetch_items(client, records[src_i], [image_id], validated[src_i])).get(
            image_id, []
        )
        theirs = (await fetch_items(client, records[pri_i], [pri_id], validated[pri_i])).get(
            pri_id, []
        )
        existing = probes_of(theirs, mapping, records[pri_i].slug, names[pri_i])
        incoming = probes_of(mine, mapping, records[src_i].slug, names[src_i])
        for decision in decide_merges(existing, incoming, iou_min=request.dedup_iou):
            if decision.kind == 'merge':
                merged += 1
            elif decision.kind == 'conflict':
                conflicts += 1
                if len(samples) < _CONFLICT_SAMPLES:
                    other = existing[decision.existing or 0]
                    samples.append(
                        {
                            'image_id': pri_id,
                            'kept': {'project': records[pri_i].slug, 'class': other.target_class},
                            'dropped': {
                                'project': records[src_i].slug,
                                'class': incoming[decision.incoming].target_class,
                            },
                        }
                    )
    return merged, conflicts, samples


def _nearest_dev(path: Path) -> int | None:
    for candidate in (path, *path.parents):
        try:
            return candidate.stat().st_dev
        except OSError:
            continue
    return None


def _byte_counts(
    records: list[ProjectRecord],
    prints: list[Fingerprint],
    duplicates: dict[tuple[int, str], tuple[int, str]],
    target_upload_root: Path,
) -> tuple[int, int]:
    link = copy = 0
    target_dev = _nearest_dev(target_upload_root)
    for fp in prints:
        if (fp.source_index, fp.image_id) in duplicates:
            continue
        root = records[fp.source_index].resources.upload_root
        try:
            Path(fp.path).relative_to(root)
            size = Path(fp.path).stat().st_size
        except (ValueError, OSError):
            continue
        if _nearest_dev(root) == target_dev:
            link += size
        else:
            copy += size
    return link, copy


async def analyze(
    client: Any, request: CombineRequest, records: list[ProjectRecord], *, target_dim: int
) -> Analysis:
    """Read the sources and compute the whole plan. ``records`` are in
    ``request.sources`` order, every source present."""
    from src.config.curation import base_curation_config
    from src.config.projects import resources_for_new

    stats, prints, class_counts, holdout = await _source_pass(client, request, records)
    registry_names = {r.slug: {c.class_name for c in class_names_registry(r)} for r in records}
    mapping = resolve_combine_mapping(request, class_counts, registry_names)
    duplicates = (
        await asyncio.to_thread(find_duplicates, prints)
        if request.dedup == 'content_hash' and not mapping.errors
        else {}
    )
    if duplicates and not mapping.errors:
        merged, conflicts, samples = await _dedup_stats(
            client, request, records, mapping, duplicates
        )
    else:
        merged, conflicts, samples = 0, 0, []
    union = set(holdout)
    for member, priority in duplicates.items():
        if member in holdout:
            union.discard(member)
            union.add(priority)
    target_root = resources_for_new(request.target.slug, base_curation_config()).upload_root
    link, copy = _byte_counts(records, prints, duplicates, target_root)
    warnings: list[CombineIssue] = []
    if conflicts:
        warnings.append(
            CombineIssue(
                code='label_conflicts',
                severity='warning',
                message=f'{conflicts} boxes disagree between sources; the first source wins',
                detail={'count': conflicts, 'samples': samples},
            )
        )
    if request.holdout == 'recompute' and any(s.holdout_images for s in stats):
        warnings.append(
            CombineIssue(
                code='holdout_recompute_contamination',
                severity='warning',
                message='a source has a frozen test split; models trained on it may have seen '
                'the recomputed test images',
            )
        )
    warnings.extend(await embedding_model_mismatch(client, records, target_dim))
    warnings.extend(await region_profiles_differ(client, request, records))
    return Analysis(
        request=request,
        records=records,
        mapping=mapping,
        stats=stats,
        fingerprints=prints,
        duplicates=duplicates,
        holdout_images=len(union),
        merged_items=merged,
        conflicts=conflicts,
        conflict_samples=samples,
        to_link_bytes=link,
        to_copy_bytes=copy,
        errors=list(mapping.errors),
        warnings=warnings,
    )


def class_names_registry(record: ProjectRecord) -> list[Any]:
    from src.clients.curation_opensearch import ClassRegistry

    return ClassRegistry(path=record.resources.class_registry_path).load().classes


# -------------------------------------------------------------- preview wire


async def source_state(client: Any, record: ProjectRecord) -> dict[str, Any]:
    """What ``preview_sha`` pins of a source: its registry revision, its items
    doc count and the newest item write."""
    index = record.resources.indexes[IndexRole.ITEMS]
    with bind_project(record, read_only=True):
        count = (await client.count(index=index, body={'query': {'match_all': {}}})).get('count')
        resp = await client.search(
            index=index,
            body={
                'size': 1,
                'query': {'match_all': {}},
                'sort': [{'updated_at': 'desc'}],
                '_source': ['updated_at'],
            },
        )
    hits = (resp.get('hits') or {}).get('hits') or []
    newest = (hits[0].get('_source') or {}).get('updated_at') if hits else None
    return {'slug': record.slug, 'revision': record.revision, 'items': count, 'newest': newest}


def compute_preview_sha(request: CombineRequest, states: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        {'request': request.model_dump(mode='json'), 'sources': states},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _target_wire(analysis: Analysis, slug_available: bool) -> dict[str, Any]:
    mapping = analysis.mapping
    counts: dict[str, int] = {}
    origins: dict[str, list[dict[str, str]]] = {}
    for st in analysis.stats:
        for name, n in st.classes.items():
            target = mapping.per_source.get(st.project, {}).get(name)
            if target is not None and target.kind == 'item' and target.class_name:
                counts[target.class_name] = counts.get(target.class_name, 0) + n
                origins.setdefault(target.class_name, []).append(
                    {'project': st.project, 'class': name}
                )
    # A combine target is always new, so there is no "before": these are the
    # counts the finished target will have. An item with no class is copied
    # unclassed (class_target), so it counts even though no mapping names it.
    unclassed = sum(st.items - sum(st.classes.values()) for st in analysis.stats)
    images = sum(s.images for s in analysis.stats) - len(analysis.duplicates)
    items = sum(counts.values()) + unclassed - analysis.merged_items - analysis.conflicts
    return {
        'slug': analysis.request.target.slug,
        'slug_available': slug_available,
        'classes': [
            {'id': i, 'name': n, 'count': counts.get(n, 0), 'from': origins.get(n, [])}
            for i, n in enumerate(mapping.target_classes)
        ],
        'projected_images': images,
        'projected_items': items,
        'unclassed_items': unclassed,
        'holdout_images': analysis.holdout_images,
    }


def build_preview(
    analysis: Analysis,
    *,
    errors: list[CombineIssue],
    warnings: list[CombineIssue],
    preview_sha: str,
    slug_available: bool,
) -> CombinePreview:
    mapping = analysis.mapping
    sources = []
    for st in analysis.stats:
        classes = []
        for name, n in sorted(st.classes.items()):
            target = mapping.per_source.get(st.project, {}).get(name)
            classes.append(
                {
                    'name': name,
                    'count': n,
                    'mapped_to': (target.class_name or target.kind) if target else None,
                }
            )
        sources.append(
            {
                'project': st.project,
                'images': st.images,
                'items': st.items,
                'labeled_items': st.labeled_items,
                'holdout_images': st.holdout_images,
                'classes': classes,
            }
        )
    counts = {s.project: s.classes for s in analysis.stats}
    return CombinePreview(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        preview_sha=preview_sha,
        suggested_mapping=suggest_combine_mapping(analysis.request, counts),
        sources=sources,
        target=_target_wire(analysis, slug_available),
        dedup={
            'identical_images': len(analysis.duplicates),
            'merged_items': analysis.merged_items,
            'conflicts': analysis.conflicts,
            'conflict_samples': analysis.conflict_samples,
            'near_duplicate_pairs_estimate': None,
        },
        bytes={'to_link': analysis.to_link_bytes, 'to_copy': analysis.to_copy_bytes},
    )


def duplicates_wire(duplicates: dict[tuple[int, str], tuple[int, str]]) -> dict[str, list[Any]]:
    """The duplicate map as JSON (``"<source index>:<image id>"`` keys)."""
    return {f'{i}:{image}': [pi, pid] for (i, image), (pi, pid) in duplicates.items()}


def duplicates_from_wire(raw: dict[str, list[Any]]) -> dict[tuple[int, str], tuple[int, str]]:
    out: dict[tuple[int, str], tuple[int, str]] = {}
    for key, (pi, pid) in raw.items():
        i, _, image = key.partition(':')
        out[(int(i), image)] = (int(pi), str(pid))
    return out


__all__ = [
    'Analysis',
    'analyze',
    'build_preview',
    'check_target',
    'compute_preview_sha',
    'duplicates_from_wire',
    'duplicates_wire',
    'probes_of',
    'resolve_sources',
    'source_state',
]
