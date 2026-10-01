"""Class mapping of a combine (projects plan section 6, W10.19).

The target registry is new, so a ``create`` row defines a target class and a
``map`` row names one by ``new_class_name``. Everything else is W10's:
:func:`~src.services.curation.dataset_import.mapping.suggest_mapping` for the
suggestions and :func:`~...resolve_mapping` for completeness and the
``map`` / ``create`` / ``skip`` / ``region`` vocabulary, applied to the target
classes as a registry. A class is its NAME: source ids are never read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.services.curation.dataset_import.mapping import (
    ClassMappingEntry,
    MapTarget,
    RegistryClassView,
    norm_class_name,
    resolve_mapping,
    suggest_mapping,
)
from src.services.projects.combine.models import CombineIssue


if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.services.projects.combine.models import CombineRequest

_ISSUE_CODE = {
    'class_mapping_incomplete': 'unmapped_class',
    'class_mapped_to_deprecated': 'mapping_target_invalid',
}


@dataclass
class CombineMapping:
    target_classes: list[str] = field(default_factory=list)
    """Target class names in id order."""
    per_source: dict[str, dict[str, MapTarget]] = field(default_factory=dict)
    """source slug -> source class name -> its target (``class_name`` is the
    target class for ``item``; ``class_id`` is the index in
    :attr:`target_classes`)."""
    errors: list[CombineIssue] = field(default_factory=list)


def _created_names(request: CombineRequest) -> dict[str, str]:
    """``norm -> name`` of every class a ``create`` row defines (first
    spelling wins; the same name in two sources is one class)."""
    created: dict[str, str] = {}
    for rows in request.class_mapping.values():
        for row in rows:
            if row.action == 'create':
                name = (row.new_class_name or row.dataset_class).strip()
                created.setdefault(norm_class_name(name), name)
    return created


def _ordered_targets(
    request: CombineRequest, created: Mapping[str, str], errors: list[CombineIssue]
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for name in request.target_classes or []:
        norm = norm_class_name(name)
        if norm not in created:
            errors.append(
                CombineIssue(
                    code='mapping_target_invalid',
                    message=f"target class '{name}' is not created by any mapping row",
                    detail={'class': name},
                )
            )
        elif norm not in seen:
            seen.add(norm)
            ordered.append(created[norm])
    for norm, name in created.items():
        if norm not in seen:
            seen.add(norm)
            ordered.append(name)
    return ordered


def _convert(
    row: ClassMappingEntry, ids: Mapping[str, int], project: str, errors: list[CombineIssue]
) -> ClassMappingEntry | None:
    """A combine row as the W10 row ``resolve_mapping`` understands: both
    ``create`` and ``map`` become a ``map`` onto a target class id."""
    if row.action in ('create', 'map'):
        name = (row.new_class_name or row.dataset_class).strip()
        target = ids.get(norm_class_name(name))
        if target is None:
            errors.append(
                CombineIssue(
                    code='mapping_target_invalid',
                    project=project,
                    message=f"'{row.dataset_class}' maps to '{name}', which no row creates",
                    detail={'class': row.dataset_class, 'new_class_name': name},
                )
            )
            return None
        return ClassMappingEntry(dataset_class=row.dataset_class, action='map', class_id=target)
    return row


def resolve_combine_mapping(
    request: CombineRequest,
    class_counts: Mapping[str, Mapping[str, int]],
    source_registry_names: Mapping[str, set[str]],
) -> CombineMapping:
    """Resolve every source class with an included label to a target.

    ``class_counts``: source slug -> class name -> included item count.
    ``source_registry_names``: each source registry's class names, so a row
    for a registered class with no included items is accepted, not an error.
    """
    result = CombineMapping()
    created = _created_names(request)
    result.target_classes = _ordered_targets(request, created, result.errors)
    ids = {norm_class_name(name): i for i, name in enumerate(result.target_classes)}
    views = [RegistryClassView(i, name) for i, name in enumerate(result.target_classes)]
    for source in request.sources:
        slug = source.project
        counts = class_counts.get(slug, {})
        rows = request.class_mapping.get(slug, [])
        reported = len(result.errors)
        unused = set(source_registry_names.get(slug, set())) - set(counts)
        entries = [
            converted
            for row in rows
            if row.dataset_class in counts or row.dataset_class not in unused
            if (converted := _convert(row, ids, slug, result.errors)) is not None
        ]
        resolved = resolve_mapping(
            sorted(counts), entries, registry_classes=views, accept_suggestions=False
        )
        already = {e.detail.get('class') for e in result.errors[reported:]}
        for err in resolved.errors:
            if err.dataset_class in already:
                continue  # its row was refused above; do not also call it unmapped
            code = _ISSUE_CODE.get(err.code, err.code)
            detail = {'class': err.dataset_class}
            if code == 'unmapped_class':
                detail['count'] = counts.get(err.dataset_class or '', 0)  # type: ignore[assignment]
            result.errors.append(
                CombineIssue(
                    code=code,
                    project=slug,
                    message=err.message or f"class '{err.dataset_class}' has no mapping",
                    detail=detail,
                )
            )
        result.per_source[slug] = dict(resolved.targets)
    return result


def suggest_combine_mapping(
    request: CombineRequest, class_counts: Mapping[str, Mapping[str, int]]
) -> dict[str, list[ClassMappingEntry]]:
    """W10's name-based suggestions: a class whose name (exact, else any
    case) already exists among the classes suggested so far maps onto it,
    anything else is created under its own name."""
    views: list[RegistryClassView] = []
    out: dict[str, list[ClassMappingEntry]] = {}
    for source in request.sources:
        rows: list[ClassMappingEntry] = []
        counts = class_counts.get(source.project, {})
        for name in sorted(counts, key=lambda n: (-counts[n], n)):
            hit = suggest_mapping(name, registry_classes=views)
            if hit.action == 'map':
                rows.append(
                    ClassMappingEntry(
                        dataset_class=name, action='map', new_class_name=hit.class_name
                    )
                )
            else:
                views.append(RegistryClassView(len(views), name))
                rows.append(
                    ClassMappingEntry(dataset_class=name, action='create', new_class_name=name)
                )
        out[source.project] = rows
    return out


__all__ = ['CombineMapping', 'resolve_combine_mapping', 'suggest_combine_mapping']
