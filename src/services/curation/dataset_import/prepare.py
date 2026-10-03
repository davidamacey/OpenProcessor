"""Everything an import decides before it writes: scan the dataset, resolve
the class mapping by name, collect issues, derive the identity key.

``prepare_import`` is the ONE function the preview, the start route, a
resume and ``expected_import_key`` all call, so they cannot disagree about
what an import would do. It reads the bound project exactly once (a
:class:`ProjectView`): the classes, the synonym table and the region
profile are pinned there and persisted with the import, so a worker that
resumes the job in another process never re-resolves them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.curation.dataset_import.issues import ISSUE_CATALOG, DatasetIssue
from src.services.curation.dataset_import.mapping import (
    ClassMappingEntry,
    MappingSuggestion,
    RegistryClassView,
    ResolvedMapping,
    norm_class_name,
    resolve_mapping,
    suggest_mapping,
)
from src.services.curation.dataset_import.regions import (
    ParentCandidate,
    attach_region_boxes,
    parents_mode,
)
from src.services.curation.dataset_import.scan import scan_dataset, source_sha
from src.utils.class_names import resolve_class_by_name


if TYPE_CHECKING:
    from pathlib import Path

    from src.clients.curation_opensearch import ClassRegistry
    from src.services.curation.dataset_import.options import DatasetPreviewRequest
    from src.services.curation.dataset_import.paths import PathGuard
    from src.services.curation.dataset_import.scan import DatasetScan


logger = get_logger(__name__)


@dataclass(frozen=True)
class PinnedProfile:
    """The active region profile as it was when the import was prepared."""

    name: str
    revision: int | None
    region_class_name: str
    parent_classes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), 'parent_classes': list(self.parent_classes)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PinnedProfile | None:
        if not raw:
            return None
        return cls(
            name=str(raw['name']),
            revision=raw.get('revision'),
            region_class_name=str(raw.get('region_class_name') or ''),
            parent_classes=tuple(raw.get('parent_classes') or ()),
        )


@dataclass(frozen=True)
class ProjectView:
    """The bound project's state an import depends on, read once."""

    project: str
    registry_classes: list[RegistryClassView]
    synonyms: dict[str, str]
    profile: PinnedProfile | None
    export_root: Path | None


def registry_class_views(registry: ClassRegistry) -> list[RegistryClassView]:
    reg = registry.load()
    return [
        RegistryClassView(
            class_id=c.class_id,
            class_name=c.class_name,
            deprecated=c.deprecated,
            merged_into=c.merged_into,
        )
        for c in reg.classes
    ]


def _active_profile_pin() -> PinnedProfile | None:
    from src.services.detection.profile_registry import get_active_region_profile

    profile = get_active_region_profile()
    if profile is None:
        return None
    revision: int | None = None
    try:
        from src.services.config_store import get_config_store

        ref = get_config_store().current.active_profile
        if isinstance(ref, tuple):
            revision = ref[1]
    except Exception as exc:
        logger.warning('import_profile_revision_unavailable', error=str(exc))
    return PinnedProfile(
        name=profile.name,
        revision=revision,
        region_class_name=profile.region_class_name or '',
        parent_classes=tuple(sorted(profile.parent_classes)),
    )


def _pack_synonyms() -> dict[str, str]:
    from src.services.labeling.vlm_prompt_resolution import active_prompt_pack

    try:
        pack = active_prompt_pack()
    except Exception as exc:
        logger.warning('import_pack_synonyms_unavailable', error=str(exc))
        return {}
    return {norm_class_name(k): v for k, v in pack.synonyms.items()}


def pin_project_view(registry: ClassRegistry) -> ProjectView:
    """Read the bound project's classes, synonyms, region profile and export
    root once."""
    from src.config.curation import get_curation_config

    cfg = get_curation_config()
    return ProjectView(
        project=cfg.project_slug,
        registry_classes=registry_class_views(registry),
        synonyms=_pack_synonyms(),
        profile=_active_profile_pin(),
        export_root=cfg.export_root,
    )


@dataclass
class PreparedImport:
    scan: DatasetScan
    view: ProjectView
    source_sha: str
    import_key: str
    suggestions: dict[str, MappingSuggestion]
    resolved: ResolvedMapping
    issues: list[DatasetIssue]
    parents: str
    """The resolved parents mode: ``labels`` or ``detect``."""
    dataset_class_ids: dict[str, int] = field(default_factory=dict)
    standalone_boxes: int = 0
    reuse_key: str | None = None
    """The key this request had BEFORE its ``create`` classes existed: set only
    when the sole mapping errors are ``class_name_exists`` (a repeat of an
    import that already created them), so a completed import is found again."""

    @property
    def blocking(self) -> bool:
        return any(i.blocking for i in self.issues)

    def force_allowed(self) -> bool:
        blocking = [i for i in self.issues if i.blocking]
        return bool(blocking) and all(i.bypassable for i in blocking)


def canonical_mapping(resolved: ResolvedMapping) -> list[dict[str, Any]]:
    """The mapping as it identifies an import: per dataset class, its kind
    and target NAME. The id is left out on purpose: class identity is the
    name, and a ``create`` has no id until the import starts."""
    out = []
    for dataset_class in sorted(resolved.targets):
        t = resolved.targets[dataset_class]
        out.append(
            {
                'dataset_class': dataset_class,
                'kind': t.kind,
                'class_name': t.class_name,
            }
        )
    return out


def compute_import_key(
    project: str, source_sha_value: str, resolved: ResolvedMapping, options_key: dict[str, Any]
) -> str:
    payload = json.dumps(
        {
            'project': project,
            'source_sha': source_sha_value,
            'mapping': canonical_mapping(resolved),
            'options': options_key,
        },
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _issue(code: str, message: str | None = None, *, count: int = 1) -> DatasetIssue:
    spec = ISSUE_CATALOG[code]
    return DatasetIssue(
        code=code,
        severity=spec.severity,
        blocking=spec.blocking,
        bypassable=spec.bypassable,
        message=message or spec.label,
        count=count,
    )


def _suggestions(scan: DatasetScan, view: ProjectView) -> dict[str, MappingSuggestion]:
    op = scan.op_export
    source_ids: dict[str, int] = dict(getattr(op, 'source_classes', None) or {})
    single_region = bool(
        op is not None
        and getattr(op, 'box_source', None) == 'region'
        and len(scan.class_box_counts) <= 1
    )
    return {
        name: suggest_mapping(
            name,
            registry_classes=view.registry_classes,
            synonyms=view.synonyms,
            region_class_name=view.profile.region_class_name if view.profile else None,
            is_single_class_region_export=single_region,
            source_class_id=source_ids.get(name),
        )
        for name in scan.class_box_counts
    }


def _mapping_issues(resolved: ResolvedMapping, view: ProjectView) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    by_code: dict[str, list[str]] = {}
    for err in resolved.errors:
        code = {'class_mapping_incomplete': 'class_unmapped'}.get(err.code, err.code)
        by_code.setdefault(code, []).append(err.dataset_class or '')
    for code, names in by_code.items():
        if code in ISSUE_CATALOG:
            issues.append(_issue(code, count=len(names)))
        else:
            issues.append(_issue('class_unmapped', count=len(names)))
    has_region = any(t.kind == 'region' for t in resolved.targets.values())
    if has_region and view.profile is None:
        issues.append(_issue('region_profile_required'))
    if has_region and view.profile is not None and view.profile.region_class_name:
        wanted = norm_class_name(view.profile.region_class_name)
        differing = [
            n
            for n, t in resolved.targets.items()
            if t.kind == 'region' and norm_class_name(n) != wanted
        ]
        if differing:
            issues.append(_issue('region_class_name_differs', count=len(differing)))
    return issues


def _index_mismatch_issue(
    scan: DatasetScan, view: ProjectView, dataset_ids: dict[str, int]
) -> DatasetIssue | None:
    """Info: the old id-for-id path would have mislabeled these classes."""
    by_id = {c.class_id: c.class_name for c in view.registry_classes if not c.deprecated}
    count = sum(
        1
        for name in scan.class_box_counts
        if dataset_ids.get(name) in by_id and by_id[dataset_ids[name]] != name
    )
    return _issue('class_index_name_mismatch', count=count) if count else None


def _standalone_boxes(scan: DatasetScan, resolved: ResolvedMapping, view: ProjectView) -> int:
    """Region boxes that would become standalone items (parents from labels)."""
    parent_classes = frozenset(view.profile.parent_classes) if view.profile else frozenset()
    total = 0
    for entry in scan.entries:
        parents, boxes = [], []
        for i, box in enumerate(entry.boxes):
            target = resolved.targets.get(box.dataset_class)
            if target is None:
                continue
            if target.kind == 'item':
                parents.append(ParentCandidate(str(i), box.bbox_norm, target.class_name))
            elif target.kind == 'region':
                boxes.append(box)
        if boxes:
            total += len(
                attach_region_boxes(parents, boxes, parent_classes=parent_classes).standalone
            )
    return total


def prepare_import(
    request: DatasetPreviewRequest, view: ProjectView, *, path_guard: PathGuard
) -> PreparedImport:
    """Scan, map and identify one import. Writes nothing.

    Raises :class:`DatasetPathNotAllowedError` / :class:`FormatUndetectedError`
    from the scan; every other dataset problem is an issue on the result.
    """
    scan = scan_dataset(
        request.source, path_guard=path_guard, missing_label=request.options.missing_label
    )
    sha = source_sha(scan)
    suggestions = _suggestions(scan, view)
    dataset_classes = [name for name, n in scan.class_box_counts.items() if n > 0]
    resolved = resolve_mapping(
        dataset_classes,
        request.mapping,
        registry_classes=view.registry_classes,
        accept_suggestions=request.accept_suggestions,
        suggestions=suggestions,
    )
    issues = list(scan.issues.issues())
    issues.extend(_mapping_issues(resolved, view))
    mismatch = _index_mismatch_issue(scan, view, scan.class_ids)
    if mismatch is not None:
        issues.append(mismatch)
    has_item_parent = any(
        t.kind == 'item'
        and (
            not view.profile
            or not view.profile.parent_classes
            or t.class_name in view.profile.parent_classes
        )
        for t in resolved.targets.values()
    )
    parents = parents_mode(request.options.parents, has_item_parent_labels=has_item_parent)
    standalone = _standalone_boxes(scan, resolved, view) if parents == 'labels' else 0
    if standalone:
        issues.append(_issue('region_box_no_parent', count=standalone))
    key = compute_import_key(view.project, sha, resolved, request.options.key_fields())
    reuse_key = _reuse_key(request, view, dataset_classes, suggestions, sha, resolved)
    return PreparedImport(
        scan=scan,
        view=view,
        source_sha=sha,
        import_key=key,
        suggestions=suggestions,
        resolved=resolved,
        issues=issues,
        parents=parents,
        dataset_class_ids=dict(scan.class_ids),
        standalone_boxes=standalone,
        reuse_key=reuse_key,
    )


def _reuse_key(
    request: DatasetPreviewRequest,
    view: ProjectView,
    dataset_classes: list[str],
    suggestions: dict[str, MappingSuggestion],
    sha: str,
    resolved: ResolvedMapping,
) -> str | None:
    """The key of this request read as "map to the class a previous run of it
    created", when that is the only thing wrong with it."""
    if not resolved.errors or any(e.code != 'class_name_exists' for e in resolved.errors):
        return None
    by_norm = {norm_class_name(c.class_name): c for c in view.registry_classes if not c.deprecated}
    entries = []
    for e in request.mapping:
        existing = by_norm.get(norm_class_name((e.new_class_name or e.dataset_class).strip()))
        if e.action == 'create' and existing is not None:
            entries.append(
                ClassMappingEntry(
                    dataset_class=e.dataset_class, action='map', class_id=existing.class_id
                )
            )
        else:
            entries.append(e)
    again = resolve_mapping(
        dataset_classes,
        entries,
        registry_classes=view.registry_classes,
        accept_suggestions=request.accept_suggestions,
        suggestions=suggestions,
    )
    if again.errors:
        return None
    return compute_import_key(view.project, sha, again, request.options.key_fields())


def materialize_created_classes(
    resolved: ResolvedMapping,
    registry: ClassRegistry,
    *,
    new_groups: dict[str, str | None] | None = None,
    adopt_existing: bool = False,
) -> None:
    """Create every ``create`` target in the registry BEFORE any item write
    (W10.5), patching the assigned id into ``resolved``.

    ``adopt_existing`` (a resume): a non-deprecated class with exactly the
    target's name was created by the interrupted run, so its id is adopted
    instead of failing as a duplicate.
    """
    groups = new_groups or {}
    for dataset_class in list(resolved.created_classes):
        target = resolved.targets[dataset_class]
        if target.class_id is not None:
            continue
        name = target.class_name or dataset_class
        new_id: int | None = None
        if adopt_existing:
            adopted = resolve_class_by_name(registry.load().classes, name)
            new_id = adopted.active.class_id if adopted.active is not None else None
        if new_id is None:
            new_id = registry.add_class(name, group=groups.get(dataset_class) or 'unknown')
        resolved.targets[dataset_class] = dataclasses.replace(target, class_id=new_id)
        resolved.created_classes[dataset_class] = new_id


def mapping_to_dict(resolved: ResolvedMapping) -> dict[str, Any]:
    return {
        'targets': {
            name: {'kind': t.kind, 'class_id': t.class_id, 'class_name': t.class_name}
            for name, t in resolved.targets.items()
        },
        'created_classes': dict(resolved.created_classes),
    }


def mapping_from_dict(raw: dict[str, Any]) -> ResolvedMapping:
    from src.services.curation.dataset_import.mapping import MapTarget

    resolved = ResolvedMapping()
    for name, t in (raw.get('targets') or {}).items():
        resolved.targets[name] = MapTarget(
            kind=t['kind'], class_id=t.get('class_id'), class_name=t.get('class_name')
        )
    resolved.created_classes = {k: int(v) for k, v in (raw.get('created_classes') or {}).items()}
    for dataset_class, target in resolved.targets.items():
        if target.kind == 'item' and target.class_id is not None:
            resolved.merged_from.setdefault(target.class_id, []).append(dataset_class)
    return resolved


__all__ = [
    'PinnedProfile',
    'PreparedImport',
    'ProjectView',
    'canonical_mapping',
    'compute_import_key',
    'mapping_from_dict',
    'mapping_to_dict',
    'materialize_created_classes',
    'pin_project_view',
    'prepare_import',
    'registry_class_views',
]
