"""Class-name mapping: a dataset's integer class ids are NEVER
compared with the project registry's ids. Every dataset class is mapped
to a target by NAME, through an explicit ``map`` / ``create`` / ``skip``
/ ``region`` action — the class identity invariant
(``any_domain_plan.md``, "Class identity invariant" section) this wave
exists to enforce.

Exported for combine-projects: ``suggest_mapping`` /
``resolve_mapping`` map one project's/dataset's class names onto another
registry by name with the same action vocabulary and completeness rule.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.utils.class_names import resolve_class_by_name


MapAction = Literal['map', 'create', 'skip', 'region']
MatchKind = Literal[
    'same_registry', 'exact', 'case_insensitive', 'merged', 'synonym', 'region', 'none'
]

# accept_suggestions only auto-fills a mapping from these match kinds
# synonym/merged/create always need an explicit human choice.
_AUTO_ACCEPTABLE_MATCHES = frozenset({'same_registry', 'exact', 'case_insensitive', 'region'})


def norm_class_name(name: str) -> str:
    """NFKC-normalize, casefold, strip; ``_``/``-`` -> space; collapse
    whitespace. Suggestions only — never used to compare against a
    registry id."""
    s = unicodedata.normalize('NFKC', name).casefold().strip()
    s = s.replace('_', ' ').replace('-', ' ')
    return ' '.join(s.split())


@dataclass(frozen=True)
class RegistryClassView:
    """The subset of a registry class ``suggest_mapping``/``resolve_mapping``
    need — decoupled from ``RegistryClassEntry`` so this module has no
    OpenSearch/registry-file import."""

    class_id: int
    class_name: str
    deprecated: bool = False
    merged_into: int | None = None


class ClassMappingEntry(BaseModel):
    """One dataset-class mapping decision in a preview/import request."""

    model_config = ConfigDict(extra='forbid')

    dataset_class: str
    action: MapAction
    class_id: int | None = None
    new_class_name: str | None = None
    new_class_group: str | None = None


@dataclass(frozen=True)
class MappingSuggestion:
    action: MapAction
    match: MatchKind
    class_id: int | None = None
    class_name: str | None = None


@dataclass(frozen=True)
class MapTarget:
    """The resolved target for one dataset class's boxes."""

    kind: Literal['item', 'region', 'skip']
    class_id: int | None = None
    class_name: str | None = None


@dataclass
class MappingError:
    code: str
    dataset_class: str | None = None
    message: str = ''

    def describe(self) -> str:
        """``class: code (what is wrong)``, the one rendering of an error."""
        base = f'{self.dataset_class}: {self.code}' if self.dataset_class else self.code
        return f'{base} ({self.message})' if self.message else base


@dataclass
class ResolvedMapping:
    targets: dict[str, MapTarget] = field(default_factory=dict)
    created_classes: dict[str, int] = field(default_factory=dict)
    """dataset_class -> newly created (or to-create) class_id."""
    merged_from: dict[int, list[str]] = field(default_factory=dict)
    """target class_id -> dataset classes mapped onto it (>1 means merged synonyms)."""
    errors: list[MappingError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def suggest_mapping(
    dataset_class: str,
    *,
    registry_classes: list[RegistryClassView],
    synonyms: dict[str, str] | None = None,
    region_class_name: str | None = None,
    is_single_class_region_export: bool = False,
    source_class_id: int | None = None,
) -> MappingSuggestion:
    """The suggestion ladder, first hit wins."""
    norm = norm_class_name(dataset_class)

    if source_class_id is not None:
        for c in registry_classes:
            if (
                c.class_id == source_class_id
                and not c.deprecated
                and norm_class_name(c.class_name) == norm
            ):
                return MappingSuggestion('map', 'same_registry', c.class_id, c.class_name)

    by_name = resolve_class_by_name(registry_classes, dataset_class)
    if by_name.active is not None:
        c = by_name.active
        if c.class_name == dataset_class:
            return MappingSuggestion('map', 'exact', c.class_id, c.class_name)
        return MappingSuggestion('map', 'case_insensitive', c.class_id, c.class_name)
    retired = by_name.entry
    if retired is not None and retired.merged_into is not None:
        target = next((t for t in registry_classes if t.class_id == retired.merged_into), None)
        if target is not None:
            return MappingSuggestion('map', 'merged', target.class_id, target.class_name)

    if synonyms:
        target_name = synonyms.get(norm)
        if target_name:
            synonym = resolve_class_by_name(registry_classes, target_name)
            if synonym.active is not None:
                c = synonym.active
                return MappingSuggestion('map', 'synonym', c.class_id, c.class_name)

    if region_class_name is not None and norm_class_name(region_class_name) == norm:
        return MappingSuggestion('region', 'region')
    if is_single_class_region_export:
        return MappingSuggestion('region', 'region')

    return MappingSuggestion('create', 'none', None, dataset_class)


def _index_entries(
    dataset_classes: list[str], entries: list[ClassMappingEntry], result: ResolvedMapping
) -> dict[str, ClassMappingEntry]:
    """Entries by dataset class; unknown and duplicate rows become errors."""
    by_class: dict[str, ClassMappingEntry] = {}
    known = set(dataset_classes)
    for e in entries:
        if e.dataset_class not in known:
            result.errors.append(
                MappingError(
                    'class_mapping_invalid',
                    e.dataset_class,
                    'this dataset has no class with that name; check the spelling against the preview',
                )
            )
        elif e.dataset_class in by_class:
            result.errors.append(
                MappingError(
                    'class_mapping_invalid',
                    e.dataset_class,
                    'the mapping lists this class more than once; keep one row',
                )
            )
        else:
            by_class[e.dataset_class] = e
    return by_class


def _accepted_suggestion(
    dataset_class: str, suggestion: MappingSuggestion | None
) -> ClassMappingEntry | None:
    """The entry an auto-acceptable suggestion stands for, else None."""
    if suggestion is None or suggestion.match not in _AUTO_ACCEPTABLE_MATCHES:
        return None
    if suggestion.action == 'map':
        return ClassMappingEntry(
            dataset_class=dataset_class, action='map', class_id=suggestion.class_id
        )
    if suggestion.action == 'region':
        return ClassMappingEntry(dataset_class=dataset_class, action='region')
    return None


def _resolve_map(
    dataset_class: str,
    entry: ClassMappingEntry,
    by_id: dict[int, RegistryClassView],
    result: ResolvedMapping,
) -> None:
    target = by_id.get(entry.class_id) if entry.class_id is not None else None
    if target is None:
        result.errors.append(
            MappingError(
                'class_mapping_invalid',
                dataset_class,
                f'action map needs the class_id of an existing project class; got {entry.class_id!r}',
            )
        )
        return
    if target.deprecated:
        result.errors.append(MappingError('class_mapped_to_deprecated', dataset_class))
        return
    result.targets[dataset_class] = MapTarget(
        kind='item', class_id=target.class_id, class_name=target.class_name
    )
    result.merged_from.setdefault(target.class_id, []).append(dataset_class)


def _resolve_create(
    dataset_class: str,
    entry: ClassMappingEntry,
    by_norm_name: dict[str, RegistryClassView],
    created_norm_names: set[str],
    result: ResolvedMapping,
) -> None:
    new_name = (entry.new_class_name or dataset_class).strip()
    norm_new = norm_class_name(new_name)
    if norm_new in by_norm_name:
        result.errors.append(MappingError('class_name_exists', dataset_class, new_name))
        return
    if norm_new in created_norm_names:
        result.errors.append(MappingError('class_mapping_conflict', dataset_class, new_name))
        return
    created_norm_names.add(norm_new)
    result.created_classes[dataset_class] = -1  # resolved to a real id at job start
    result.targets[dataset_class] = MapTarget(kind='item', class_id=None, class_name=new_name)


def resolve_mapping(
    dataset_classes: list[str],
    entries: list[ClassMappingEntry],
    *,
    registry_classes: list[RegistryClassView],
    accept_suggestions: bool = False,
    suggestions: dict[str, MappingSuggestion] | None = None,
) -> ResolvedMapping:
    """Resolve every dataset class (with >=1 box) to a :class:`MapTarget`.

    ``suggestions`` (precomputed per dataset class, e.g. from a prior
    ``suggest_mapping`` pass) is required for ``accept_suggestions`` to
    fill in anything; entries always take precedence over a suggestion.
    """
    result = ResolvedMapping()
    by_class = _index_entries(dataset_classes, entries, result)
    by_id = {c.class_id: c for c in registry_classes}
    by_norm_name = {norm_class_name(c.class_name): c for c in registry_classes if not c.deprecated}
    created_norm_names: set[str] = set()

    for dataset_class in dataset_classes:
        entry = by_class.get(dataset_class)
        if entry is None and accept_suggestions and suggestions is not None:
            entry = _accepted_suggestion(dataset_class, suggestions.get(dataset_class))

        if entry is None:
            result.errors.append(MappingError('class_mapping_incomplete', dataset_class))
        elif entry.action == 'skip':
            result.targets[dataset_class] = MapTarget(kind='skip')
        elif entry.action == 'region':
            result.targets[dataset_class] = MapTarget(kind='region')
        elif entry.action == 'map':
            _resolve_map(dataset_class, entry, by_id, result)
        elif entry.action == 'create':
            _resolve_create(dataset_class, entry, by_norm_name, created_norm_names, result)
        else:
            result.errors.append(
                MappingError(
                    'class_mapping_invalid',
                    dataset_class,
                    f'unknown action {entry.action!r}; use map, create, skip or region',
                )
            )

    return result


__all__ = [
    'ClassMappingEntry',
    'MapAction',
    'MapTarget',
    'MappingError',
    'MappingSuggestion',
    'MatchKind',
    'RegistryClassView',
    'ResolvedMapping',
    'norm_class_name',
    'resolve_mapping',
    'suggest_mapping',
]
