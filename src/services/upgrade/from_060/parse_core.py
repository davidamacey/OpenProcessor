"""Projects, class registries, config documents, activations, settings, keymap and view state:
0.6.0 documents and files in, :mod:`model` records out. Pure; no database, no OpenSearch."""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import TYPE_CHECKING, Any

from src.db.models.config import ACTIVATION_AXES
from src.services.upgrade.from_060 import sources as legacy
from src.services.upgrade.from_060.ids import stable_id
from src.services.upgrade.from_060.model import (
    ActivationSrc,
    ClassSrc,
    ConfigDocSrc,
    ConfigRevisionSrc,
    GlobalData,
    KeymapSrc,
    ProbeSrc,
    ProjectData,
    ProjectSrc,
    RegistrySrc,
    RetiredSlug,
    SettingsSrc,
    ViewStateSrc,
    VlmEndpointSrc,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.services.upgrade.from_060.snapshot import Snapshot

POLICY_KEYS = ('ingest_policy', 'vlm_policy', 'vlm_scope_reps')
_AXIS_KIND = dict(ACTIVATION_AXES)
_REVISION_ID = re.compile(r'^(?P<prefix>[a-z]+):(?P<name>.+)@(?P<rev>\d+)$')
_CURRENT_ID = re.compile(r'^(?P<prefix>[a-z]+):(?P<name>[^@]+)$')
_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


def parse_ts(value: Any, default: dt.datetime) -> dt.datetime:
    """An ISO timestamp as an aware UTC datetime; ``default`` for a blank or unreadable one."""
    if not value:
        return default
    try:
        parsed = dt.datetime.fromisoformat(str(value))
    except ValueError:
        return default
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


# --- projects ---------------------------------------------------------------------------------


def parse_projects(
    snap: Snapshot, *, now: dt.datetime
) -> tuple[list[ProjectSrc], list[RetiredSlug], int]:
    """Live projects, retired slugs (deleted projects) and the projects revision counter."""
    docs = snap.index_docs(legacy.PROJECTS_INDEX)
    revision = int((docs.get(legacy.PROJECTS_REVISION_DOC_ID) or {}).get('revision', 0))
    live: list[ProjectSrc] = []
    retired: list[RetiredSlug] = []
    for doc_id in sorted(d for d in docs if d.startswith(legacy.PROJECT_DOC_PREFIX)):
        doc = docs[doc_id]
        slug = str(doc['slug'])
        created = parse_ts(doc.get('created_at'), now)
        updated = parse_ts(doc.get('updated_at'), created)
        if doc.get('status') == 'deleted':
            retired.append(RetiredSlug(slug, created, updated))
            continue
        live.append(_project(snap, doc, slug, created, updated))
    return live, retired, revision


def _project(
    snap: Snapshot, doc: Mapping[str, Any], slug: str, created: dt.datetime, updated: dt.datetime
) -> ProjectSrc:
    resources = doc['resources']
    roles: dict[str, str] = dict(resources['indexes'])
    indexes = {role: roles[role] for role in legacy.KEPT_ROLES if role in roles}
    items = indexes.get('items', '')
    flag = snap.file(f'state:{slug}', legacy.PAUSE_FLAG)
    status = str(doc['status'])
    return ProjectSrc(
        id=stable_id('project', slug),
        slug=slug,
        display_name=str(doc.get('display_name') or slug),
        description=str(doc.get('description') or ''),
        status=status,
        pre_delete_status=doc.get('pre_delete_status') if status == 'deleting' else None,
        revision=int(doc.get('revision') or 1),
        created_at=created,
        updated_at=updated,
        origin=doc.get('origin'),
        indexes=indexes,
        model_prefix=str(resources.get('model_prefix', '')),
        mlflow_experiment=str(resources.get('mlflow_experiment') or f'openprocessor-{slug}'),
        index_prefix=items.removesuffix('items'),
        storage_prefix=f'projects/{slug}',
        paused_at=parse_ts(flag.mtime, updated) if flag is not None else None,
        legacy_indexes={role: roles[role] for role in legacy.CONTROL_ROLES if role in roles},
    )


# --- class registry ---------------------------------------------------------------------------


def _class(entry: Mapping[str, Any], now: dt.datetime) -> ClassSrc:
    group = entry.get('group', 'unknown')
    return ClassSrc(
        class_id=int(entry.get('class_id', entry.get('id'))),
        name=str(entry.get('class_name', entry.get('name'))),
        class_group=None if group in (None, '') else str(group),
        deprecated=bool(entry.get('deprecated', False)),
        merged_into=None if entry.get('merged_into') is None else int(entry['merged_into']),
        hotkey_letter=entry.get('hotkey_letter') or None,
        notes=str(entry.get('notes') or ''),
        added_at=parse_ts(entry.get('added_at', entry.get('added')), now),
    )


def parse_registry(snap: Snapshot, project: ProjectSrc, *, now: dt.datetime) -> RegistrySrc | None:
    """The class registry document when there is one (``main`` wrote it), else the file."""
    settings_index = project.legacy_indexes.get('settings') or project.legacy_indexes.get('configs')
    doc = (
        snap.index_docs(settings_index or '').get(legacy.CLASS_REGISTRY_DOC_ID)
        if settings_index
        else None
    )
    raw: Mapping[str, Any] | None
    if doc is not None and doc.get('class_registry') is not None:
        raw, origin = doc['class_registry'], 'opensearch_registry_doc'
    else:
        file = snap.file(f'class_registry:{project.slug}', legacy.CLASS_REGISTRY_FILE)
        if file is None:
            return None
        raw, origin = file.json(), 'class_registry_file'
    classes = tuple(
        sorted((_class(c, now) for c in raw.get('classes', [])), key=lambda c: c.class_id)
    )
    highest = max((c.class_id for c in classes), default=-1)
    return RegistrySrc(
        classes=classes,
        next_class_id=max(int(raw.get('next_class_id') or 0), highest + 1),
        revision=int(raw.get('registry_revision') or 0),
        migrated_from=origin,
        migrated_at=now,
    )


# --- config documents and activations ---------------------------------------------------------


def _revisions_of(
    docs: Mapping[str, Mapping[str, Any]], prefix: str, name: str, now: dt.datetime
) -> tuple[ConfigRevisionSrc, ...]:
    out: list[ConfigRevisionSrc] = []
    for doc_id, doc in docs.items():
        match = _REVISION_ID.match(doc_id)
        if match and match['prefix'] == prefix and match['name'] == name:
            out.append(
                ConfigRevisionSrc(
                    revision=int(match['rev']),
                    body=doc.get('body') or {},
                    created_at=parse_ts(doc.get('updated_at') or doc.get('created_at'), now),
                    cloned_from=doc.get('cloned_from'),
                )
            )
    return tuple(sorted(out, key=lambda r: r.revision))


def _referenced(activations: list[ActivationSrc]) -> set[tuple[str, str]]:
    refs: set[tuple[str, str]] = set()
    for event in activations:
        refs.add((event.kind, event.to_name))
        if event.from_name:
            refs.add((event.kind, event.from_name))
    return refs


def parse_config_docs(
    docs: Mapping[str, Mapping[str, Any]],
    project_slug: str | None,
    referenced: set[tuple[str, str]],
    *,
    now: dt.datetime,
) -> tuple[ConfigDocSrc, ...]:
    """Current documents, and documents deleted since but still named by an activation (their
    revision copies stayed, so a pinned activation keeps resolving)."""
    names: dict[tuple[str, str], Mapping[str, Any] | None] = {}
    for doc_id, doc in docs.items():
        if doc.get('doc_type') != 'config':
            continue
        match = _CURRENT_ID.match(doc_id)
        kind = legacy.CONFIG_PREFIX_TO_KIND.get(match['prefix']) if match else None
        if match and kind and kind == doc.get('kind'):
            names[(kind, match['name'])] = doc
    for kind, name in referenced:
        names.setdefault((kind, name), None)
    prefix_of = {kind: prefix for prefix, kind in legacy.CONFIG_PREFIX_TO_KIND.items()}
    out: list[ConfigDocSrc] = []
    for (kind, name), current in sorted(names.items()):
        if kind not in prefix_of:
            continue
        revisions = _revisions_of(docs, prefix_of[kind], name, now)
        if not revisions:
            continue
        stamp = parse_ts((current or {}).get('updated_at'), revisions[-1].created_at)
        out.append(
            ConfigDocSrc(
                project_slug=project_slug,
                kind=kind,
                name=name,
                description=str((current or {}).get('description') or ''),
                revisions=revisions,
                created_at=parse_ts((current or {}).get('created_at'), revisions[0].created_at),
                updated_at=stamp,
            )
        )
    return tuple(out)


def parse_activations(
    docs: Mapping[str, Mapping[str, Any]], *, now: dt.datetime
) -> list[ActivationSrc]:
    """Activation events per axis in time order, ending in the axis's current activation.

    An axis whose current activation is "off" (a null name) or the ``env`` built-in (no
    revision) has nothing 0.7.0 can point at, so it contributes no events.
    """
    events: dict[str, list[ActivationSrc]] = {}
    currents: dict[str, Mapping[str, Any]] = {}
    for doc_id, doc in sorted(docs.items()):
        if doc_id.startswith(legacy.ACTIVATION_DOC_PREFIX):
            currents[str(doc.get('axis'))] = doc
        elif doc_id.startswith(legacy.ACTIVATION_EVENT_DOC_PREFIX) and doc.get('name') is not None:
            axis = str(doc['axis'])
            if axis not in _AXIS_KIND or doc.get('revision') is None:
                continue
            previous = doc.get('previous') or {}
            events.setdefault(axis, []).append(
                ActivationSrc(
                    axis=axis,
                    from_name=previous.get('name'),
                    from_revision=previous.get('revision'),
                    to_name=str(doc['name']),
                    to_revision=int(doc['revision']),
                    kind=_AXIS_KIND[axis],
                    at=parse_ts(doc.get('activated_at'), now),
                )
            )
    out: list[ActivationSrc] = []
    for axis in sorted(set(events) | set(currents)):
        if axis not in _AXIS_KIND:
            continue
        current = currents.get(axis)
        history = sorted(events.get(axis, []), key=lambda e: e.at)
        if current is not None and (current.get('name') is None or current.get('revision') is None):
            continue
        if current is not None:
            last = history[-1] if history else None
            if last is None or (last.to_name, last.to_revision) != (
                current['name'],
                int(current['revision']),
            ):
                previous = current.get('previous') or {}
                history.append(
                    ActivationSrc(
                        axis,
                        previous.get('name'),
                        previous.get('revision'),
                        str(current['name']),
                        int(current['revision']),
                        _AXIS_KIND[axis],
                        parse_ts(current.get('activated_at'), now),
                    )
                )
        out.extend(history)
    return out


# --- settings, keymap, view state -------------------------------------------------------------


def parse_settings(
    snap: Snapshot, project: ProjectSrc
) -> tuple[SettingsSrc | None, dict[str, Any] | None]:
    """The settings document (and the cluster-refresh state inside it)."""
    index = project.legacy_indexes.get('settings')
    doc = snap.index_docs(index or '').get(legacy.SETTINGS_DOC_ID) if index else None
    if doc is None:
        return None, None
    defaults = {k: v for k, v in (doc.get('defaults') or {}).items() if v is not None}
    policies = {key: dict(doc[key]) for key in POLICY_KEYS if doc.get(key)}
    settings = SettingsSrc(
        defaults=defaults,
        policies=policies,
        updated_at=parse_ts(doc.get('updated_at'), _EPOCH) if doc.get('updated_at') else None,
        updated_by=doc.get('updated_by'),
    )
    refresh = dict(doc['cluster_refresh']) if doc.get('cluster_refresh') else None
    return settings, refresh


def parse_keymap(snap: Snapshot, project: ProjectSrc) -> KeymapSrc | None:
    index = project.legacy_indexes.get('configs')
    doc = snap.index_docs(index or '').get(legacy.KEYMAP_DOC_ID) if index else None
    if doc is None:
        return None
    return KeymapSrc(
        overrides=dict(doc.get('overrides') or {}),
        revision=max(1, int(doc.get('revision') or 1)),
        updated_at=parse_ts(doc.get('updated_at'), _EPOCH) if doc.get('updated_at') else None,
    )


def _version_number(value: Any) -> int:
    match = re.search(r'(\d+)$', str(value))
    return int(match.group(1)) if match else 1


def parse_view_state(
    snap: Snapshot, project: ProjectSrc, *, now: dt.datetime
) -> ViewStateSrc | None:
    viz_index = project.legacy_indexes.get('umap_viz_state')
    doc = snap.index_docs(viz_index or '').get(legacy.VIZ_STATE_DOC_ID) if viz_index else None
    if doc is None or 'n_points' not in doc:
        return None
    reducer_doc = snap.index_docs(project.legacy_indexes.get('umap_state', '')).get('current')
    return ViewStateSrc(
        projection_version=_version_number(doc.get('projection_version')),
        scope=str(doc.get('scope') or 'all'),
        cluster_id=doc.get('cluster_id'),
        n_points=int(doc['n_points']),
        n_components=int(doc.get('n_components') or 2),
        metric=str(doc.get('metric') or 'cosine'),
        fitted_at=parse_ts(doc.get('fitted_at'), now),
        reducer_b64=(reducer_doc or {}).get('reducer_b64'),
    )


# Files of the region-drain area that are project markers, not jobs: file name -> project_state key.
_MARKER_FILES = {
    'state.json': 'region_drain',
    'dependency_health.json': 'region_dependency_health',
}


def parse_markers(snap: Snapshot, project: ProjectSrc) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rel, key in _MARKER_FILES.items():
        file = snap.file(f'region_drain:{project.slug}', rel)
        if file is not None:
            out[key] = file.json()
    return out


# --- one project, and the deployment-wide documents -------------------------------------------


def parse_project_data(snap: Snapshot, project: ProjectSrc, *, now: dt.datetime) -> ProjectData:
    configs_index = project.legacy_indexes.get('configs', '')
    docs = snap.index_docs(configs_index)
    activations = parse_activations(docs, now=now)
    settings, refresh = parse_settings(snap, project)
    return ProjectData(
        markers=parse_markers(snap, project),
        project=project,
        registry=parse_registry(snap, project, now=now),
        configs=parse_config_docs(docs, project.slug, _referenced(activations), now=now),
        activations=tuple(activations),
        settings=settings,
        keymap=parse_keymap(snap, project),
        cluster_refresh=refresh,
        view_state=parse_view_state(snap, project, now=now),
    )


def parse_global(snap: Snapshot, *, now: dt.datetime) -> GlobalData:
    docs = snap.index_docs(legacy.GLOBAL_CONFIGS_INDEX)
    probes: dict[str, list[ProbeSrc]] = {}
    for doc in docs.values():
        if doc.get('doc_type') != 'vlm_probe' or '@' not in str(doc.get('probe_key', '')):
            continue
        name, _, rev = str(doc['probe_key']).rpartition('@')
        record = (doc.get('body') or {}).get('record') or {}
        probes.setdefault(name, []).append(
            ProbeSrc(
                name=name,
                revision=int(rev),
                probed_at=parse_ts(doc.get('probed_at') or record.get('probed_at'), now),
                ok=bool(record.get('ok', False)),
                result=doc.get('body') or {},
            )
        )
    endpoints = tuple(
        VlmEndpointSrc(
            doc=config, probes=tuple(sorted(probes.get(config.name, []), key=lambda p: p.revision))
        )
        for config in parse_config_docs(docs, None, set(), now=now)
        if config.kind == 'vlm_endpoint'
    )
    desired = docs.get(legacy.LOCAL_DESIRED_DOC_ID)
    return GlobalData(
        endpoints=endpoints,
        local_vlm_desired=dict(desired) if desired else None,
    )


def parse_all(
    snap: Snapshot, *, now: dt.datetime
) -> tuple[GlobalData, list[ProjectData], list[RetiredSlug]]:
    live, retired, revision = parse_projects(snap, now=now)
    global_data = parse_global(snap, now=now)
    projects = [parse_project_data(snap, p, now=now) for p in live]
    return (
        GlobalData(global_data.endpoints, global_data.local_vlm_desired, revision),
        projects,
        retired,
    )


def dumps(value: Any) -> str:
    """Canonical JSON, for checksums."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'))
