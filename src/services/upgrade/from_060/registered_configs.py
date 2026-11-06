"""Built-in configurations a 0.6.0 project activated, stored as documents.

In 0.6.0 a project could activate a built-in or file prompt pack, or an env or registered region
profile, by name alone: the activation had no revision and the body came from the code, the files
and the environment. 0.7.0 gives every activation a stored document to point at, and activating a
built-in stores its body as a first revision (``activation_apply.registered_for`` resolves the body,
``Registered`` carries it). The upgrade does the same once per project and name, with the body the
environment it runs in resolves, and points the activation events at revision 1.

A body that does not resolve here stops the upgrade (:class:`UnresolvedBuiltinError`): dropping the
activation would silently change what the project runs.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from src.services.config_store.activation_apply import registered_for
from src.services.upgrade.from_060.model import (
    ConfigDocSrc,
    ConfigRevisionSrc,
    UnresolvedBuiltinError,
)


if TYPE_CHECKING:
    import datetime as dt

    from src.services.upgrade.from_060.model import ActivationSrc, ProjectData

REGISTERED_REVISION = 1


def _builtin_refs(events: tuple[ActivationSrc, ...]) -> dict[tuple[str, str, str], dt.datetime]:
    """(axis, kind, name) of every built-in an event names, with the time it first did."""
    refs: dict[tuple[str, str, str], dt.datetime] = {}
    for e in sorted(events, key=lambda e: e.at):
        for name, revision in ((e.to_name, e.to_revision), (e.from_name, e.from_revision)):
            if name is not None and revision is None:
                refs.setdefault((e.axis, e.kind, name), e.at)
    return refs


def _document(pd: ProjectData, axis: str, kind: str, name: str, at: dt.datetime) -> ConfigDocSrc:
    if any(d.kind == kind and d.name == name for d in pd.configs):
        msg = (
            f'project {pd.project.slug!r} activated the built-in {axis} configuration {name!r}, but '
            f'it also holds a stored {kind} of that name; rename or delete the stored one first'
        )
        raise UnresolvedBuiltinError(msg)
    registered = registered_for(axis, name)  # type: ignore[arg-type]  # axis is a ConfigAxis here
    if registered is None:
        msg = (
            f'project {pd.project.slug!r} activated {name!r} on the {axis} axis with no stored '
            'revision, and no built-in, file or environment configuration of that name resolves '
            'where the upgrade runs; restore the 0.6.0 configuration (prompt pack files, region '
            'profile environment) and run the upgrade again'
        )
        raise UnresolvedBuiltinError(msg)
    return ConfigDocSrc(
        project_slug=pd.project.slug,
        kind=kind,
        name=name,
        description=registered.description,
        revisions=(
            ConfigRevisionSrc(
                revision=REGISTERED_REVISION, body=registered.body, created_at=at, cloned_from=None
            ),
        ),
        created_at=at,
        updated_at=at,
    )


def store_builtins(pd: ProjectData) -> ProjectData:
    """``pd`` with one stored document per built-in it activated and its events pointing at them."""
    refs = _builtin_refs(pd.activations)
    if not refs:
        return pd
    docs = tuple(_document(pd, axis, kind, name, at) for (axis, kind, name), at in refs.items())
    events = tuple(
        dataclasses.replace(
            e,
            to_revision=(
                REGISTERED_REVISION
                if e.to_name is not None and e.to_revision is None
                else e.to_revision
            ),
            from_revision=(
                REGISTERED_REVISION
                if e.from_name is not None and e.from_revision is None
                else e.from_revision
            ),
        )
        for e in pd.activations
    )
    return dataclasses.replace(pd, configs=(*pd.configs, *docs), activations=events)
