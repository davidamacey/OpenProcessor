"""The embedding model catalog: ``examples/embedding/catalog.tsv`` synced into Postgres.

One file, one reader (the shape of ``examples/vlm/catalog.tsv``). A synced row is immutable:
a changed input spec, weights or dimension is a new version, so a file that disagrees with a
stored row is refused, never applied. Whether a model can back a space is not stored: it is
computed from the engines that are ready (``availability``) and fails closed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.db.repos import embedding_spaces as repo
from src.db.session import sync_deployment_tx
from src.domain.embedding_spaces import MODEL_KEY_RE, input_spec_sha256
from src.schemas.models import EncoderInputSpec, EngineInputContract


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.orm import Session

DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[3] / 'examples' / 'embedding' / 'catalog.tsv'
)
CREATED_BY = 'catalog'

#: The first non-comment line names these columns; new ones are only ever appended.
COLUMNS = (
    'model_key',
    'version',
    'modalities',
    'native_dim',
    'matryoshka_dims',
    'distance',
    'input_spec',
    'weights_ref',
    'weights_sha256',
    'precision',
    'triton_image_model',
    'triton_text_model',
    'licence',
    'status',
)


class CatalogError(ValueError):
    """The catalog file is malformed, or disagrees with a synced (immutable) row."""


def _ints(raw: str) -> tuple[int, ...]:
    return tuple(int(p) for p in raw.split(',') if p.strip())


def _entry(row: dict[str, str]) -> repo.NewModelVersion:
    key = row['model_key'].strip()
    if not MODEL_KEY_RE.match(key):
        msg = f'catalog: {key!r} is not a valid model key'
        raise CatalogError(msg)
    spec = EncoderInputSpec.model_validate(json.loads(row['input_spec']))
    canonical = spec.model_dump(mode='json', exclude_none=True)
    modalities = tuple(p.strip() for p in row['modalities'].split(',') if p.strip())
    return repo.NewModelVersion(
        model_key=key,
        version=int(row['version']),
        modalities=modalities,
        native_dim=int(row['native_dim']),
        matryoshka_dims=_ints(row['matryoshka_dims']),
        distance=row['distance'].strip(),
        input_spec=canonical,
        input_spec_sha256=input_spec_sha256(canonical),
        weights_ref=row['weights_ref'].strip(),
        weights_sha256=row['weights_sha256'].strip(),
        precision=row['precision'].strip(),
        triton_image_model=row['triton_image_model'].strip() or None,
        triton_text_model=row['triton_text_model'].strip() or None,
        licence=row['licence'].strip(),
        status=row['status'].strip(),
        created_by=CREATED_BY,
    )


def load_catalog(path: Path | str | None = None) -> list[repo.NewModelVersion]:
    """Every catalog row, file order."""
    resolved = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    lines = [
        line
        for line in resolved.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.startswith('#')
    ]
    reader = csv.DictReader(lines, delimiter='\t', restval='')
    missing = set(COLUMNS) - set(reader.fieldnames or ())
    if missing:
        msg = f'{resolved}: catalog is missing column(s) {sorted(missing)}'
        raise CatalogError(msg)
    return [_entry(row) for row in reader]


#: Columns that identify what a version *is*; the status alone may change.
_IMMUTABLE = (
    'modalities',
    'native_dim',
    'matryoshka_dims',
    'distance',
    'input_spec',
    'input_spec_sha256',
    'weights_ref',
    'weights_sha256',
    'precision',
    'triton_image_model',
    'triton_text_model',
    'licence',
)


@dataclass(frozen=True)
class CatalogSyncResult:
    inserted: tuple[str, ...]
    status_changed: tuple[str, ...]


def sync_catalog(session: Session, path: Path | str | None = None) -> CatalogSyncResult:
    """Insert the catalog rows that are missing and move a changed ``status``.

    Runs in its own ``deployment_tx`` (the session must have no transaction open) and commits
    on exit. A row whose immutable columns differ from the stored one raises ``CatalogError``
    before anything is written: the change belongs in a new version.
    """
    entries = load_catalog(path)
    inserted: list[str] = []
    status_changed: list[str] = []
    with sync_deployment_tx(session, 'embedding catalog sync'):
        for entry in entries:
            ident = f'{entry.model_key}_v{entry.version}'
            stored = repo.get_model_version(session, entry.model_key, entry.version)
            if stored is None:
                repo.insert_model_version(session, entry)
                inserted.append(ident)
                continue
            differs = [
                c for c in _IMMUTABLE if _norm(getattr(stored, c)) != _norm(getattr(entry, c))
            ]
            if differs:
                msg = (
                    f'catalog row {ident} differs from the stored version in {differs}; '
                    'a changed model is a new version'
                )
                raise CatalogError(msg)
            if stored.status != entry.status:
                repo.set_model_version_status(session, stored.id, entry.status)
                status_changed.append(ident)
    return CatalogSyncResult(tuple(inserted), tuple(status_changed))


def _norm(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str | None = None


def availability(
    version: repo.ModelVersionRow, ready_engines: Mapping[str, Mapping[str, Any]]
) -> Availability:
    """Can this version back a space now? Fail closed: a missing or different engine says no.

    ``ready_engines`` maps a Triton model name to the ``input_contract`` of its ready engine.
    The image part of the version's ``input_spec`` must equal that contract exactly.
    """
    if version.status != 'available':
        return Availability(False, f'the model version is {version.status}')
    for name in (version.triton_image_model, version.triton_text_model):
        if name is not None and name not in ready_engines:
            return Availability(False, f'Triton model {name!r} has no ready engine (not installed)')
    if version.triton_image_model is not None:
        want = EncoderInputSpec.model_validate(version.input_spec).image
        have = EngineInputContract.model_validate(ready_engines[version.triton_image_model])
        if want != have:
            return Availability(
                False,
                f'input_spec image differs from the input_contract of {version.triton_image_model!r}',
            )
    return Availability(True)
