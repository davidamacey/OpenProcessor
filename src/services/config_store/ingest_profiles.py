"""Ingest detector profile CRUD (the ``ingest_profile`` kind / axis). Mirrors
:mod:`src.services.config_store.open_vocab`: a profile is either stored in the
project's config index or a read-only shipped template
(``examples/ingest_profiles/``). The deployment's ``OP_INGEST_*`` profile is the
default a project with no active profile runs; it is not a stored profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.services.config_store.documents import ActiveConflictError, RevisionConflictError
from src.services.config_store.kind_records import KindRecord, StoredKind


@dataclass(frozen=True)
class IngestProfileRecord(KindRecord):
    pass


_KIND = StoredKind(
    IngestProfileRecord,
    kind='ingest_profile',
    axis='ingest_profile',
    etag_prefix='ingest_profile',
    templates_dir=Path(__file__).resolve().parents[3] / 'examples' / 'ingest_profiles',
    stored=lambda cur: cur.ingest_profiles,
    active=lambda cur: cur.active_ingest_profile,
)

load_template = _KIND.load_template
template_names = _KIND.template_names
all_known_names = _KIND.all_known_names
build_record = _KIND.build_record
get_revision_record = _KIND.get_revision_record
list_revisions = _KIND.list_revisions
save_profile = _KIND.save
delete_profile = _KIND.delete
activate_profile = _KIND.activate
rollback_profile = _KIND.rollback

__all__ = [
    'ActiveConflictError',
    'IngestProfileRecord',
    'RevisionConflictError',
    'activate_profile',
    'all_known_names',
    'build_record',
    'delete_profile',
    'get_revision_record',
    'list_revisions',
    'load_template',
    'rollback_profile',
    'save_profile',
    'template_names',
]
