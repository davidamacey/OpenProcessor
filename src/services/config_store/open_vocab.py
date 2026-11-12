"""Open-vocabulary prompt-set CRUD (the ``open_vocab_set`` kind /
``open_vocab`` axis). Mirrors :mod:`src.services.config_store.profiles`
without the env/registered sources: a set is either stored in the project's
config index or a read-only shipped template (``examples/open_vocab/``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.services.config_store import get_config_store
from src.services.config_store.documents import ActiveConflictError, RevisionConflictError
from src.services.config_store.kind_records import KindRecord, StoredKind


if TYPE_CHECKING:
    from src.services.detection.open_vocab_set import OpenVocabSet


@dataclass(frozen=True)
class OpenVocabRecord(KindRecord):
    pass


_KIND = StoredKind(
    OpenVocabRecord,
    kind='open_vocab_set',
    axis='open_vocab',
    etag_prefix='open_vocab',
    templates_dir=Path(__file__).resolve().parents[3] / 'examples' / 'open_vocab',
    stored=lambda cur: cur.open_vocab_sets,
    active=lambda cur: cur.active_open_vocab,
)

load_template = _KIND.load_template
template_names = _KIND.template_names
all_known_names = _KIND.all_known_names
build_record = _KIND.build_record
get_revision_record = _KIND.get_revision_record
list_revisions = _KIND.list_revisions
save_set = _KIND.save
delete_set = _KIND.delete
activate_set = _KIND.activate
rollback_set = _KIND.rollback


def active_open_vocab_set() -> OpenVocabSet | None:
    """The bound project's ACTIVATED set (the pinned revision's body), or
    ``None`` when none is active or the pinned body does not decode. Reads
    the process snapshot: callers that must see another worker's activation
    refresh the store first."""
    from src.services.detection.open_vocab_set import decode_open_vocab_set

    pinned = get_config_store().current.active_open_vocab_body
    if pinned is None:
        return None
    try:
        return decode_open_vocab_set(pinned.name, pinned.body)
    except ValueError:
        return None


__all__ = [
    'ActiveConflictError',
    'OpenVocabRecord',
    'RevisionConflictError',
    'activate_set',
    'active_open_vocab_set',
    'all_known_names',
    'build_record',
    'delete_set',
    'get_revision_record',
    'list_revisions',
    'load_template',
    'rollback_set',
    'save_set',
    'template_names',
]
