"""build_record / get_revision_record of the stored-or-template kinds (ingest profiles,
open-vocabulary sets): stored wins, a pinned revision never falls back to a template, a
template is read-only with a content-digest etag, and the active pair flags the record."""

from __future__ import annotations

import types
from typing import Any

import pytest

from src.services.config_store import ingest_profiles, kind_records, open_vocab


_STORED = types.SimpleNamespace(
    revision=3,
    body={'k': 1},
    description='d',
    created_at='c',
    updated_at='u',
    cloned_from=None,
)


def _store(monkeypatch: pytest.MonkeyPatch, module: Any, attr: str, active_attr: str, active: Any):
    current = types.SimpleNamespace(
        **{attr: {'mine': _STORED}, active_attr: active},
    )
    monkeypatch.setattr(
        kind_records, 'get_config_store', lambda: types.SimpleNamespace(current=current)
    )


@pytest.mark.parametrize(
    ('module', 'attr', 'active_attr', 'prefix', 'template'),
    [
        (ingest_profiles, 'ingest_profiles', 'active_ingest_profile', 'ingest_profile', 'coco80'),
        (open_vocab, 'open_vocab_sets', 'active_open_vocab', 'open_vocab', 'street_objects'),
    ],
)
def test_build_record_stored_pinned_and_template(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    attr: str,
    active_attr: str,
    prefix: str,
    template: str,
) -> None:
    _store(monkeypatch, module, attr, active_attr, None)
    rec = module.build_record('mine')
    assert (rec.source, rec.read_only, rec.revision, rec.etag) == (
        'stored',
        False,
        3,
        f'{prefix}:mine:3',
    )
    assert (rec.description, rec.body, rec.active, rec.active_revision) == (
        'd',
        {'k': 1},
        False,
        None,
    )
    # a pinned revision other than the stored one never falls back to anything
    assert module.build_record('mine', revision=2) is None
    assert module.build_record('mine', revision=3).revision == 3

    tpl = module.build_record(template)
    assert (tpl.source, tpl.read_only, tpl.revision) == ('template', True, None)
    assert tpl.etag.startswith(f'{prefix}:{template}:')
    assert 'name' not in tpl.body
    assert module.build_record(template, revision=1) is None
    assert module.build_record('absent') is None


def test_active_pair_flags_the_matching_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    _store(monkeypatch, ingest_profiles, 'ingest_profiles', 'active_ingest_profile', 'placeholder')
    monkeypatch.setattr(kind_records.templates, 'active_ref', lambda _a: ('mine', 3))
    rec = ingest_profiles.build_record('mine')
    assert rec is not None
    assert (rec.active, rec.active_revision) == (True, 3)
    monkeypatch.setattr(kind_records.templates, 'active_ref', lambda _a: ('mine', 2))
    rec = ingest_profiles.build_record('mine')
    assert rec is not None
    assert (rec.active, rec.active_revision) == (False, 2)
    monkeypatch.setattr(kind_records.templates, 'active_ref', lambda _a: ('other', 3))
    rec = ingest_profiles.build_record('mine')
    assert rec is not None
    assert (rec.active, rec.active_revision) == (False, None)
