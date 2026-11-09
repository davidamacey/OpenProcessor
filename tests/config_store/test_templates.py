"""The shared starter-template reader and active-ref reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.config_store import templates


if TYPE_CHECKING:
    from pathlib import Path


def test_template_names_sorted_by_stem_and_empty_for_missing_dir(tmp_path: Path) -> None:
    (tmp_path / 'b.json').write_text('{}')
    (tmp_path / 'a.json').write_text('{}')
    (tmp_path / 'skip.txt').write_text('x')
    assert list(templates.template_names(tmp_path)) == ['a', 'b']
    assert templates.template_names(tmp_path / 'absent') == {}


def test_load_template_returns_none_for_unknown_or_bad_json(tmp_path: Path) -> None:
    (tmp_path / 'ok.json').write_text('{"k": 1}')
    (tmp_path / 'bad.json').write_text('{not json')
    assert templates.load_template(tmp_path, 'ok') == {'k': 1}
    assert templates.load_template(tmp_path, 'bad') is None
    assert templates.load_template(tmp_path, 'missing') is None


def test_active_ref_reduces_none_off_and_pairs() -> None:
    assert templates.active_ref(None) == (None, None)
    assert templates.active_ref('off') == (None, None)
    assert templates.active_ref(('p', 3)) == ('p', 3)
