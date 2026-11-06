"""Storage keys: relative, slash-separated, no escape from the root."""

from __future__ import annotations

import pytest

from src.clients.storage.keys import ROOT_NAMES, ParsedKey, make_key, parse_key


UNSAFE = 'unsafe storage|unknown storage root|max_side'


def test_round_trip() -> None:
    key = make_key('projects', 'cats', 'uploads', 'a.jpg')
    assert key == 'projects/cats/uploads/a.jpg'
    assert parse_key(key) == ParsedKey(root='projects', parts=('cats', 'uploads', 'a.jpg'))


def test_root_only_key() -> None:
    assert parse_key(make_key('models')) == ParsedKey(root='models', parts=())


@pytest.mark.parametrize('part', ['..', 'a/../b', '/abs', 'a\\b', '', '.', 'a//b'])
def test_make_key_refuses_unsafe_parts(part: str) -> None:
    with pytest.raises(ValueError, match=UNSAFE):
        make_key('projects', 'cats', part)


def test_make_key_refuses_unknown_root() -> None:
    with pytest.raises(ValueError, match=UNSAFE):
        make_key('etc', 'passwd')


@pytest.mark.parametrize(
    'key',
    ['', '/projects/a', 'projects/../x', 'projects//x', 'projects\\x', 'nope/x', 'projects/a/'],
)
def test_parse_key_refuses_unsafe_keys(key: str) -> None:
    with pytest.raises(ValueError, match=UNSAFE):
        parse_key(key)


def test_root_names_are_the_plan_roots() -> None:
    assert {'source', 'projects', 'models', 'cache', 'upgrade'} == ROOT_NAMES
