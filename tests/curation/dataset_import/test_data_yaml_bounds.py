"""A hostile ``data.yaml`` fails fast with a clean issue code, whichever
reader reaches it, and never expands past the file's own size."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
import yaml

from src.services.curation.dataset_import import yolo
from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.limits import MAX_YAML_DEPTH, MAX_YAML_NODES
from src.services.curation.dataset_import.safe_yaml import YamlTooComplexError, load_bounded_yaml


if TYPE_CHECKING:
    from pathlib import Path


def _bomb(depth: int, fan: int = 9) -> str:
    lines = ['a0: &a0 ["x","x","x","x","x","x","x","x","x"]']
    lines.extend(f'a{i}: &a{i} [' + ','.join([f'*a{i - 1}'] * fan) + ']' for i in range(1, depth))
    return '\n'.join(lines) + f'\ntrain: images/train\nnames: [*a{depth - 1}]\n'


def _codes(issues: IssueCollector) -> list[str]:
    return [i.code for i in issues.issues()]


def test_an_alias_bomb_is_a_clean_issue_not_a_hang(tmp_path: Path) -> None:
    (tmp_path / 'data.yaml').write_text(_bomb(12))
    issues = IssueCollector()
    started = time.monotonic()
    splits, names = yolo.discover_yolo(tmp_path, issues)
    assert time.monotonic() - started < 1.0
    assert (splits, names) == ({}, {})
    assert _codes(issues) == ['data_yaml_invalid']


def test_any_alias_is_refused() -> None:
    with pytest.raises(YamlTooComplexError):
        load_bounded_yaml('a: &x [1]\nb: *x\n')


def test_a_merge_key_is_refused_because_it_is_an_alias() -> None:
    with pytest.raises(YamlTooComplexError):
        load_bounded_yaml('base: &b {x: 1}\nchild:\n  <<: *b\n')


def test_a_plain_data_yaml_still_parses() -> None:
    doc = load_bounded_yaml('path: .\ntrain: images/train\nnames:\n  0: car\n  1: truck\n')
    assert doc['names'] == {0: 'car', 1: 'truck'}


def test_nesting_depth_is_capped_without_a_recursion_error(tmp_path: Path) -> None:
    (tmp_path / 'data.yaml').write_text('names: ' + '[' * 200_000 + ']' * 200_000)
    issues = IssueCollector()
    yolo.discover_yolo(tmp_path, issues)
    assert _codes(issues) == ['data_yaml_invalid']
    deepest = '[' * (MAX_YAML_DEPTH - 1) + ']' * (MAX_YAML_DEPTH - 1)
    assert load_bounded_yaml(f'names: {deepest}')['names']


def test_node_count_is_capped() -> None:
    with pytest.raises(YamlTooComplexError):
        load_bounded_yaml('names: [' + ','.join(['a'] * (MAX_YAML_NODES + 1)) + ']')


def test_a_non_scalar_class_name_is_invalid(tmp_path: Path) -> None:
    (tmp_path / 'data.yaml').write_text('train: images/train\nnames: [[a, b], c]\n')
    issues = IssueCollector()
    yolo.discover_yolo(tmp_path, issues)
    assert _codes(issues) == ['data_yaml_invalid']


def test_a_non_string_split_entry_is_invalid(tmp_path: Path) -> None:
    (tmp_path / 'data.yaml').write_text('train: [[a, b]]\nnames: [car]\n')
    issues = IssueCollector()
    yolo.discover_yolo(tmp_path, issues)
    assert _codes(issues) == ['data_yaml_invalid']


def test_a_document_one_level_past_the_depth_cap_is_refused() -> None:
    too_deep = '[' * MAX_YAML_DEPTH + ']' * MAX_YAML_DEPTH
    with pytest.raises(YamlTooComplexError, match='nested deeper'):
        load_bounded_yaml(f'names: {too_deep}')


@pytest.mark.parametrize(
    'text',
    [
        'train: images/train\nnames: [' + '9' * 5000 + ']\n',
        'train: images/train\nnames: [2001-13-45]\n',
        'train: 2001-02-30\nnames: [car]\n',
    ],
    ids=['over-long integer', 'impossible month', 'impossible day'],
)
def test_a_scalar_the_constructor_rejects_is_a_clean_issue(tmp_path: Path, text: str) -> None:
    (tmp_path / 'data.yaml').write_text(text)
    issues = IssueCollector()
    assert yolo.discover_yolo(tmp_path, issues) == ({}, {})
    assert _codes(issues) == ['data_yaml_invalid']
    with pytest.raises(yaml.YAMLError):
        load_bounded_yaml(text)
