"""Untrusted XML: no DTD, no entities, no XXE."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.native_fixtures import voc_object, write_voc
from src.services.curation.dataset_import.safe_xml import parse_xml_bytes
from src.services.curation.dataset_import.voc import scan_voc


if TYPE_CHECKING:
    from pathlib import Path

_XXE = '<!DOCTYPE a [<!ENTITY e SYSTEM "file:///etc/passwd">]><a>&e;</a>'


def test_external_entity_is_refused() -> None:
    with pytest.raises(ValueError, match='not allowed'):
        parse_xml_bytes(_XXE.encode())


def test_billion_laughs_is_refused_fast() -> None:
    decls = '<!ENTITY l0 "lol">' + ''.join(
        f'<!ENTITY l{i} "&l{i - 1};&l{i - 1};&l{i - 1};&l{i - 1};">' for i in range(1, 10)
    )
    start = time.monotonic()
    with pytest.raises(ValueError, match='not allowed'):
        parse_xml_bytes(f'<!DOCTYPE a [{decls}]><a>&l9;</a>'.encode())
    assert time.monotonic() - start < 1.0


@pytest.mark.parametrize('doc', ['<!doctype a><a/>', '<! DOCTYPE a><a/>', '<!  ENTITY e "x"><a/>'])
def test_case_and_whitespace_variants_are_refused(doc: str) -> None:
    with pytest.raises(ValueError, match='not allowed'):
        parse_xml_bytes(doc.encode())


def test_a_utf16_doctype_is_refused_as_not_utf8() -> None:
    with pytest.raises(ValueError, match='UTF-8'):
        parse_xml_bytes(_XXE.encode('utf-16'))


def test_a_bom_prefixed_document_parses() -> None:
    root = parse_xml_bytes(b'\xef\xbb\xbf<annotation><filename>a.jpg</filename></annotation>')
    assert root.tag == 'annotation'
    assert root.findtext('filename') == 'a.jpg'


def test_malformed_xml_is_a_value_error() -> None:
    with pytest.raises(ValueError, match=r'.'):
        parse_xml_bytes(b'<annotation><unclosed></annotation>')


def test_scan_voc_reports_an_xxe_file_without_leaking_the_target(tmp_path: Path) -> None:
    write_voc(tmp_path, {'a': [voc_object('dog', (1, 1, 50, 50))]})
    (tmp_path / 'Annotations/a.xml').write_text(_XXE)
    scan = scan_voc(tmp_path)
    assert scan.entries == []
    issues = {i.code: i for i in scan.issues.issues()}
    assert 'voc_xml_invalid' in issues
    assert 'root:' not in repr(issues['voc_xml_invalid'].samples)
