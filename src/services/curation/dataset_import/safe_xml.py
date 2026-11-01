"""Parse untrusted XML with the standard library, refusing every DTD first.

Entities can only be declared inside a DOCTYPE, so a document with no
DOCTYPE has no external entity (XXE), no parameter entity and no entity
expansion. The text is decoded to ``str`` before the scan so an encoding
trick (UTF-16) cannot hide a ``<!DOCTYPE`` from it. Memory is bounded by the
caller's per-file byte cap.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # nosec B405  untrusted XML is parsed only by parse_xml_bytes, which refuses any DTD first


_DTD = re.compile(r'<!\s*(DOCTYPE|ENTITY)', re.IGNORECASE)


def parse_xml_bytes(raw: bytes) -> ET.Element:
    """Parse untrusted XML. Raises ``ValueError`` for non-UTF-8 input, any
    DOCTYPE/ENTITY declaration, or malformed XML."""
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('not UTF-8') from exc
    if _DTD.search(text):
        raise ValueError('DOCTYPE and ENTITY declarations are not allowed')
    try:
        return ET.fromstring(text)  # nosec B314  no DTD reaches the parser (refused above), so no entity can be declared or expanded
    except ET.ParseError as exc:
        raise ValueError(str(exc)[:200]) from exc


__all__ = ['parse_xml_bytes']
