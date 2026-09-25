"""Parses a multipart/form-data request body captured by the stubbed
Playwright ``Stub`` fixture (``e2e/conftest.py``), for
``e2e/stubbed/test_ingest.py``'s multipart assertions.

``Stub._dispatch`` json-decodes ``request.post_data`` when possible and
falls back to the raw string otherwise — a multipart body always falls
into the string branch, so ``stub.calls`` records the raw multipart text
verbatim. This module turns that raw text back into named fields (and,
for the repeated ``images`` file parts, just their filenames — the tests
care about which files were sent, not their bytes).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class ParsedMultipart:
    fields: dict[str, str] = field(default_factory=dict)
    image_filenames: list[str] = field(default_factory=list)

    @property
    def image_paths(self) -> list[str]:
        """The decoded JSON list from the ``image_paths`` field."""
        raw = self.fields.get("image_paths")
        return json.loads(raw) if raw else []

    @property
    def source(self) -> str | None:
        return self.fields.get("source")


_DISPOSITION_RE = re.compile(
    r'Content-Disposition:\s*form-data;\s*name="([^"]+)"(?:;\s*filename="([^"]*)")?',
    re.IGNORECASE,
)


def parse_multipart(raw: str) -> ParsedMultipart:
    """Splits on the boundary line (the first line of ``raw``) and reads
    each part's Content-Disposition header plus body."""
    lines = raw.split("\r\n") if "\r\n" in raw else raw.split("\n")
    if not lines or not lines[0].startswith("--"):
        raise ValueError("not a multipart body (no leading boundary line)")
    boundary = lines[0]
    parts = raw.split(boundary)
    result = ParsedMultipart()
    for part in parts:
        part = part.strip("\r\n-")
        if not part:
            continue
        m = _DISPOSITION_RE.search(part)
        if not m:
            continue
        name, filename = m.group(1), m.group(2)
        # Body is everything after the blank line following the headers.
        body = part.split("\r\n\r\n", 1)
        if len(body) != 2:
            body = part.split("\n\n", 1)
        value = body[1].strip("\r\n") if len(body) == 2 else ""
        if filename is not None:
            result.image_filenames.append(filename)
        else:
            result.fields[name] = value
    return result
