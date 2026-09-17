#!/usr/bin/env python3
"""Pre-commit regression guard: no `'plate_...'` literals in ported files.

``RegionFields`` (``src/config/region_fields.py``) is the single source
of truth for OpenSearch region field names — see
``docs/design/oss_genericization_phase2_plan.md`` §3.2. On the working
branch a ``'plate_...'``/``"plate_..."`` string literal is *always* a
mistake: it means a file was copied across from the reference tree
without being genericized to read fields via ``RegionFields``.

Run by the ``check-no-literal-region-fields`` pre-commit hook, which
passes every changed ``*.py`` file under ``src/``, ``scripts/`` and
``tests/`` as a positional argument (same wiring style as
``check_file_size.py``). Of those, this script only actually checks
files that fall under ``PORTED_PATHS`` — a growing allowlist of
already-ported paths (§3.2 "Per-chunk enforcement guard"). It starts
empty in Chunk 0; each later wave appends its newly-ported paths in the
same commit that ports them. This gives a ratchet: once a module is
ported, it can never regress to hardcoding a `plate_*` literal again.

Two hardcoded exemptions (never driven by ``PORTED_PATHS``):
- ``src/config/region_fields.py`` — its docstrings legitimately name
  `plate_*` as the illustrative override example.
- ``tests/curation/test_region_fields.py`` — the overridability fixture
  legitimately constructs a `plate_*`-named instance.

Plus a line-level skip for Pydantic attribute declarations of the shape
``plate_foo: ...`` (matching ``^\\s*plate_[a-z_]+\\s*:``) — those are
the frozen HTTP wire contract with the labeler frontend (see
``docs/design/labeler_api_contract.md``), not an OpenSearch field
reference, and are explicitly out of ``RegionFields``' scope.

Run manually: `python3 scripts/codegen/check_no_literal_region_fields.py <files...>`
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Growing allowlist of paths (files or directory prefixes, POSIX,
# relative to the repo root) that have been ported to the `curation`
# namespace and are expected to be free of `plate_*` OpenSearch-field
# literals. Starts empty in Chunk 0 (scaffolding only, nothing ported
# yet). Each later wave appends its newly-ported paths here in the same
# commit that ports them — see §3.2 "Per-chunk enforcement guard".
#
# Chunk 1 (foundations). Note: `test_region_fields_mapping_
# coverage.py` is deliberately NOT listed here — like
# `test_region_fields.py`, it legitimately constructs a `plate_*`-named
# RegionFields instance to prove overridability (§3.2).
PORTED_PATHS: tuple[str, ...] = (
    # commit (a) — OpenSearch client
    'src/clients/curation_opensearch.py',
    'tests/curation/test_curation_opensearch.py',
    # commit (b) — router `_common` foundations
    'src/routers/curation/_common.py',
    'src/routers/curation/__init__.py',
    'tests/curation/test_ensure_indexes.py',
)

# Hardcoded exemptions — never touched by PORTED_PATHS growth.
_FULLY_EXEMPT_FILES = frozenset(
    {
        'src/config/region_fields.py',
        'tests/curation/test_region_fields.py',
    }
)

_LITERAL_RE = re.compile(r"""['"](plate_[a-z_]+)['"]""")
_PYDANTIC_ATTR_RE = re.compile(r'^\s*plate_[a-z_]+\s*:')


def _is_ported(rel_posix: str) -> bool:
    return any(
        rel_posix == prefix or rel_posix.startswith(prefix.rstrip('/') + '/')
        for prefix in PORTED_PATHS
    )


def _is_exempt(rel_posix: str) -> bool:
    if rel_posix in _FULLY_EXEMPT_FILES:
        return True
    return rel_posix.startswith('docs/')


def _scan_file(path: Path) -> list[tuple[int, str]]:
    violations: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return violations
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _PYDANTIC_ATTR_RE.match(line):
            continue
        if _LITERAL_RE.search(line):
            violations.append((lineno, line.strip()))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    exit_code = 0

    for path in args.files:
        try:
            rel_posix = path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            rel_posix = path.as_posix()

        if not _is_ported(rel_posix) or _is_exempt(rel_posix):
            continue

        violations = _scan_file(path)
        if violations:
            exit_code = 1
            sys.stderr.write(f'{rel_posix}:\n')
            for lineno, line in violations:
                sys.stderr.write(f'  {lineno}: {line}\n')

    if exit_code:
        sys.stderr.write(
            "\nERROR: 'plate_...' literal(s) found in ported curation "
            'code. Route field access through a RegionFields instance '
            'instead (src/config/region_fields.py). See '
            'docs/design/oss_genericization_phase2_plan.md §3.2.\n'
        )
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
