"""Shared reader/comparer for the ratchet baselines in ``scripts/codegen/baselines/``.

A baseline file holds the violations that existed when a guard landed. Every
line is ``key<TAB>value<TAB>reason`` (``value`` is an integer: a line count, a
complexity score, an occurrence count). The guard fails when:

* a violation is not in the baseline (new debt),
* a violation's value grew past its recorded value (the debt got worse),
* a recorded value is higher than reality (loosened baseline: lower it), or
* a baseline entry no longer has a violation (stale: delete the line).

So a baseline can only shrink. ``--tighten`` on a guard rewrites it for the last
two cases; it never adds a line and never raises a value.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Entry:
    value: int
    reason: str


def load(path: Path) -> dict[str, Entry]:
    """Parse ``path``; a missing file is an empty baseline. Raises ``ValueError`` on a bad line."""
    if not path.exists():
        return {}
    entries: dict[str, Entry] = {}
    for lineno, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        parts = raw.split('\t')
        if len(parts) != 3 or not parts[0] or not parts[2].strip():
            msg = f'{path}:{lineno}: expected "key<TAB>value<TAB>reason" with a reason: {raw!r}'
            raise ValueError(msg)
        try:
            value = int(parts[1])
        except ValueError:
            msg = f'{path}:{lineno}: value must be an integer: {parts[1]!r}'
            raise ValueError(msg) from None
        if parts[0] in entries:
            msg = f'{path}:{lineno}: duplicate key {parts[0]!r}'
            raise ValueError(msg)
        entries[parts[0]] = Entry(value, parts[2].strip())
    return entries


def compare(current: dict[str, int], baseline: dict[str, Entry], *, unit: str) -> list[str]:
    """Error lines for ``current`` (key -> value of every violation) against ``baseline``."""
    errors: list[str] = []
    for key in sorted(current):
        value = current[key]
        if key not in baseline:
            errors.append(f'{key}: {value} {unit} (new violation, not in the baseline)')
        elif value > baseline[key].value:
            errors.append(f'{key}: {value} {unit} grew past the baseline {baseline[key].value}')
        elif value < baseline[key].value:
            errors.append(
                f'{key}: {value} {unit} is below the baseline {baseline[key].value}; '
                'lower it (run the guard with --tighten)'
            )
    errors.extend(
        f'{key}: stale baseline entry, no violation left (delete the line or run --tighten)'
        for key in sorted(baseline)
        if key not in current
    )
    return errors


def tighten(path: Path, current: dict[str, int], baseline: dict[str, Entry]) -> int:
    """Rewrite ``path`` with lowered values and without stale entries. Returns lines changed."""
    changed = 0
    lines: list[str] = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        parts = raw.split('\t')
        line = raw
        if len(parts) == 3 and parts[0] in baseline:
            key = parts[0]
            if key not in current:
                changed += 1
                continue
            if current[key] < baseline[key].value:
                parts[1] = str(current[key])
                line = '\t'.join(parts)
                changed += 1
        lines.append(line)
    path.write_text('\n'.join(lines) + '\n' if lines else '', encoding='utf-8')
    return changed


def run_scan_ratchet(
    argv: list[str] | None,
    *,
    doc: str | None,
    baseline_path: Path,
    repo_root: Path,
    scan: Callable[[Path], tuple[dict[str, int], list[str]]],
    unit: str,
    unreadable: str,
    failure: str,
) -> int:
    """The shared ``main`` of the scan-and-compare ratchets: ``scan(root)`` returns
    ``(found, unreadable files)``; ``unreadable`` is the per-file message suffix; those errors
    stay under ``--tighten``."""
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument('--baseline', type=Path, default=baseline_path)
    parser.add_argument('--tighten', action='store_true', help='Lower baseline values, drop stale.')
    parser.add_argument('--root', type=Path, default=repo_root)
    args = parser.parse_args(argv)

    found, broken = scan(args.root)
    baseline = load(args.baseline)
    errors = [f'{rel}: {unreadable}' for rel in broken]
    errors += compare(found, baseline, unit=unit)
    if args.tighten:
        changed = tighten(args.baseline, found, baseline)
        sys.stderr.write(f'tightened {changed} baseline line(s)\n')
        errors = [e for e in errors if 'new violation' in e or 'grew past' in e or unreadable in e]
    if errors:
        sys.stderr.write(f'ERROR: {failure}\n')
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0
