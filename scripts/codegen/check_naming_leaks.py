#!/usr/bin/env python3
"""Pre-commit guard: no private-company or domain-vendor vocabulary leaks
into the public tree.

Three independent scans, each a single ``git grep -n -I -P`` pass over the
whole tracked tree:

* **Scan A** — hard leak scan, zero tolerance. Company/product initials and
  filesystem paths that must never appear in the public repo at all
  (``reference``, ``legacy_``/``LEGACY_``/``Legacy[A-Z]``, ``/legacy`` prefix, ``Provider``,
  ``example-org``, private mount paths, ...).
* **Scan B** — domain-vocabulary scan (vendor model names the generic
  plumbing must not hardcode: ``gemma``, ``lpr``, ``sam_worker``, ``v6``,
  ``plate_*`` field literals, ``hdd_source``, and any spelling of
  "license plate" used outside the public reference example).
* **Scan C** — private class-registry vocabulary (domain vehicle class
  names that must not leak into generic code or test fixtures).

Each scan's raw hits are filtered through the allowlist
(``scripts/codegen/naming_leak_allowlist.txt``): every hit either matches an
allowlist entry (and is dropped) or is reported and fails the check. New,
deliberate hits are allowlisted explicitly, with a reason -- this file
never grows an exemption automatically.

Run directly: ``python3 scripts/codegen/check_naming_leaks.py``.
Wired into ``.pre-commit-config.yaml`` as a ``language: system``,
``pass_filenames: false``, ``always_run: true`` hook (like its sibling
``check_no_literal_region_fields.py``), so it always scans the whole tree
regardless of which files are staged.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = REPO_ROOT / 'scripts' / 'codegen' / 'naming_leak_allowlist.txt'

# Paths excluded from `git grep` entirely rather than allowlisted, for files
# where matching almost the whole file line-by-line would be noise. Empty
# today; kept as the mechanism for the next such file.
_ALWAYS_EXCLUDE_PATHS: tuple[str, ...] = ()

SCAN_A = (
    r'reference|Reference|REFERENCE|\blegacy_|\bLEGACY_|\bLegacy[A-Z]|/legacy\b|Provider|provider|'
    r'example-org|/data/archive|/data|domain|workstation|host\.local'
)
SCAN_B = (
    r'(?i)gemma|(?<![a-z])lpr|nanov11|sam_worker|sam_drain|segment_plate|'
    r'Sam3Client|SAM3_(URL|URLS|HTTPX|SKIP)|(?<![a-z])v6(?![0-9])|'
    r'(?<![a-z_])plates?_|hdd_source|(?i)license[ _-]?plate'
)
SCAN_C = r'(?i)sportbike|cruiserbike|dirtbike|touring-adventurebikes|harley|\bhonda\b'

SCANS: dict[str, str] = {'A': SCAN_A, 'B': SCAN_B, 'C': SCAN_C}


class AllowlistEntry:
    """One allowlist line: either a whole-path exemption, a directory-prefix
    exemption (path ends with ``/``), or a ``path:regex`` line-level
    exemption (only lines in that path matching ``regex`` are allowed)."""

    def __init__(self, raw_path: str, raw_regex: str | None, reason: str, lineno: int) -> None:
        self.raw_path = raw_path
        self.is_dir_prefix = raw_path.endswith('/')
        self.regex = re.compile(raw_regex) if raw_regex else None
        self.reason = reason
        self.lineno = lineno
        self.used = False

    def matches_path(self, path: str) -> bool:
        if self.is_dir_prefix:
            return path.startswith(self.raw_path)
        return path == self.raw_path

    def allows(self, path: str, line_text: str) -> bool:
        if not self.matches_path(path):
            return False
        if self.regex is None:
            return True
        return bool(self.regex.search(line_text))


def load_allowlist(path: Path) -> list[AllowlistEntry]:
    if not path.exists():
        return []
    entries: list[AllowlistEntry] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if '#' not in line:
            raise ValueError(
                f'{path}:{lineno}: allowlist entry missing a " # reason" suffix: {raw!r}'
            )
        entry_part, _, reason = line.partition('#')
        entry_part = entry_part.strip()
        reason = reason.strip()
        if not entry_part:
            raise ValueError(f'{path}:{lineno}: allowlist entry has no path: {raw!r}')
        if not reason:
            raise ValueError(f'{path}:{lineno}: allowlist entry missing a reason: {raw!r}')
        if ':' in entry_part and not entry_part.endswith('/'):
            file_path, _, regex_src = entry_part.partition(':')
        else:
            file_path, regex_src = entry_part, ''
        entries.append(AllowlistEntry(file_path, regex_src or None, reason, lineno))
    return entries


def git_grep(pattern: str, repo_root: Path, exclude_paths: tuple[str, ...]) -> list[str]:
    pathspecs = [f':(exclude){p}' for p in exclude_paths]
    proc = subprocess.run(
        ['git', 'grep', '-n', '-I', '-P', pattern, '--', '.', *pathspecs],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f'git grep failed (exit {proc.returncode}): {proc.stderr}')
    return [line for line in proc.stdout.splitlines() if line]


def run_scans(
    repo_root: Path,
    allowlist_path: Path,
    exclude_paths: tuple[str, ...] = (),
    scans: dict[str, str] = SCANS,
) -> tuple[dict[str, list[str]], list[AllowlistEntry]]:
    """Run every scan against ``repo_root``, filtered through the allowlist
    at ``allowlist_path``. Returns ``(non_allowlisted_hits_by_scan,
    allowlist_entries)`` -- the caller inspects ``entry.used`` for the
    unused-allowlist-entry check. Pure of ``sys.exit``/printing so tests can
    call it directly against a synthetic repo."""
    allowlist = load_allowlist(allowlist_path)
    hits_by_scan: dict[str, list[str]] = {}
    for name, pattern in scans.items():
        hits: list[str] = []
        for line in git_grep(pattern, repo_root, exclude_paths):
            path, _, rest = line.partition(':')
            _lineno, _, text = rest.partition(':')
            entry = next((e for e in allowlist if e.allows(path, text)), None)
            if entry is not None:
                entry.used = True
                continue
            hits.append(line)
        hits_by_scan[name] = hits
    return hits_by_scan, allowlist


def main() -> int:
    hits_by_scan, allowlist = run_scans(REPO_ROOT, ALLOWLIST_PATH, _ALWAYS_EXCLUDE_PATHS)
    any_failed = False

    for name, hits in hits_by_scan.items():
        if hits:
            any_failed = True
            print(f'Scan {name}: {len(hits)} non-allowlisted hit(s):', file=sys.stderr)
            for h in hits:
                print(f'  {h}', file=sys.stderr)
            print(file=sys.stderr)

    if any_failed:
        print(
            'Fix the leak, or if this is a deliberate, reviewed exception, add it to\n'
            f'{ALLOWLIST_PATH.relative_to(REPO_ROOT)} with a reason.',
            file=sys.stderr,
        )
        return 1

    unused = [e for e in allowlist if not e.used]
    if unused and '--strict-allowlist' in sys.argv:
        print('Unused allowlist entries (safe to delete):', file=sys.stderr)
        for e in unused:
            print(f'  {ALLOWLIST_PATH.name}:{e.lineno}: {e.raw_path}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
