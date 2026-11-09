"""Only the project-delete code sets the flag the immutability triggers honour (RN-1).

``op.project_delete`` opens the guarded deletes of config revisions, ready partitions and
calibrations. Every writer of it is a way around those triggers, so the literal may appear in
application code in one module only; the migrations that define the checks are the other place.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'src/services/projects/delete.py'}
MIGRATIONS = 'src/db/migrations/'
FLAG = 'op.project_delete'


def _files_naming_the_flag() -> set[str]:
    found = set()
    for base in ('src', 'scripts'):
        for path in (ROOT / base).rglob('*.py'):
            relative = path.relative_to(ROOT).as_posix()
            if not relative.startswith(MIGRATIONS) and FLAG in path.read_text(encoding='utf-8'):
                found.add(relative)
    return found


def test_the_flag_is_named_by_the_delete_module_only() -> None:
    assert _files_naming_the_flag() == ALLOWED
