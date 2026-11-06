"""``class_registry.json`` is written only into export snapshots and training pins (2B-6).

The registry lives in PostgreSQL. The file under ``OP_PROJECTS_DATA_ROOT/<slug>/`` is no longer
read at runtime: its path is reached only from the project-resource plumbing that still names it
(lane 2A removes the resource) and from the 0.6.0 upgrade (lane 2E).
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_PLUMBING = {
    'src/config/curation.py',
    'src/config/curation_accessors.py',
    'src/config/projects.py',
    'src/routers/curation/_project_views.py',
    'src/services/projects/registry.py',
    'src/services/projects/delete_paths.py',
    # writes the 0.6.0 project documents the upgrade fixture is made of
    'scripts/db/make_v060_fixture.py',
}


def test_no_module_reads_the_per_project_class_registry_path() -> None:
    users = {
        p.relative_to(REPO_ROOT).as_posix()
        for root in ('src', 'scripts')
        for p in (REPO_ROOT / root).rglob('*.py')
        if 'class_registry_path' in p.read_text(encoding='utf-8')
    }
    upgrade = {u for u in users if u.startswith('src/services/upgrade/')}
    assert users - upgrade <= PATH_PLUMBING, sorted(users - upgrade - PATH_PLUMBING)
