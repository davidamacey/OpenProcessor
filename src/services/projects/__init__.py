"""Project registry: the Postgres-backed in-process snapshot of every project,
and startup bootstrap of the ``default`` project row.

See the project-isolation design §2/§4.
"""

from __future__ import annotations

from src.services.projects.bootstrap import bootstrap_default_project
from src.services.projects.registry import ProjectRegistry, get_project_registry


__all__ = [
    'ProjectRegistry',
    'bootstrap_default_project',
    'get_project_registry',
]
