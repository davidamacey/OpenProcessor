"""Project registry: the ``op_projects`` index, its in-process snapshot,
and startup bootstrap of the ``default`` project record.

See ``docs/design/openprocessor_internal/projects_plan.md`` §2/§4.
"""

from __future__ import annotations

from src.services.projects.bootstrap import bootstrap_default_project
from src.services.projects.registry import ProjectRegistry, get_project_registry


__all__ = [
    'ProjectRegistry',
    'bootstrap_default_project',
    'get_project_registry',
]
