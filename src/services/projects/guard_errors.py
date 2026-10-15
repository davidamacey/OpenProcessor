"""The project guard's refusal exceptions and the counter of refused requests.

Split out of ``guard.py``: :class:`CrossProjectAccess`,
:class:`ProjectReadOnly`, the refusal counter and ``_refuse``, which every
parsing and policy step of the guard raises through.
"""

from __future__ import annotations

from prometheus_client import Counter

from src.config.project_context import try_current_project
from src.core.logging import get_logger


logger = get_logger(__name__)


class CrossProjectAccess(RuntimeError):  # noqa: N818 - name fixed by the projects plan (§2.4)
    """A request reaches an index the bound project does not own, or has a
    shape the guard does not allow."""


class ProjectReadOnly(RuntimeError):  # noqa: N818 - name fixed by the projects plan (§2.4)
    """A write was attempted while bound read-only (e.g. to an archived
    project, or a combine source read via ``bind_project(..., read_only=True)``)."""


OP_CROSS_PROJECT_ACCESS = Counter(
    'op_cross_project_access_total',
    'OpenSearch requests the project guard refused.',
    ['bound', 'target'],
)


_cross_project_access_count = 0


def cross_project_access_count() -> int:
    """Process-wide count of refused requests (the same events the
    ``op_cross_project_access_total`` counter records, unlabelled)."""
    return _cross_project_access_count


def _refuse(reason: str, *, target: str = '') -> CrossProjectAccess:
    global _cross_project_access_count  # noqa: PLW0603 - process-wide rejection counter

    bound = try_current_project()
    slug = bound.record.slug if bound is not None else '-'
    _cross_project_access_count += 1
    OP_CROSS_PROJECT_ACCESS.labels(bound=slug, target=target or '-').inc()
    logger.error('cross_project_access', bound=slug, target=target, reason=reason)
    return CrossProjectAccess(f"project guard refused a request (bound '{slug}'): {reason}")
