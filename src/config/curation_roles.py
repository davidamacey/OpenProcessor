"""Logical roles of the curation OpenSearch indexes.

``IndexRole`` names a stable role; :func:`index_name` resolves the
deployment's configured index name for a role from a ``CurationConfig``.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.config.curation import CurationConfig


class IndexRole(str, Enum):  # noqa: UP042  # (str, Enum) keeps str() and format() as Class.MEMBER; StrEnum would change both
    """Logical role of a curation OpenSearch index.

    Unlike the reference private string enum, member *values* are
    stable role identifiers, not deployment-specific index names — the
    actual index name for a role is resolved via :func:`index_name`
    against a :class:`CurationConfig` instance, so it can be
    overridden per deployment without touching this enum.
    """

    IMAGES = 'images'
    ITEMS = 'items'
    # Transitional: the job lease documents (``src/clients/job_lease.py``) have no other
    # home until lane 6A deletes the B1 leases; they are the only documents it holds.
    LEASES = 'leases'


INDEX_ROLE_ATTR: dict[IndexRole, str] = {
    IndexRole.IMAGES: 'images_index',
    IndexRole.ITEMS: 'items_index',
    IndexRole.LEASES: 'leases_index',
}


def index_name(cfg: CurationConfig, role: IndexRole) -> str:
    """Resolve the configured OpenSearch index name for a logical role.

    Replaces reference call sites of the shape ``<PrivateIndexEnum>.X.value`` —
    those hardcoded a *value*; this looks the value up on ``cfg`` so it
    is deployment-overridable.
    """
    typed: str = getattr(cfg, INDEX_ROLE_ATTR[role])
    return typed
