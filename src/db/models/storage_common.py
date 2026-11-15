"""Column helpers shared by the storage-record models (15a section 1).

Every project table carries ``project_id`` and a foreign-key index per referencing column
(``ix_<table>_<column>``, partial ``WHERE <column> IS NOT NULL`` when nullable). The migration
``0010_storage_record`` writes the same names; ``tests/storage`` compares both with the plan.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import REAL


# real[4] boxes and real[] polygons; PostgreSQL ``real`` is float4 (``REAL``)
RealArray = ARRAY(REAL)


def project_column() -> Mapped[uuid.UUID]:
    """``project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE`` (15a "PROJECT")."""
    return mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))


def fk_index(table: str, column: str, *, partial: bool = False) -> Index:
    where: Any = text(f'{column} IS NOT NULL') if partial else None
    return Index(f'ix_{table}_{column}', column, postgresql_where=where)


def bbox_check(column: str) -> str:
    return f'{column} IS NULL OR cardinality({column}) = 4'
