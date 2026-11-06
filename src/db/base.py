"""Declarative base and the constraint naming convention.

The convention is set before the first revision because constraint names cannot be changed
later without renaming every constraint (Alembic: "The Importance of Naming Constraints").
A ``CheckConstraint`` must carry an explicit name: the convention builds the full name from
the table and that name, which also keeps a name a later revision alters stable.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import DateTime, MetaData, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


NAMING_CONVENTION = {
    'ix': 'ix_%(column_0_label)s',
    'uq': 'uq_%(table_name)s_%(column_0_name)s',
    'ck': 'ck_%(table_name)s_%(constraint_name)s',
    'fk': 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s',
    'pk': 'pk_%(table_name)s',
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {
        str: Text(),
        uuid.UUID: Uuid(as_uuid=True),
        dt.datetime: DateTime(timezone=True),
        dict[str, Any]: JSONB(),
    }
