"""A tiny renderer for the plan's DDL, so every constraint gets the naming convention's name.

15a writes the tables as SQL. The migration keeps that form (a reviewer reads it against the plan)
and this module adds the names ``src.db.base.NAMING_CONVENTION`` would give: ``pk_<t>``,
``fk_<t>_<col>_<ref>``, ``ck_<t>_<name>``, and an explicit name for every unique key.

A column line may end ``-> ref(col) ACTION`` for a single-column foreign key; ``ACTION`` is
``CASCADE``, ``SET NULL`` or absent (``NO ACTION``). ``PROJECT`` is the plan's shorthand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from alembic import op


PROJECT = 'project_id uuid NOT NULL -> projects(id) CASCADE'
IDENTITY = 'GENERATED ALWAYS AS IDENTITY'

_REFERENCE = re.compile(
    r'^(?P<column>.*?) -> (?P<ref>\w+)\((?P<ref_column>\w+)\)(?: (?P<action>CASCADE|SET NULL))?$'
)


@dataclass(frozen=True)
class Constraint:
    kind: str  # 'pk', 'uq', 'ck' or 'fk'
    name: str
    body: str


def pk(*columns: str) -> Constraint:
    return Constraint('pk', '', ', '.join(columns))


def uq(name: str, *columns: str) -> Constraint:
    return Constraint('uq', name, ', '.join(columns))


def ck(name: str, expression: str) -> Constraint:
    return Constraint('ck', name, expression)


def fk(columns: tuple[str, ...], ref: str, ref_columns: tuple[str, ...]) -> Constraint:
    """A composite foreign key with NO ACTION."""
    return Constraint(
        'fk',
        '_'.join((columns[0], ref)),
        f'({", ".join(columns)}) REFERENCES {ref} ({", ".join(ref_columns)})',
    )


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[str, ...]
    constraints: tuple[Constraint, ...] = ()

    def create_sql(self) -> str:
        lines: list[str] = []
        references: list[str] = []
        for column in self.columns:
            match = _REFERENCE.match(column)
            if match is None:
                lines.append(column)
                continue
            name = match['column'].split()[0]
            lines.append(match['column'])
            on_delete = f' ON DELETE {match["action"]}' if match['action'] else ''
            references.append(
                f'CONSTRAINT fk_{self.name}_{name}_{match["ref"]} FOREIGN KEY ({name}) '
                f'REFERENCES {match["ref"]}({match["ref_column"]}){on_delete}'
            )
        lines += [self._render(c) for c in self.constraints] + references
        body = ',\n  '.join(lines)
        return f'CREATE TABLE {self.name} (\n  {body}\n)'

    def _render(self, constraint: Constraint) -> str:
        t = self.name
        if constraint.kind == 'pk':
            return f'CONSTRAINT pk_{t} PRIMARY KEY ({constraint.body})'
        if constraint.kind == 'uq':
            return f'CONSTRAINT {constraint.name} UNIQUE ({constraint.body})'
        if constraint.kind == 'ck':
            return f'CONSTRAINT ck_{t}_{constraint.name} CHECK ({constraint.body})'
        return f'CONSTRAINT fk_{t}_{constraint.name} FOREIGN KEY {constraint.body}'


def create(tables: tuple[Table, ...]) -> None:
    for table in tables:
        op.execute(table.create_sql())


def closed_set(column: str, values: tuple[str, ...]) -> str:
    quoted = ', '.join("'" + v.replace("'", "''") + "'" for v in values)
    return f'{column} IN ({quoted})'


# (table, column, partial): the foreign-key indexes of 15a section 1, "Indexes". Written out, not
# derived: a reviewer sees each one, and the plan test (tests/db/_storage_plan.py) lists the same set.
FOREIGN_KEY_INDEXES: tuple[tuple[str, str, bool], ...] = (
    *((t, 'project_id', False) for t in (
        'photo_provenance', 'claims', 'analysis_ledger', 'photo_capture', 'detections',
        'class_opinions', 'item_region_results', 'region_boxes', 'box_texts', 'text_lines',
        'probe_predictions', 'item_scores', 'item_projections', 'assist_answers', 'item_notes',
        'vector_gaps', 'item_state', 'box_state', 'entity_edges', 'entity_state',
        'membership_state', 'face_state', 'photo_state', 'photo_files', 'photo_derivatives',
        'photo_signals', 'photo_places', 'erasure_jobs',
    )),
    ('photos', 'ingest_job_id', True), ('photo_provenance', 'import_id', True),
    ('photo_provenance', 'combine_job_id', True), ('items', 'photo_key', False),
    ('entities', 'cluster_uid', True), ('detections', 'photo_key', False),
    ('detections', 'import_id', True), ('class_opinions', 'import_id', True),
    ('region_boxes', 'import_id', True), ('text_lines', 'photo_key', False),
    ('text_lines', 'item_key', True), ('item_notes', 'item_key', False),
    ('item_notes', 'import_id', True), ('faces', 'photo_key', False),
    ('faces', 'person_item_key', True), ('box_state', 'box_key', True),
    ('entity_state', 'parent_key', True), ('entity_state', 'cover_photo_key', True),
    ('face_state', 'person_entity_key', True), ('erasure_jobs', 'job_id', True),
)  # fmt: skip


def foreign_key_index_sql(table: str, column: str, *, partial: bool) -> str:
    where = f' WHERE {column} IS NOT NULL' if partial else ''
    return f'CREATE INDEX ix_{table}_{column} ON {table} ({column}){where}'
