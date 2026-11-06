"""Each closed list of ``clustering_vocab`` equals the ``CHECK`` of the migrated database."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import psycopg
import pytest

from src.domain import clustering_vocab as vocab


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

# (table, constraint, the list it must equal)
CASES = [
    ('cluster_partitions', 'target', vocab.TARGETS),
    ('cluster_calibrations', 'target', vocab.TARGETS),
    ('cluster_identities', 'target', vocab.TARGETS),
    ('cluster_activations', 'target', vocab.TARGETS),
    ('cluster_partitions', 'level', vocab.LEVELS),
    ('cluster_calibrations', 'level', vocab.LEVELS),
    ('cluster_activations', 'level', vocab.LEVELS),
    ('cluster_identities', 'level', vocab.IDENTITY_LEVELS),
    ('cluster_partitions', 'method', vocab.METHODS),
    ('cluster_partitions', 'backend', vocab.BACKENDS),
    ('cluster_partitions', 'status', vocab.PARTITION_STATUSES),
    ('cluster_memberships', 'state', vocab.MEMBERSHIP_STATES),
    ('partition_clusters', 'transition', vocab.TRANSITIONS),
    ('cluster_calibrations', 'kind', vocab.CALIBRATION_KINDS),
    ('cluster_identities', 'name_source', vocab.NAME_SOURCES),
    ('cluster_activation_events', 'reason', vocab.ACTIVATION_REASONS),
]


@pytest.fixture(scope='module')
def constraints(pg: PgDatabase) -> dict[str, str]:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        return dict(
            conn.execute(
                "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint WHERE contype = 'c'"
            ).fetchall()
        )


@pytest.mark.parametrize(('table', 'name', 'values'), CASES)
def test_the_check_lists_exactly_the_vocabulary(constraints, table, name, values) -> None:
    definition = constraints[f'ck_{table}_{name}']
    listed = re.findall(r"'([^']*)'::text", definition)
    assert sorted(listed) == sorted(values), definition
    assert len(set(listed)) == len(listed)


def test_every_list_is_checked_here() -> None:
    checked = {id(values) for *_, values in CASES}
    for name in vars(vocab):
        value = getattr(vocab, name)
        if name.isupper() and isinstance(value, tuple):
            assert id(value) in checked, f'{name} has no CHECK case'
