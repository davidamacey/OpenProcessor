"""``upgrade_from_060 --reset`` (review fix R-9b): undo a committed load whose verify failed.

One transaction as the schema owner (``TRUNCATE`` is not an application right): empties every
table the load wrote, deletes the ``upgrade_runs`` rows and leaves ``upgrade_state`` at
``in_progress``, so the next run loads again from the bundle. Refused once ``delete_sources``
completed, because then the sources are gone and the load cannot be redone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from src.services.upgrade import gate


if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

# ``TRUNCATE ... CASCADE`` follows every foreign key that points at a project, so one statement
# empties the load's project-scoped tables; the deployment-wide rows are removed by name.
STATEMENTS = (
    'TRUNCATE project_slugs, deployment_counters, events CASCADE',
    'DELETE FROM config_documents WHERE project_id IS NULL',
    "DELETE FROM deployment_settings WHERE key <> 'upgrade_state'",
    'DELETE FROM upgrade_runs',
)


def reset(conn: Connection) -> None:
    """Run the reset on an owner connection; raises :class:`gate.UpgradeStateError` after delete_sources."""
    done = conn.execute(
        text(
            "SELECT 1 FROM upgrade_runs WHERE from_version = :v AND step = 'delete_sources' "
            "AND status = 'completed'"
        ),
        {'v': gate.FROM_VERSION},
    ).first()
    if done is not None:
        msg = 'delete_sources completed; the sources are gone and the load cannot be redone'
        raise gate.UpgradeStateError(msg)
    for statement in STATEMENTS:
        conn.execute(text(statement))
    conn.commit()
