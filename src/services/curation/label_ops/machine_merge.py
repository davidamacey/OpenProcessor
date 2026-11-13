"""What every automated label operation's merge refuses to touch, in one place.

The operations (:mod:`~src.services.curation.label_ops.adopt_merge`,
:mod:`~src.services.curation.label_ops.accept_merge`) re-check each item on the freshest source at
write time: a person may have labelled, excluded or held it out since the ids were snapshotted.
"""

from __future__ import annotations

from typing import Any

from src.services.curation.class_write_guard import class_write_locked


def machine_write_blocked(current: dict[str, Any]) -> bool:
    """True for an item no automated label operation may write: a locked or validated class
    (a human's, an import's, the test holdout) or an item a person excluded."""
    return class_write_locked(current) or bool(current.get('class_excluded'))


__all__ = ['machine_write_blocked']
