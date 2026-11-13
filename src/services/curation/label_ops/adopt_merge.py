"""The merge of the ``adopt_detector`` label operation: each item gets its detector's class.

Kept apart from :mod:`~src.services.curation.detector_adoption` (which queues the operation) so
the operation's job body can build the merge without importing the code that starts it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.class_label import ItemLabel, class_label_update
from src.services.curation.label_ops.machine_merge import machine_write_blocked
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

ADOPT_WRITER = 'machine:adopt_detector'
"""Writer id of the operation (a machine one: the lock rule never mistakes it for a human)."""


def adoption_merger(
    classes: Mapping[str, tuple[int, str]], op_id: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """``classes`` maps a detector name (registry-name form) to ``(class_id, class_name)``.

    A locked, validated, excluded or already classed item, and a name with no class in ``classes``, is left
    as it is: an empty merge writes nothing."""

    def merge(current: dict[str, Any]) -> dict[str, Any]:
        # class_excluded is re-checked here: a person may exclude an item after the id snapshot.
        if machine_write_blocked(current) or current.get('class_id') is not None:
            return {}
        found = classes.get(normalize_class_name(str(current.get('detector_class_name') or '')))
        if found is None:
            return {}
        return class_label_update(
            current,
            ItemLabel.detector(
                class_id=found[0], class_name=found[1], writer=ADOPT_WRITER, op_id=op_id
            ),
        )

    return merge


__all__ = ['ADOPT_WRITER', 'adoption_merger']
