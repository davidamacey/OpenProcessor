"""The D-47 class reconciliation: items whose ``class_id`` is not in the registry or whose
``class_name`` disagrees with it. Report only: nothing is written.

Shared by the upgrade's verify step and ``openprocessor db verify`` (one piece of code, so the
two never disagree about what drift is). Reads the items index with ``search_after`` through a
:class:`~src.services.upgrade.from_060.admin_client.ControlReader`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.services.upgrade.from_060.admin_client import ControlReader

SAMPLE_LIMIT = 20


@dataclass
class ClassDrift:
    """Items checked and the drift found; ``samples`` is bounded."""

    items: int = 0
    unknown_class_id: int = 0
    name_disagrees: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.unknown_class_id or self.name_disagrees)

    def to_json(self) -> dict[str, Any]:
        return {
            'items': self.items,
            'unknown_class_id': self.unknown_class_id,
            'name_disagrees': self.name_disagrees,
            'samples': self.samples,
        }


async def reconcile_classes(
    reader: ControlReader, items_index: str, registry: Mapping[int, str]
) -> ClassDrift:
    """Compare every item with ``registry`` (class id to name)."""
    drift = ClassDrift()
    async for doc_id, source in reader.iter_docs(items_index):
        drift.items += 1
        class_id = source.get('class_id')
        if class_id is None:
            continue
        if class_id not in registry:
            drift.unknown_class_id += 1
            kind = 'unknown_class_id'
        elif source.get('class_name') is not None and normalize_class_name(
            str(source['class_name'])
        ) != normalize_class_name(registry[class_id]):
            drift.name_disagrees += 1
            kind = 'name_disagrees'
        else:
            continue
        if len(drift.samples) < SAMPLE_LIMIT:
            drift.samples.append(
                {
                    'item': doc_id,
                    'kind': kind,
                    'class_id': class_id,
                    'class_name': source.get('class_name'),
                    'registry_name': registry.get(class_id),
                }
            )
    return drift
