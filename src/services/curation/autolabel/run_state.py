"""What every stage of one auto-label run shares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.services.curation.item_filter import ItemFilter


@dataclass(frozen=True)
class Run:
    """What every stage of one run shares."""

    opensearch: Any
    progress: Any
    class_id: int | None
    cluster_id: int | None
    item_filter: dict[str, Any] | None
    scope_filter: ItemFilter


def start_stage(run: Run, name: str, **kwargs: Any) -> None:
    if run.progress is not None:
        run.progress.start_stage(name, **kwargs)


def start_cancellable_stage(run: Run, name: str) -> None:
    if run.progress is not None:
        run.progress.start_stage(name)
        run.progress.raise_if_cancelled()
