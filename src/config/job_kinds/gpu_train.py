"""Job kinds of the `gpu_train` family; filled by the lane that owns it."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.config.job_kinds.kind import JobKind


KINDS: tuple[JobKind, ...] = ()
