"""``op.periodic.reload_promoted``: put promoted models back after a Triton restart.

Triton in explicit-control mode loads only its ``--load-model`` list at startup, so a promoted model
(a live ``promotions`` row with its directory in the repository) is UNAVAILABLE after any Triton
restart until something posts ``/load`` again. Every 15 s this re-loads the ones Triton does not
report READY and that an operator did not unload on purpose. The call is idempotent (a READY model
is skipped) and reports problems in its result instead of raising, so no lock is needed.
"""

from __future__ import annotations

from typing import Any

from src.services.training.triton_reload import reload_promoted_models


async def tick() -> dict[str, Any]:
    return await reload_promoted_models()
