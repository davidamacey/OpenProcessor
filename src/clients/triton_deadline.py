"""A deadline on every Triton call that does not go through the async pool.

``tritonclient`` ``infer(..., client_timeout=None)`` waits forever, so a hung Triton would hold
the caller's worker thread or request for good. The async pool already passes
``client_timeout``; these wrappers give the sync clients and the aio embedding calls the same.
A gRPC DEADLINE_EXCEEDED becomes :class:`TritonDeadlineError`, which the API answers with 504.
"""

from __future__ import annotations

from typing import Any

import grpc
from tritonclient.utils import InferenceServerException

from src.utils.retry import TritonDeadlineError


#: Seconds a sync or embedding Triton call may take. Above every measured call; the async pool's
#: own default is the same 30 s.
TRITON_SYNC_CLIENT_TIMEOUT_S = 30.0

_DEADLINE_STATUS = str(grpc.StatusCode.DEADLINE_EXCEEDED)


def _as_deadline(model_name: str, exc: InferenceServerException) -> TritonDeadlineError | None:
    if exc.status() != _DEADLINE_STATUS:
        return None
    return TritonDeadlineError(
        f'Triton did not answer {model_name!r} within {TRITON_SYNC_CLIENT_TIMEOUT_S:.0f}s',
        last_error=exc,
    )


def infer_with_deadline(client: Any, model_name: str, inputs: list, outputs: list) -> Any:
    """``client.infer`` with ``client_timeout``; a missed deadline raises ``TritonDeadlineError``."""
    try:
        return client.infer(
            model_name,
            inputs,
            outputs=outputs,
            client_timeout=TRITON_SYNC_CLIENT_TIMEOUT_S,
        )
    except InferenceServerException as exc:
        deadline = _as_deadline(model_name, exc)
        if deadline is None:
            raise
        raise deadline from exc


async def ainfer_with_deadline(client: Any, model_name: str, inputs: list, outputs: list) -> Any:
    """Async twin of :func:`infer_with_deadline` for a ``tritonclient.grpc.aio`` client."""
    try:
        return await client.infer(
            model_name,
            inputs,
            outputs=outputs,
            client_timeout=TRITON_SYNC_CLIENT_TIMEOUT_S,
        )
    except InferenceServerException as exc:
        deadline = _as_deadline(model_name, exc)
        if deadline is None:
            raise
        raise deadline from exc
