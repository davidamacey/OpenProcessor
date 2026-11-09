"""Stub a queued label operation: the POST answers 202 ``LabelOpAccepted`` and
the job it names answers ``GET /jobs/{id}`` with its result, the way the client
follows every move, bulk label and undo."""

from __future__ import annotations

import re
from typing import Any

from fixtures.contract import job, label_op_accepted, undo_response


def stub_label_op(
    stub: Any,
    post_pattern: str,
    op_id: str,
    *,
    updated: int = 1,
    conflicts: int = 0,
    method: str = "POST",
    calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> str:
    """Register the POST and its job; return the job id. ``calls`` collects
    ``(path, body)`` of each request so a test can assert the sent body."""
    job_id = f"job-{op_id}"

    def accept(request: Any, match: "re.Match[str]") -> Any:
        if calls is not None:
            calls.append((match.string, request.post_data_json or {}))
        return (202, label_op_accepted(op_id, job_id))

    stub.on(method, post_pattern, accept)
    stub.on(
        "GET",
        rf"/jobs/{re.escape(job_id)}$",
        job(
            job_id,
            result={
                "op_id": op_id,
                "updated": updated,
                "conflicts": conflicts,
                "complete": True,
            },
        ),
    )
    return job_id


def stub_label_op_undo(
    stub: Any,
    op_id: str,
    *,
    restored: int = 1,
    kept_changed: int = 0,
    calls: list[dict[str, Any]] | None = None,
) -> None:
    """``POST /label_ops/{op_id}/undo``: a dry run answers the counts at once; a
    real undo is queued as a ``label_undo`` job whose result is the same body."""
    job_id = f"job-undo-{op_id}"
    result = undo_response(op_id, restored=restored, kept_changed=kept_changed, matched=restored)

    def undo(request: Any, _match: "re.Match[str]") -> Any:
        body = request.post_data_json or {}
        if calls is not None:
            calls.append(body)
        if body.get("dry_run", True):
            return (200, {**result, "dry_run": True})
        return (202, label_op_accepted(op_id, job_id))

    stub.on("POST", rf"/label_ops/{re.escape(op_id)}/undo$", undo)
    stub.on("GET", rf"/jobs/{re.escape(job_id)}$", job(job_id, kind="label_undo", result=result))
