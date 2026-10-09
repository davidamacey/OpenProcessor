"""Promote-gate evaluation and class-name resolution for promoting a run (no FastAPI)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.clients.curation_opensearch.registry import get_class_registry
from src.core.logging import get_logger
from src.services.training import jobs as train_jobs


logger = get_logger(__name__)


# Promote-gate thresholds. Mirrored as named constants so
# tests can monkey-patch them without re-parsing the router source.
PROMOTE_GATE_MAP50_MIN = 0.65
PROMOTE_GATE_PER_CLASS_PRECISION_MIN = 0.50
PROMOTE_GATE_PER_CLASS_SUPPORT_MIN = 5


class PromoteGateFailure(BaseModel):
    """One machine-readable promote-gate failure.

    F-64 (fresh-start E2E findings 2026-09-25): the 422 used to carry only
    free-text strings in ``failures``; a UI-only user saw just "API 422"
    because there was nothing structured to render. ``code`` is a stable
    identifier a frontend can switch on without string-parsing;
    ``message`` is still the human-readable detail for display.
    """

    code: str
    message: str
    class_name: str | None = None


class PromoteGateFailedDetail(BaseModel):
    """The ``detail`` body of every promote-blocking 422.

    ``force_allowed`` tells the caller (and the UI) whether re-submitting
    with ``force=true`` can get past *this specific* failure — some 422s
    (e.g. the job simply isn't finished yet) force cannot bypass.
    """

    message: str
    failures: list[PromoteGateFailure]
    force_allowed: bool
    override: str | None = None
    thresholds: dict[str, float | int] | None = None


class PromoteGateFailedResponse(BaseModel):
    """Documents the actual FastAPI error envelope: ``{"detail": ...}``."""

    detail: PromoteGateFailedDetail


def evaluate_promote_gate(eval_block: dict[str, Any] | None) -> list[PromoteGateFailure]:
    """Return a list of structured gate failures, [] when the gate passes.

    The gate runs against ``status.json``'s ``eval`` block:
    mAP50 floor + per-class precision floor + per-class support floor.
    """
    failures: list[PromoteGateFailure] = []
    if not eval_block:
        failures.append(
            PromoteGateFailure(
                code='no_eval_block',
                message='no eval block in status.json — trainer never ran val()',
            )
        )
        return failures

    map50 = eval_block.get('map50')
    if not isinstance(map50, int | float) or map50 < PROMOTE_GATE_MAP50_MIN:
        failures.append(
            PromoteGateFailure(
                code='map50_below_floor',
                message=f'mAP50 {map50!r} < {PROMOTE_GATE_MAP50_MIN} (promote-gate floor)',
            )
        )

    per_class = eval_block.get('per_class') or []
    if not isinstance(per_class, list):
        failures.append(
            PromoteGateFailure(code='per_class_not_list', message='eval.per_class is not a list')
        )
        return failures

    for row in per_class:
        if not isinstance(row, dict):
            continue
        name = row.get('name') or f'class_id={row.get("class_id")}'
        precision = row.get('precision')
        support = row.get('support')
        # A non-numeric metric (null, string, missing) is a gate FAILURE,
        # not a skip — an unreadable eval is exactly the case the gate
        # exists to catch. isinstance(x, bool) is deliberately not
        # excluded from the numeric check for precision since Triton/trainer
        # never emits bool there; support uses `int` so a JSON `true`/`false`
        # would (correctly) still gate-fail on the < comparison below it if
        # it ever slipped through as a bool.
        if not isinstance(precision, int | float):
            failures.append(
                PromoteGateFailure(
                    code='per_class_precision_not_numeric',
                    message=f'{name}: precision {precision!r} is not numeric (promote-gate floor)',
                    class_name=name,
                )
            )
        elif precision < PROMOTE_GATE_PER_CLASS_PRECISION_MIN:
            failures.append(
                PromoteGateFailure(
                    code='per_class_precision_below_floor',
                    message=(
                        f'{name}: precision {precision:.3f} '
                        f'< {PROMOTE_GATE_PER_CLASS_PRECISION_MIN}'
                    ),
                    class_name=name,
                )
            )
        if not isinstance(support, int):
            failures.append(
                PromoteGateFailure(
                    code='per_class_support_not_int',
                    message=f'{name}: support {support!r} is not an int (promote-gate floor)',
                    class_name=name,
                )
            )
        elif support < PROMOTE_GATE_PER_CLASS_SUPPORT_MIN:
            failures.append(
                PromoteGateFailure(
                    code='per_class_support_below_floor',
                    message=(
                        f'{name}: support {support} < {PROMOTE_GATE_PER_CLASS_SUPPORT_MIN} '
                        'test crops (promoting on no test data)'
                    ),
                    class_name=name,
                )
            )
    return failures


async def resolve_full_registry_for_promote(job_id: str) -> dict[int, str]:
    """Resolve class_id -> name for ``labels.txt``, preferring the registry
    snapshot pinned at submit time over the live registry.

    ``labels.txt`` used to be rebuilt from the *live* registry at promote
    time — a rename between export and promote silently mislabeled the
    served model. Falls back to the live registry only when no pin is
    available (older runs, or a pin/read failure), and always logs loudly
    when it does so the gap is visible in the API logs.
    """
    job_raw = await train_jobs.read_job_spec(job_id)
    snapshot_path = (job_raw or {}).get('registry_snapshot_path')
    if snapshot_path:
        try:
            import json
            from pathlib import Path

            data = json.loads(Path(snapshot_path).read_text(encoding='utf-8'))
            pinned = {
                int(c['class_id']): str(c['class_name'])
                for c in data.get('classes') or []
                if not c.get('deprecated')
            }
        except Exception as exc:
            logger.warning(
                'curation_promote_registry_pin_unreadable',
                job_id=job_id,
                snapshot_path=snapshot_path,
                error=str(exc),
            )
        else:
            if pinned:
                return pinned
            logger.warning(
                'curation_promote_registry_pin_empty', job_id=job_id, snapshot_path=snapshot_path
            )
    else:
        logger.warning(
            'curation_promote_registry_pin_missing',
            job_id=job_id,
            note=(
                'no registry_snapshot_path on this job — falling back to the LIVE '
                'registry. A class rename since submit would silently relabel '
                'this model.'
            ),
        )
    registry_snapshot = get_class_registry().load()
    return {c.class_id: c.class_name for c in registry_snapshot.classes if not c.deprecated}


def registry_ids_contiguous_from_zero(full_registry: dict[int, str]) -> bool:
    """``True`` iff ``full_registry``'s ids are exactly ``0..N-1``.

    Only in that case does the legacy identity map (``labels.txt`` line
    ``i`` = registry class ``i``'s name) ever agree with the dense export
    id a full-class-trained model actually predicted -- ``_build_export_id_map``
    (``src/services/curation/export_support.py``) assigns dense ids in
    ascending registry-id order over the *non-deprecated* classes only, so
    any gap (a deprecated class, or ids that don't start at 0) makes dense
    id != registry id for every class after the gap. Callers must pass the
    registry snapshot pinned at export/submit time, never the live
    registry, since "no gaps at export time" is the only claim this proves.
    """
    ids = sorted(full_registry)
    return ids == list(range(len(ids)))
