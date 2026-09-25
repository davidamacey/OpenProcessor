"""Curation-score scorer interface.

Mirrors ``cluster_methods/base.py``'s ``ClusterMethod`` Protocol, but for the
*overlay* axis: a :class:`CropScorer` never assigns a crop to a cluster — it
only writes scalar/keyword score fields onto the crop doc it was asked to
score. ``writes`` is a ``ClassVar`` tuple of the exact OpenSearch field names
a scorer may write; :func:`crop_scores.job.run_scoring_job` uses it to build
bulk-update bodies, and
``tests/curation/test_crop_scores.py::test_no_scorer_writes_cluster_fields``
asserts none of them ever contains ``cluster_id`` / ``cluster_subid`` /
``cluster_distance``.

Shape contract:

* Input — a list of ``crop_id`` strings and an aligned ``(n, d)`` float32
  array of L2-normalized embeddings (fetched ONCE per job by
  :mod:`crop_scores.job` and hand to every enabled scorer — the dominant
  cost is the OpenSearch read, not the math, so re-fetching per scorer would
  be wasteful).
* Output — a :class:`ScoreResult` whose ``fields`` dict is keyed by
  ``crop_id`` and maps to a ``{field_name: value}`` dict ready to merge into
  a bulk ``update`` doc body.

Scorers that need more than embeddings (e.g. mistakenness, which reads
``probe_pred_*`` + ``class_name`` off the crop doc) accept the raw
``opensearch`` client and fetch their own supplementary fields — only the
embedding fetch is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable


if TYPE_CHECKING:
    import numpy as np
    from opensearchpy import AsyncOpenSearch


@dataclass(frozen=True)
class ScoreResult:
    """One scorer's output for one job run."""

    scorer: str
    """Canonical scorer name (matches :attr:`CropScorer.name`)."""

    version: str
    """Scorer algorithm version — bumped whenever the math changes so a
    partial re-score never leaves a mixed-version field. ``/curation/scores/coverage`` reports distinct
    ``(method, version)`` pairs."""

    scored_at: str
    """ISO-8601 UTC timestamp shared by every field this call wrote."""

    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    """``{crop_id: {field_name: value, ...}}`` — ready to merge into a bulk
    ``update`` doc body. Only crops this scorer actually produced a value
    for are present (e.g. near-dup singletons are omitted)."""

    n_scored: int = 0
    """Number of crop docs this result has an entry for (== len(fields))."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Free-form telemetry (e.g. n_groups, n_dropped, AUROC on a validation
    subset) — surfaced verbatim in the job summary."""


@runtime_checkable
class CropScorer(Protocol):
    """Protocol every curation-score overlay implements."""

    name: ClassVar[str]
    """Canonical registry key (``'uniqueness'`` / ``'mistakenness'`` /
    ``'near_dup'``)."""

    writes: ClassVar[tuple[str, ...]]
    """OpenSearch field names this scorer may write. MUST NOT contain
    ``cluster_id`` / ``cluster_subid`` / ``cluster_distance`` — enforced by
    a regression test."""

    version: ClassVar[str]
    """Algorithm version stamped onto every field this scorer writes."""

    async def score(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        *,
        opensearch: AsyncOpenSearch | None = None,
        progress: Any = None,
    ) -> ScoreResult: ...


__all__ = ['CropScorer', 'ScoreResult']
