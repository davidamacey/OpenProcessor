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
  array of embeddings, both drawn from the scorer's declared :attr:`~CropScorer.pool`
  (see below) and fetched ONCE per job by :mod:`item_scores.job` — every
  scorer sharing a given pool gets the same fetch, not a re-fetch each,
  since the dominant cost is the OpenSearch read, not the math. A scorer
  on the ``'probe_scored'`` pool gets an empty ``(n, 0)`` array — no
  embeddings are fetched for that pool at all.
* Output — a :class:`ScoreResult` whose ``fields`` dict is keyed by
  ``crop_id`` and maps to a ``{field_name: value}`` dict ready to merge into
  a bulk ``update`` doc body.

Scorers that need more than embeddings (e.g. mistakenness, which reads
``probe_pred_*`` + ``class_name`` off the crop doc and ignores the
embeddings array entirely) accept the raw ``opensearch`` client and fetch
their own supplementary fields.
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

    pool: ClassVar[str]
    """Which pre-fetched item pool this scorer expects. ``item_scores.job``
    builds each needed pool ONCE per job (not once per scorer, and not
    unconditionally — only pools a requested scorer actually needs are
    fetched) and hands the matching one to every scorer that declares it.

    Two pools exist today:

    * ``'residual'`` — the clustering *residual* pool
      (:func:`~src.services.curation.clustering.embedding_reduce.fetch_residual_embeddings_parallel`):
      crops with no confident class assignment, i.e. excluding
      ``class_validated``, every ``class_source`` in
      ``CONFIDENT_CLASS_SOURCES`` (classifier/VLM machine labels), and
      ``class_excluded``. Fetched WITH embeddings. Correct for
      embedding-based scorers aimed at unlabeled/residual items
      (``uniqueness``, ``near_dup``) — scoring an already-labeled item's
      embedding neighbourhood isn't what those scorers are for.
    * ``'probe_scored'`` — every item the probe has scored (``exists
      probe_pred_class``), excluding ``test_holdout``, ``class_excluded``,
      and ``class_validated`` (human-validated labels are human-owned and
      not audited by an automated mistakenness pass). Fetched as ids
      ONLY — no embeddings are read from OpenSearch; scorers on this pool
      receive an empty ``(n, 0)`` embeddings array. Correct for scorers
      that audit machine labels via non-embedding fields (``mistakenness``,
      which reads ``probe_pred_*`` / ``class_name`` per item and never
      touches embeddings) — the residual pool wrongly excludes exactly the
      confidently machine-labeled items mistakenness exists to audit.

    Default is ``'residual'`` (the pre-existing behavior); scorers must
    still declare it explicitly (this is a plain ``Protocol``, not a base
    class, so there is no inherited default).
    """

    async def score(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        *,
        opensearch: AsyncOpenSearch | None = None,
        progress: Any = None,
    ) -> ScoreResult: ...


__all__ = ['CropScorer', 'ScoreResult']
