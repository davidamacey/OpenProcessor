"""The per-project ingest policy: which detections are stored and which get a vector.

Two independent stages, both off by default (an absent policy changes nothing):

* ``detect``: a filter on the detector output. Filtered detections are not
  stored (the explicit user choice); the count is reported.
* ``embedding``: ``all`` (default) embeds every stored detection, ``selected``
  embeds only those matching the criteria, ``lazy`` embeds none at ingest.
  A detection without a vector is still stored, with ``embedding_state``
  ``not_selected`` or ``deferred``, and can be embedded later.

An item that carries a human or imported label always gets a vector.
Criteria combine with AND; class names are an OR-list matched by NAME
(:mod:`~src.services.curation.name_match`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.services.curation.class_write_guard import class_write_locked
from src.services.curation.embedding_state import DEFERRED, EMBEDDED, NOT_SELECTED
from src.services.curation.name_match import name_matches, normalized_names
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.services.curation.item_doc import DetectedItem


class _Criteria(BaseModel):
    model_config = ConfigDict(extra='forbid')

    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    min_box_area_frac: float | None = Field(default=None, ge=0.0, le=1.0)
    max_per_image: int | None = Field(default=None, ge=1)


class DetectFilter(_Criteria):
    """Which detector outputs are stored. Every field empty = store all.

    ``class_resolution`` ``by_name`` gives a detection the registry class whose
    name equals the detector's own label (``traffic light`` -> ``traffic_light``),
    written like a classifier's label so the VLM skips it; ``proposal`` (the
    default) leaves every detection an unlabeled proposal.
    """

    classes: list[str] | None = None
    exclude_classes: list[str] = Field(default_factory=list)
    class_resolution: Literal['proposal', 'by_name'] = 'proposal'


class DetectorOverride(BaseModel):
    """A per-project ingest detector: replaces the deployment's primary model for
    this project. It must serve the end2end four-tensor contract, and its class
    ids are never read as registry ids (its detections are proposals, or are
    resolved by name)."""

    model_config = ConfigDict(extra='forbid')

    model: str = Field(min_length=1)
    version: str = '1'
    input_size: int | None = Field(default=None, ge=32, le=4096)
    # labels.txt-style file naming the model's classes; empty reads the one in
    # the model's own directory.
    labels_path: str = ''


class EmbeddingPolicy(_Criteria):
    mode: Literal['all', 'selected', 'lazy'] = 'all'
    classes: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def _selected_needs_a_criterion(self) -> Self:
        if self.mode == 'selected' and not (
            self.classes
            or self.min_confidence is not None
            or self.min_box_area_frac is not None
            or self.max_per_image is not None
        ):
            msg = "mode 'selected' needs at least one criterion (an empty selection embeds nothing)"
            raise ValueError(msg)
        return self


class IngestPolicyBody(BaseModel):
    model_config = ConfigDict(extra='forbid')

    detect: DetectFilter = Field(default_factory=DetectFilter)
    embedding: EmbeddingPolicy = Field(default_factory=EmbeddingPolicy)
    # null = the deployment's primary detector.
    detector: DetectorOverride | None = None


class IngestPolicy(IngestPolicyBody):
    revision: int = 0


@dataclass(frozen=True)
class Candidate:
    """What the policy looks at, for one detection (at ingest) or one stored
    item (preview): independent of where it came from."""

    score: float
    area_frac: float
    class_name: str | None
    proposal_name: str | None
    labeled: bool = False


def candidate_from_item(item: DetectedItem, width: int, height: int) -> Candidate:
    x1, y1, x2, y2 = item.bbox_pixel
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1, width * height)
    return Candidate(item.score, area, item.class_name, item.proposal_name, item.label is not None)


def candidate_from_doc(source: dict[str, Any]) -> Candidate:
    """A stored items document as a candidate (a human-owned or validated class counts as labeled)."""
    return Candidate(
        float(source.get('confidence') or 0.0),
        float(source.get('crop_area_norm') or 0.0),
        source.get('class_name'),
        source.get('proposal_name'),
        class_write_locked(source),
    )


def _best_first(cands: Sequence[Candidate]) -> list[int]:
    """Indices ordered by score, then box area, both descending."""
    return sorted(range(len(cands)), key=lambda i: (-cands[i].score, -cands[i].area_frac, i))


def _meets(c: Candidate, criteria: _Criteria, classes: Sequence[str] | None) -> bool:
    if classes is not None and not name_matches(
        classes, class_name=c.class_name, proposal_name=c.proposal_name
    ):
        return False
    if criteria.min_confidence is not None and c.score < criteria.min_confidence:
        return False
    return not (criteria.min_box_area_frac is not None and c.area_frac < criteria.min_box_area_frac)


def _mask(
    cands: Sequence[Candidate], criteria: _Criteria, classes: Sequence[str] | None
) -> list[bool]:
    """Per-candidate pass flags (all candidates are one image's): the
    per-detection criteria first, then the per-image cap over the survivors,
    best score first."""
    mask = [_meets(c, criteria, classes) for c in cands]
    if criteria.max_per_image is not None:
        kept = 0
        for i in _best_first(cands):
            if not mask[i]:
                continue
            if kept >= criteria.max_per_image:
                mask[i] = False
            else:
                kept += 1
    return mask


def detect_keep_mask(cands: Sequence[Candidate], flt: DetectFilter) -> list[bool]:
    """Which of one image's detections the detect filter keeps."""
    mask = _mask(cands, flt, flt.classes)
    if flt.exclude_classes:
        mask = [
            ok
            and not name_matches(
                flt.exclude_classes, class_name=c.class_name, proposal_name=c.proposal_name
            )
            for ok, c in zip(mask, cands, strict=True)
        ]
    return mask


def apply_detect_filter(
    items: list[DetectedItem], flt: DetectFilter, width: int, height: int
) -> tuple[list[DetectedItem], int]:
    """``(kept items in their original order, number dropped)``."""
    mask = detect_keep_mask([candidate_from_item(it, width, height) for it in items], flt)
    kept = [it for it, ok in zip(items, mask, strict=True) if ok]
    return kept, len(items) - len(kept)


def embedding_states(cands: Sequence[Candidate], policy: EmbeddingPolicy) -> list[str]:
    """Per candidate of one image: ``embedded`` when it should be embedded
    now, else the state it is stored with (``not_selected`` for mode
    ``selected``, ``deferred`` for ``lazy``). A labeled one always embeds."""
    if policy.mode == 'all':
        return [EMBEDDED] * len(cands)
    if policy.mode == 'lazy':
        wanted = [False] * len(cands)
        skipped = DEFERRED
    else:
        wanted = _mask(cands, policy, policy.classes or None)
        skipped = NOT_SELECTED
    return [EMBEDDED if ok or c.labeled else skipped for ok, c in zip(wanted, cands, strict=True)]


def select_for_embedding(
    items: Sequence[DetectedItem], policy: EmbeddingPolicy, width: int, height: int
) -> list[str]:
    """:func:`embedding_states` for one image's freshly detected items."""
    return embedding_states([candidate_from_item(it, width, height) for it in items], policy)


def assign_classes_by_name(
    items: Sequence[DetectedItem], registry: Mapping[str, tuple[int, str]], detector_name: str
) -> int:
    """Give each unlabeled detection the registry class named like its proposal
    (``registry`` maps normalized class name to ``(class_id, class_name)``).
    Returns how many were resolved. The class is written with the detector's
    classifier-style ``class_source`` (``{detector_name}_model``); the proposal
    name is kept for lineage."""
    resolved = 0
    for item in items:
        if item.class_id is not None or item.label is not None or not item.proposal_name:
            continue
        found = registry.get(normalize_class_name(item.proposal_name))
        if found is None:
            continue
        item.class_id, item.class_name = found
        item.detector_class_id = found[0]
        item.class_source = f'{detector_name}_model'
        resolved += 1
    return resolved


def registry_name_index(registry: Any) -> dict[str, tuple[int, str]]:
    """``{normalized name: (class_id, class_name)}`` of the classes an item may be
    given (active, and not the region class)."""
    from src.services.curation.region_class import item_classes

    return {
        normalize_class_name(c.class_name): (c.class_id, c.class_name)
        for c in item_classes(registry.load().classes)
    }


def unknown_names(policy: IngestPolicyBody, known_slugs: set[str]) -> list[str]:
    """Class names the policy mentions that match no known label or registry
    class. Accepted (a model switch or a later class can make them valid) and
    returned as a warning."""
    mentioned = [
        *(policy.detect.classes or []),
        *policy.detect.exclude_classes,
        *policy.embedding.classes,
    ]
    return sorted(
        {n for n in mentioned if normalized_names([n]) and normalized_names([n]) - known_slugs}
    )


__all__ = [
    'Candidate',
    'DetectFilter',
    'DetectorOverride',
    'EmbeddingPolicy',
    'IngestPolicy',
    'IngestPolicyBody',
    'apply_detect_filter',
    'assign_classes_by_name',
    'candidate_from_doc',
    'candidate_from_item',
    'detect_keep_mask',
    'embedding_states',
    'registry_name_index',
    'select_for_embedding',
    'unknown_names',
]
