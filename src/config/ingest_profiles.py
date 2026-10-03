"""Ingest item-detector profiles, resolved from the environment.

Two :class:`~src.config.DetectionProfile` instances drive ingest:

* **Primary** (``OP_INGEST_PRIMARY_<FIELD>``, default name ``item``) — an
  end2end item proposer. ``assigns_class`` says whether its label space
  *is* the class registry. When false (the default) its detections are
  always unlabeled proposals: no ``class_id``/``class_name``, a
  ``{name}_proposal`` class_source, and the proposer's own label (from
  ``labels_path``) kept as the proposal name.
* **Secondary** (``OP_INGEST_SECONDARY_<FIELD>``, default name
  ``secondary``) — an optional raw-output classifier; off unless
  ``OP_INGEST_SECONDARY_DETECTOR_MODEL`` is set. It writes
  ``{name}_model`` on the boxes it classifies.

Lives in config (not the router) so services and workers can derive the
``class_source`` vocabulary from the same resolution
(:mod:`src.services.curation.ingest_class_sources`).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from src.config.detection_profile import DetectionProfile, reject_legacy_detection_env
from src.core.logging import get_logger
from src.utils.class_names import get_class_names


logger = get_logger(__name__)


INGEST_PRIMARY_ENV_PREFIX = 'OP_INGEST_PRIMARY_'
INGEST_SECONDARY_ENV_PREFIX = 'OP_INGEST_SECONDARY_'
# A deployment-wide hard drop by model class id. Retired: it dropped detections
# for every project before anything was stored; the per-project ingest policy
# replaces it.
RETIRED_CLASS_IDS_ENV = 'OP_INGEST_PRIMARY_CLASS_IDS'


def ingest_primary_profile() -> DetectionProfile:
    """The primary item-proposer profile (``OP_INGEST_PRIMARY_*``)."""
    reject_legacy_detection_env()
    if RETIRED_CLASS_IDS_ENV in os.environ:
        logger.warning(
            'retired_env_ignored',
            name=RETIRED_CLASS_IDS_ENV,
            use='the per-project ingest policy (detect.classes / detect.exclude_classes)',
        )
    return DetectionProfile.from_env(INGEST_PRIMARY_ENV_PREFIX, name='item')


def ingest_secondary_profile() -> DetectionProfile | None:
    """The secondary classifier profile, or ``None`` when not configured."""
    profile = DetectionProfile.from_env(INGEST_SECONDARY_ENV_PREFIX, name='secondary')
    return profile if profile.detector_model else None


@lru_cache(maxsize=8)
def load_label_names(path: str) -> tuple[str, ...]:
    """Read a ``labels.txt``-style file (one label per line, line index =
    model class id). Raises ``OSError`` if unreadable — a configured but
    missing labels file is a deployment error, not something to guess."""
    return tuple(line.strip() for line in Path(path).read_text().splitlines())


def proposer_label(profile: DetectionProfile, class_id: int) -> str:
    """The primary model's own name for ``class_id``: its ``labels_path``
    entry when configured, else the ``labels.txt`` shipped in the model's own
    Triton directory, else the bare id as a string. (A region profile's
    ``parent_classes`` select items by this name, so an unconfigured install
    must still get ``car``, not ``2``.)"""
    if profile.labels_path:
        labels = load_label_names(profile.labels_path)
        if 0 <= class_id < len(labels) and labels[class_id]:
            return labels[class_id]
        return str(class_id)
    return get_class_names(profile.detector_model).get(class_id) or str(class_id)


__all__ = [
    'INGEST_PRIMARY_ENV_PREFIX',
    'INGEST_SECONDARY_ENV_PREFIX',
    'ingest_primary_profile',
    'ingest_secondary_profile',
    'load_label_names',
    'proposer_label',
]
