"""Ingest item-detector profiles, resolved from the environment.

Two :class:`~src.config.DetectionProfile` instances drive ingest:

* **Primary** (``OP_INGEST_PRIMARY_<FIELD>``, default name ``item``) — a
  fused-output item proposer. ``assigns_class`` says whether its label space
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
(:mod:`src.domain.ingest_class_sources`).
"""

from __future__ import annotations

from functools import lru_cache

from src.config.detection_profile import DetectionProfile, reject_legacy_detection_env
from src.config.env import env_is_set
from src.core.logging import get_logger
from src.utils.class_names import confine_labels_path, get_class_names


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
    if env_is_set(RETIRED_CLASS_IDS_ENV):
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


def ingest_primary_name() -> str:
    """The primary profile's name (``item`` unless ``OP_INGEST_PRIMARY_NAME``): the
    ``class_source`` prefix. Deployment-wide: a project's ingest profile never
    changes it."""
    return ingest_primary_profile().name


def ingest_secondary_name() -> str:
    """The secondary profile's name (``secondary`` unless ``OP_INGEST_SECONDARY_NAME``),
    whether or not the deployment configures a secondary model: a project's own
    secondary writes ``{name}_model`` too."""
    return DetectionProfile.from_env(INGEST_SECONDARY_ENV_PREFIX, name='secondary').name


@lru_cache(maxsize=8)
def load_label_names(path: str) -> tuple[str, ...]:
    """Read a ``labels.txt``-style file (one label per line, line index =
    model class id). The path must lie inside the model repository
    (``ValueError`` otherwise, nothing read). Raises ``OSError`` if unreadable
    — a configured but missing labels file is a deployment error, not
    something to guess."""
    return tuple(line.strip() for line in confine_labels_path(path).read_text().splitlines())


def model_class_name(profile: DetectionProfile, class_id: int) -> str | None:
    """The detector model's own name for its output index ``class_id``, or ``None`` when it
    has none. The name is the model's only stable class identity; the index is not."""
    if profile.labels_path:
        labels = load_label_names(profile.labels_path)
        return labels[class_id] or None if 0 <= class_id < len(labels) else None
    return get_class_names(profile.detector_model).get(class_id) or None


def proposer_label(profile: DetectionProfile, class_id: int) -> str:
    """The primary model's own name for ``class_id``: its ``labels_path``
    entry when configured, else the ``labels.txt`` shipped in the model's own
    Triton directory, else the bare id as a string. (A region profile's
    ``parent_classes`` select items by this name, so an unconfigured install
    must still get ``car``, not ``2``.)"""
    return model_class_name(profile, class_id) or str(class_id)


__all__ = [
    'INGEST_PRIMARY_ENV_PREFIX',
    'INGEST_SECONDARY_ENV_PREFIX',
    'ingest_primary_name',
    'ingest_primary_profile',
    'ingest_secondary_name',
    'ingest_secondary_profile',
    'load_label_names',
    'model_class_name',
    'proposer_label',
]
