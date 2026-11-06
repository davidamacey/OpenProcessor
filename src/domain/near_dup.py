"""Near-duplicate vocabulary: the closed lists of the ``near_dup_*`` tables.

Pure data. The ``CHECK`` constraints are built from these tuples in ``src/db/models/near_dup.py``.
"""

from __future__ import annotations


LEVELS: tuple[str, ...] = ('image', 'item', 'core')
PROJECT_LEVELS: tuple[str, ...] = ('image', 'item')
PURPOSES: tuple[str, ...] = ('dedup', 'leak')
PENDING_REASONS: tuple[str, ...] = ('added', 'vector_changed', 'boxes_changed', 'deleted')
RUN_MODES: tuple[str, ...] = ('full', 'incremental')
RUN_STATUSES: tuple[str, ...] = ('building', 'current', 'superseded', 'failed')
EDGE_KINDS: tuple[str, ...] = ('embedding',)
DECISIONS: tuple[str, ...] = ('same', 'different')

# Today's behaviour (0.98 cosine on the whole frame and on the crop) until the prototype runs
# say otherwise.
DEFAULT_IMAGE_DEDUP_COSINE = 0.98
DEFAULT_ITEM_DEDUP_COSINE = 0.98
DEFAULT_K = 64
DEFAULT_MAX_GROUP_SIZE = 1000
MIN_K = 8
MAX_K = 256
MIN_LEAK_COSINE = 0.5
MIN_GROUP_SIZE_GUARD = 2
