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
