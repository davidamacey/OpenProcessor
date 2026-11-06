"""The closed lists of the clustering tables: one Python source per ``CHECK``.

Pure data. The ``CHECK`` constraints of ``cluster_partitions``, ``cluster_identities``,
``partition_clusters``, ``cluster_memberships``, ``cluster_calibrations``, ``cluster_activations``
and ``cluster_activation_events`` are built from these tuples in ``src/db/models/clustering.py``, and
``tests/domain/test_clustering_vocab_matches_check.py`` compares them with the migrated database.
A research method or a calibration kind arrives together with a migration that alters the ``CHECK``
and the list here in one change.
"""

from __future__ import annotations


TARGETS: tuple[str, ...] = ('item', 'region_box', 'face', 'image')
LEVELS: tuple[str, ...] = ('class', 'identity', 'scene')
# Levels a v0.7.0 path writes for an identity: class (items and boxes) and identity (faces).
IDENTITY_LEVELS: tuple[str, ...] = ('class', 'identity')
METHODS: tuple[str, ...] = ('ivf', 'minibatch_kmeans', 'union_find', 'detector_class', 'edit')
BACKENDS: tuple[str, ...] = ('cpu', 'gpu')
PARTITION_STATUSES: tuple[str, ...] = ('building', 'ready', 'failed', 'discarded')
MEMBERSHIP_STATES: tuple[str, ...] = ('assigned', 'review', 'unassigned')
TRANSITIONS: tuple[str, ...] = ('new', 'survived', 'split', 'merged', 'absorbed')
CALIBRATION_KINDS: tuple[str, ...] = ('assign_bands',)
NAME_SOURCES: tuple[str, ...] = ('human', 'import', 'vlm')
ACTIVATION_REASONS: tuple[str, ...] = ('auto', 'human', 'rollback', 'edit', 'upgrade')
