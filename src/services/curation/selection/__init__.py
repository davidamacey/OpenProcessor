"""Registry-adjacent package for the **overlay/selection** axis —
pool-scale, pure-selection operations
that layer on top of whatever cluster assignment already exists without
ever writing ``cluster_id``/``cluster_subid``/``cluster_distance``.

Currently a single method: :func:`k_center_greedy` (diversity / core-set
sampling). Unlike ``cluster_methods/`` and ``crop_scores/``,
there is no per-request "registry lookup by name" here yet — one method,
one function — so no ``get_method``-style dispatcher is introduced until
a second overlay method actually exists (avoid a one-entry registry).
"""

from __future__ import annotations

from src.services.curation.selection.kcenter_greedy import k_center_greedy


__all__ = ['k_center_greedy']
