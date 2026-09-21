"""Curation clustering subsystem.

Namespaced under ``src/services/curation/clustering/`` (not
``src/services/clustering.py``) to avoid colliding with the
pre-existing, unrelated FAISS visual-search clustering module already
public at that path — see
``docs/design/curation_design_rationale.md`` §6 for the related
``ClusterIndex.VEHICLES`` naming note.
"""

from __future__ import annotations
