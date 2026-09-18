"""Curation clustering subsystem.

Namespaced under ``src/services/curation/clustering/`` (not
``src/services/clustering.py``) to avoid colliding with the
pre-existing, unrelated FAISS visual-search clustering module already
public at that path — see
``docs/design/oss_genericization_phase2_plan.md`` §0.11.
"""

from __future__ import annotations
