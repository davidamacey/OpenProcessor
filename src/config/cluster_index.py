"""Names of the FAISS cluster indexes, matching the OpenSearch visual-search indexes."""

from __future__ import annotations

from enum import Enum


class ClusterIndex(str, Enum):  # noqa: UP042  # (str, Enum) keeps str() and format() as Class.MEMBER; StrEnum would change both
    """FAISS cluster index names matching OpenSearch indexes."""

    GLOBAL = 'global'
    VEHICLES = 'vehicles'
    PEOPLE = 'people'
    FACES = 'faces'
