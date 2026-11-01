"""Names of the FAISS cluster indexes, matching the OpenSearch visual-search indexes."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


# Relative to the API's working directory (/app in the container, where ./data is mounted).
FAISS_INDEX_DIR = Path('data/faiss_indexes')


class ClusterIndex(str, Enum):  # noqa: UP042  # (str, Enum) keeps str() and format() as Class.MEMBER; StrEnum would change both
    """FAISS cluster index names matching OpenSearch indexes."""

    GLOBAL = 'global'
    VEHICLES = 'vehicles'
    PEOPLE = 'people'
    FACES = 'faces'
