"""Names and vocabulary of the embedding registry (pure: no I/O, no database).

An encoder build is a *model version*; one vector field in one project index written by one
model version for one target at one dimension is a *space*. The registry stores the physical
field names; readers take them from a resolved space and never compute them
(embeddings plan 5.1, 5.3).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MODEL_KEY_RE = re.compile(r'^[a-z][a-z0-9_]{1,39}$')
SPACE_KEY_RE = re.compile(r'^[a-z][a-z0-9_]{2,47}$')
MAX_LIVE_SPACES_PER_TARGET = 3

# The default spaces of the 0.6.0 layout keep their physical names.
LEGACY_ITEM_VECTOR_FIELD = 'pe_embedding'
LEGACY_ITEM_STATE_FIELD = 'embedding_state'
LEGACY_BOX_NESTED_PATH = 'region_box_embeddings'
LEGACY_BOX_VECTOR_LEAF = 'embedding'


class Target(StrEnum):
    ITEM = 'item'
    BOX = 'box'
    FRAME = 'frame'
    ITEM_TEXT = 'item_text'


class Role(StrEnum):
    PRIMARY = 'primary'
    OPTIONAL = 'optional'
    CANDIDATE = 'candidate'


class Status(StrEnum):
    PENDING = 'pending'
    BUILDING = 'building'
    READY = 'ready'
    RETIRING = 'retiring'
    RETIRED = 'retired'
    FAILED = 'failed'


#: A space counts against the per-target limit until it is retired or failed.
LIVE_STATUSES = (Status.PENDING, Status.BUILDING, Status.READY, Status.RETIRING)

#: Allowed status moves (embeddings plan 5.5); anything else is refused.
TRANSITIONS: dict[Status, frozenset[Status]] = {
    Status.PENDING: frozenset({Status.BUILDING, Status.FAILED, Status.RETIRED}),
    Status.BUILDING: frozenset({Status.READY, Status.FAILED, Status.RETIRING}),
    Status.READY: frozenset({Status.RETIRING}),
    Status.RETIRING: frozenset({Status.RETIRED}),
    Status.FAILED: frozenset({Status.BUILDING, Status.RETIRED}),
    Status.RETIRED: frozenset(),
}


class KnnMethodSpec(BaseModel):
    """The k-NN method of one vector field, stored in ``embedding_spaces.knn_method``."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    engine: Literal['lucene', 'faiss'] = 'lucene'
    space_type: Literal['cosinesimil'] = 'cosinesimil'
    m: int = Field(default=16, ge=2)
    ef_construction: int = Field(default=128, ge=1)
    mode: Literal['in_memory', 'on_disk'] | None = None
    compression_level: Literal['1x', '2x', '4x', '8x', '16x', '32x'] | None = None
    data_type: Literal['float', 'byte', 'binary'] | None = None


def space_key(
    model_key: str, version: int, dim: int | None = None, native_dim: int | None = None
) -> str:
    """``<model_key>_v<version>``, plus ``_d<dim>`` when ``dim`` is a Matryoshka cut."""
    key = f'{model_key}_v{version}'
    if dim is not None and native_dim is not None and dim != native_dim:
        key = f'{key}_d{dim}'
    if not SPACE_KEY_RE.match(key):
        msg = f'not a valid space key: {key!r}'
        raise ValueError(msg)
    return key


def index_role_for(target: Target) -> Literal['images', 'items']:
    """Frame vectors live in the images index; item, box and item_text in the items index."""
    return 'images' if target is Target.FRAME else 'items'


@dataclass(frozen=True)
class PhysicalNames:
    vector_field: str
    state_field: str | None
    nested_path: str | None


def physical_names(key: str, target: Target) -> PhysicalNames:
    """Field names of a NEW space (the upgraded default spaces keep the 0.6.0 names).

    Box spaces get their own nested sibling so one space's writer never rewrites another's.
    """
    if target is Target.BOX:
        return PhysicalNames(f'boxvec_{key}.embedding', None, f'boxvec_{key}')
    state = None if target is Target.FRAME else f'vec_{key}_state'
    return PhysicalNames(f'vec_{key}', state, None)


def legacy_names(target: Target) -> PhysicalNames:
    """Names of the default ``pe_core_l14_336`` v1 space of each target in the 0.6.0 layout."""
    match target:
        case Target.ITEM:
            return PhysicalNames(LEGACY_ITEM_VECTOR_FIELD, LEGACY_ITEM_STATE_FIELD, None)
        case Target.FRAME:
            return PhysicalNames(LEGACY_ITEM_VECTOR_FIELD, None, None)
        case Target.BOX:
            return PhysicalNames(
                f'{LEGACY_BOX_NESTED_PATH}.{LEGACY_BOX_VECTOR_LEAF}', None, LEGACY_BOX_NESTED_PATH
            )
        case Target.ITEM_TEXT:
            msg = 'item_text has no 0.6.0 default space'
            raise ValueError(msg)


def input_spec_sha256(input_spec: dict[str, Any]) -> str:
    """Digest of the canonical JSON of an encoder input spec (also the ``preprocess_id``)."""
    canonical = json.dumps(input_spec, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class ResolvedSpace:
    """What a reader needs to query or write one space; every name comes from the registry."""

    space_key: str
    target: Target
    vector_field: str
    state_field: str | None
    nested_path: str | None
    dim: int
    index_role: str
    model_key: str
    model_version: int
    triton_image_model: str | None
    triton_text_model: str | None
    knn_method: KnnMethodSpec


class EmbeddingSpaceError(Exception):
    """A registry rule refused the operation; ``code`` is the stable wire code, ``status`` the HTTP one."""

    def __init__(self, code: str, detail: str, *, status: int = 409) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status = status
