"""Bodies of the research config kinds: ``serving_profile``, ``cluster_profile``,
``assist_loop_preset`` and ``near_dup_profile``.

Each is the validated body of a revisioned config document (``extra='forbid'``: a misspelt option is
a 422, never ignored). The assist-loop preset reuses the loop request's own sub-models, so a preset
field and the same request field cannot drift apart.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.schemas.curation.assist_loop import LoopAudit, LoopBudget, LoopGate, LoopRecursion
from src.schemas.curation.config_common import ActiveRef
from src.schemas.curation.near_duplicates import NearDupProfileBody


class _Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


# --- serving_profile: deployment-wide, one active per deployment ---------------------------------


class ServingServer(_Strict):
    """One Triton server, one per card."""

    name: str = Field(min_length=1)
    grpc_url: str = Field(min_length=1)
    metrics_url: str = Field(min_length=1)
    gpu_uuids: list[str] = Field(min_length=1, description='NVML UUIDs of the cards it serves.')
    numa_node: int | None = Field(default=None, ge=0)


class ServingImagePrep(_Strict):
    instances: int = Field(ge=1)
    workers: int = Field(ge=1)
    max_in_flight: int = Field(ge=1)
    device_budget_mib: int = Field(ge=0)
    min_megapixels: float = Field(default=0, ge=0)


class ServingPriorities(_Strict):
    interactive: int = Field(ge=0)
    bulk: int = Field(ge=0)


class ServingProfileBody(_Strict):
    servers: list[ServingServer] = Field(min_length=1)
    load_list: dict[str, list[str]] = Field(
        description='Per server name, the models it keeps resident.'
    )
    instances: dict[str, int] = Field(description='Instances per model.')
    image_prep: ServingImagePrep
    priorities: ServingPriorities

    @model_validator(mode='after')
    def _names_agree(self) -> ServingProfileBody:
        names = [s.name for s in self.servers]
        if len(set(names)) != len(names):
            msg = 'server names must be unique'
            raise ValueError(msg)
        unknown = sorted(set(self.load_list) - set(names))
        if unknown:
            msg = f'load_list names servers that do not exist: {unknown}'
            raise ValueError(msg)
        if any(count < 1 for count in self.instances.values()):
            msg = 'every model runs at least one instance'
            raise ValueError(msg)
        return self


# --- cluster_profile: project-scoped, two activation axes (item and region box) ------------------


class ClusterRefreshRule(_Strict):
    growth_threshold: float = Field(gt=0, description='Growth since the last run that starts one.')
    max_deferral_s: int = Field(ge=0)
    retry_backoff_s: int = Field(ge=0)


class ClusterProfileBody(_Strict):
    target: Literal['item', 'region_box']
    level: str = Field(min_length=1)
    embedding_field: str = Field(min_length=1)
    method: Literal['ivf', 'minibatch_kmeans']
    params: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    pool: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    assign: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    refresh: ClusterRefreshRule
    activation: Literal['auto_when_safe', 'human'] = 'human'
    backend: Literal['cpu', 'gpu_if_validated'] = 'cpu'


# --- assist_loop_preset: project-scoped, the project's default preset -----------------------------


class AssistLoopPresetBody(_Strict):
    strategy: Literal['cluster_sampled'] = 'cluster_sampled'
    gate: LoopGate = Field(default_factory=LoopGate)
    recursion: LoopRecursion = Field(default_factory=LoopRecursion)
    budget: LoopBudget
    on_budget: Literal['complete', 'pause'] = 'pause'
    audit: LoopAudit = Field(default_factory=LoopAudit)
    vlm_concurrency: int = Field(default=16, ge=1, le=128)
    node_concurrency: int = Field(default=8, ge=1, le=32)


RESEARCH_BODY_MODELS: dict[str, type[BaseModel]] = {
    'serving_profile': ServingProfileBody,
    'cluster_profile': ClusterProfileBody,
    'assist_loop_preset': AssistLoopPresetBody,
    'near_dup_profile': NearDupProfileBody,
}


# --- the wire of the routes ----------------------------------------------------------------------


class ResearchConfigDoc(BaseModel):
    kind: str
    name: str
    revision: int
    description: str
    body: dict[str, Any]
    created_at: str
    updated_at: str
    updated_by: str


class ResearchConfigList(BaseModel):
    items: list[ResearchConfigDoc]


class ResearchConfigPut(BaseModel):
    model_config = ConfigDict(extra='forbid')

    body: dict[str, Any]
    description: str = ''
    expected_revision: int | None = Field(
        default=None,
        ge=1,
        description='null creates the document; a number saves the next revision.',
    )


class ResearchActivateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    axis: str | None = Field(
        default=None, description='The axis; the first one of the kind when omitted.'
    )
    revision: int | None = Field(default=None, ge=1, description='null = the current revision.')
    expected_active: ActiveRef | None = Field(
        default=None, description='What is active now; null only when nothing is.'
    )


class ResearchActive(BaseModel):
    axis: str
    state: Literal['none', 'active', 'off']
    name: str | None = None
    revision: int | None = None
    activated_at: str | None = None


class ServingActivateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    revision: int | None = Field(default=None, ge=1, description='null = the current revision.')
    expected_revision: int | None = Field(
        default=None,
        ge=1,
        description='The setting revision read from GET /serving_profiles/active; null if none.',
    )


class ServingActive(BaseModel):
    name: str
    revision: int
    setting_revision: int
    body: dict[str, Any]
