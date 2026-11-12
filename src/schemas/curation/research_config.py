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

    name: str = Field(min_length=1, description='The name `load_list` refers to; unique.')
    grpc_url: str = Field(min_length=1, description='host:port clients send inference to.')
    metrics_url: str = Field(min_length=1, description='host:port of its metrics endpoint.')
    gpu_uuids: list[str] = Field(min_length=1, description='NVML UUIDs of the cards it serves.')
    numa_node: int | None = Field(
        default=None, ge=0, description='The NUMA node it is pinned to; null = not pinned.'
    )


class ServingImagePrep(_Strict):
    """How images are decoded and resized before inference."""

    instances: int = Field(ge=1, description='Image-prep model instances.')
    workers: int = Field(ge=1, description='Worker threads per instance.')
    max_in_flight: int = Field(ge=1, description='Images prepared at once before callers wait.')
    device_budget_mib: int = Field(
        ge=0, description='Device memory the image prep may use, in MiB.'
    )
    min_megapixels: float = Field(
        default=0, ge=0, description='Images smaller than this are prepared on the CPU.'
    )


class ServingPriorities(_Strict):
    """Triton request priority per traffic class (a lower number is served first)."""

    interactive: int = Field(ge=0, description='Priority of requests a person is waiting on.')
    bulk: int = Field(ge=0, description='Priority of ingest and batch work.')


class ServingProfileBody(_Strict):
    servers: list[ServingServer] = Field(
        min_length=1, description='The Triton servers, one per card.'
    )
    load_list: dict[str, list[str]] = Field(
        description='Per server name, the models it keeps resident.'
    )
    instances: dict[str, int] = Field(description='Instances per model.')
    image_prep: ServingImagePrep = Field(description='Image decode and resize settings.')
    priorities: ServingPriorities = Field(description='Request priorities per traffic class.')

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
    max_deferral_s: int = Field(
        ge=0, description='Longest a run waits for ingest to go quiet, in seconds.'
    )
    retry_backoff_s: int = Field(
        ge=0, description='Wait before retrying a run that failed or was cancelled, in seconds.'
    )


class ClusterProfileBody(_Strict):
    target: Literal['item', 'region_box'] = Field(
        description='What is clustered; it must match the axis the profile is activated on.'
    )
    level: str = Field(min_length=1, description="The partition level, 'class' by default.")
    embedding_field: str = Field(
        min_length=1, description='The vector field (the primary space of the target) to cluster.'
    )
    method: Literal['ivf', 'minibatch_kmeans'] = Field(description='The clustering method.')
    params: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict, description='Options of the method, such as n_clusters.'
    )
    pool: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict, description='Stored with the profile; the fit does not read it yet.'
    )
    assign: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict, description='Stored with the profile; the fit does not read it yet.'
    )
    refresh: ClusterRefreshRule = Field(description='When a new run starts.')
    activation: Literal['auto_when_safe', 'human'] = Field(
        default='human',
        description='auto_when_safe: a new partition becomes serving when its diff is safe; '
        'human: a person activates it.',
    )
    backend: Literal['cpu', 'gpu_if_validated'] = Field(
        default='cpu', description='gpu_if_validated uses the GPU only where it was validated.'
    )


# --- assist_loop_preset: project-scoped, the project's default preset -----------------------------


class AssistLoopPresetBody(_Strict):
    strategy: Literal['cluster_sampled'] = Field(
        default='cluster_sampled', description='How crops are picked: one sample per cluster.'
    )
    gate: LoopGate = Field(default_factory=LoopGate)
    recursion: LoopRecursion = Field(default_factory=LoopRecursion)
    budget: LoopBudget = Field(description='Hard caps; the loop never goes past one.')
    on_budget: Literal['complete', 'pause'] = Field(
        default='pause',
        description="'pause': a reached limit pauses the loop; 'complete': the nodes the budget "
        'cannot cover are left unvisited.',
    )
    audit: LoopAudit = Field(default_factory=LoopAudit)
    vlm_concurrency: int = Field(
        default=16, ge=1, le=128, description='VLM requests in flight at once.'
    )
    node_concurrency: int = Field(
        default=8, ge=1, le=32, description='Cluster nodes worked on at once.'
    )


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


class ResearchAxis(BaseModel):
    """A place a project kind can be active; a kind may be active on several."""

    axis: str = Field(description='The id `activate` and `active` take.')
    label: str
    target: str | None = Field(
        description='What the axis serves (a cluster profile body names its `target`); null when '
        'the axis does not depend on the body.'
    )


class ResearchConfigSchema(BaseModel):
    kind: str
    axes: list[ResearchAxis] = Field(
        description='Where the kind can be activated; empty for the deployment-wide serving profile.'
    )
    json_schema: dict[str, Any] = Field(
        description='The JSON Schema (draft 2020-12) of the document body, generated from the model '
        'the server validates it with.'
    )


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


class ServingActiveState(BaseModel):
    """``GET /serving_profiles/active``: the active profile, or ``null`` when none was activated."""

    active: ServingActive | None
