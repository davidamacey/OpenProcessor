"""Wire models of the auto-label loop (``/assist/loops``).

The request is the loop's whole configuration: it is validated once at start, stored
normalized in the loop's ``request.json`` and read back by the worker, so what runs is what
was accepted. ``extra='forbid'`` everywhere: a misspelt option is a 422, never ignored.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoopScope(BaseModel):
    model_config = ConfigDict(extra='forbid')

    cluster_uids: list[str] | None = Field(
        default=None,
        description=(
            'Clusters (their `cluster_uid`) of the serving partition to work on; '
            'null = every cluster of it.'
        ),
    )
    min_root_size: int = Field(
        default=40, ge=1, description='Smaller clusters are left alone (no root).'
    )

    @field_validator('cluster_uids')
    @classmethod
    def _uids_are_uuids(cls, uids: list[str] | None) -> list[str] | None:
        """Canonical lower-case uuid text, so the stored request matches the index's keywords."""
        if uids is None:
            return None
        try:
            return [str(uuid.UUID(u)) for u in uids]
        except ValueError as exc:
            msg = f'cluster_uids must be cluster uuids: {exc}'
            raise ValueError(msg) from exc


class LoopGate(BaseModel):
    """When a sampled cluster may pass its class to all its other members."""

    model_config = ConfigDict(extra='forbid')

    theta: float = Field(
        default=0.9, ge=0.7, le=0.99, description='Homogeneity the cluster must be shown to have.'
    )
    alpha: float = Field(
        default=0.05,
        ge=0.001,
        le=0.1,
        description='Chance allowed of passing a cluster below theta.',
    )
    max_dissent: int = Field(
        default=1, ge=0, le=3, description='Sampled answers allowed to disagree with the top class.'
    )


class LoopRecursion(BaseModel):
    """How a cluster that fails the gate is split and tried again."""

    model_config = ConfigDict(extra='forbid')

    max_depth: int = Field(default=3, ge=0, le=6)
    branching: int = Field(default=4, ge=2, le=8)
    min_node_size: int = Field(
        default=40, ge=2, description='Smaller failing nodes are left for review.'
    )


class LoopBudget(BaseModel):
    """Hard caps; the loop never goes past one. ``max_vlm_crops`` has no default."""

    model_config = ConfigDict(extra='forbid')

    max_vlm_crops: int = Field(ge=1)
    max_vlm_requests: int | None = Field(
        default=None, ge=1, description='null = crops / images per call, plus the concurrency.'
    )
    max_wall_s: float | None = Field(default=None, gt=0, description='null = 6 hours.')


class LoopAudit(BaseModel):
    """A second model re-checks a sample of the labels the loop propagated."""

    model_config = ConfigDict(extra='forbid')

    vlm: str | None = Field(
        default=None, description="The audit model's endpoint name; null = the audit is off."
    )
    acknowledge_external: bool = False
    rate: float = Field(default=0.01, gt=0, le=1)
    min_per_batch: int = Field(default=30, ge=1)
    max_crops: int = Field(default=500, ge=1, description="The audit's own crop budget.")
    floor: float = Field(default=0.85, gt=0, lt=1)
    on_fail: Literal['pause', 'rollback'] = 'pause'


class AssistLoopRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    scope: LoopScope = Field(default_factory=LoopScope)
    vlm: str | None = Field(
        default=None, description="A VLM endpoint name; null = the project's active endpoint."
    )
    prompt_pack: str | None = Field(
        default=None, description="A prompt pack name; null = the project's effective default."
    )
    acknowledge_external: bool = False
    seed: int = 0
    gate: LoopGate = Field(default_factory=LoopGate)
    recursion: LoopRecursion = Field(default_factory=LoopRecursion)
    budget: LoopBudget | None = Field(
        default=None, description='Required to start; a dry run sizes the loop without one.'
    )
    audit: LoopAudit = Field(default_factory=LoopAudit)
    on_budget: Literal['pause', 'complete'] = Field(
        default='pause',
        description=(
            "'pause': a reached limit pauses the loop (raise it with the budget route, then "
            "resume); 'complete': the nodes the budget cannot cover are left unvisited."
        ),
    )
    vlm_concurrency: int = Field(default=16, ge=1, le=128)
    node_concurrency: int = Field(default=8, ge=1, le=32)
    dry_run: bool = True


# ---------------------------------------------------------------------- the record

LoopStatus = Literal[
    'dry_run',
    'queued',
    'running',
    'paused',
    'cancelling',
    'completed',
    'completed_with_errors',
    'cancelled',
    'failed',
    'interrupted',
]


class LoopVlmPin(BaseModel):
    """An endpoint the loop is pinned to: the revision it was accepted against."""

    name: str | None = None
    revision: int | None = None
    model: str | None = None


class LoopPackPin(BaseModel):
    name: str | None = None
    revision: int | None = None


class LoopPinned(BaseModel):
    vlm: LoopVlmPin
    prompt_pack: LoopPackPin
    audit_vlm: LoopVlmPin | None = Field(
        default=None, description='null: the audit is off (stated, not silent).'
    )


class LoopRequiredSample(BaseModel):
    """Answers the gate needs for a cluster of ``n_members`` eligible members, by the number
    of sampled answers allowed to disagree (index = dissent)."""

    n_members: int
    required_by_dissent: list[int]


class LoopPlan(BaseModel):
    roots: int = Field(description='Clusters the loop would work on.')
    eligible_items: int = Field(description='Members it could ask about and write.')
    required_sample_table: list[LoopRequiredSample] = Field(
        description='Per distinct cluster size (largest first, at most 25 rows).'
    )
    estimated_vlm_crops_root_level: int = Field(
        description='VLM crops the root-level draws need if every cluster passes with no dissent.'
    )
    estimated_vlm_requests_root_level: int
    fits_budget: bool | None = Field(
        default=None, description='Whether that estimate fits max_vlm_crops; null without a budget.'
    )


class LoopPartition(BaseModel):
    kind: str
    digest: str
    snapshot_at: str
    roots: int
    eligible_items: int


class LoopUnitBudget(BaseModel):
    limit: int | None = Field(description='null: uncapped.')
    reserved: int
    spent: int


class LoopPoolBudget(BaseModel):
    max_crops: int
    max_requests: int
    crops_spent: int
    crops_reserved: int
    requests_spent: int
    requests_reserved: int
    prompt_tokens: LoopUnitBudget | None = Field(
        default=None, description='Reserved as an estimate; spent from the endpoint usage.'
    )
    completion_tokens: LoopUnitBudget | None = None


class LoopBudgetView(BaseModel):
    vlm: LoopPoolBudget
    audit: LoopPoolBudget | None = None
    wall_s: float
    max_wall_s: float | None = None
    usage_missing: int = Field(
        default=0, description='Answered calls whose response had no usage; charged in full.'
    )


class AssistLoopControl(BaseModel):
    """What may be done with the loop now; decided by the backend."""

    state: Literal[
        'queued',
        'running',
        'pausing',
        'paused',
        'cancelling',
        'completed',
        'completed_with_errors',
        'failed',
        'cancelled',
        'interrupted',
    ] = Field(description="'pausing' is a running loop with a pause requested.")
    can_pause: bool
    can_resume: bool
    can_cancel: bool
    can_edit_budget: bool
    can_rollback: bool
    resume_blocked_reason: Literal['project_paused', 'vlm_unavailable'] | None = None


class LoopBudgetLimit(BaseModel):
    model_config = ConfigDict(extra='forbid')

    pool: Literal['vlm', 'audit']
    unit: Literal['crops', 'requests', 'prompt_tokens', 'completion_tokens', 'wall_s']
    limit_value: int | None = Field(
        ge=0, description='null: uncapped (never for the vlm crop limit).'
    )


class AssistLoopBudgetEdit(BaseModel):
    model_config = ConfigDict(extra='forbid')

    limits: list[LoopBudgetLimit] = Field(min_length=1, max_length=10)


class LoopLevel(BaseModel):
    level: int
    nodes: int
    passed: int
    split: int
    review_leaf: int
    review_small: int
    stale: int
    unvisited: int
    items_propagated: int
    items_direct: int
    crops_spent: int


class LoopTotals(BaseModel):
    items_propagated: int
    items_direct: int
    clusters_left_for_review: int
    items_left_for_review: int
    skipped: dict[str, int]


class LoopAuditView(BaseModel):
    status: str = Field(description="'off', 'pending', 'ok', 'fail' or 'budget_exhausted'.")
    sampled: int = 0
    agreed: int = 0
    agreement: float | None = None
    lower_bound: float | None = None
    crops_spent: int = 0


class AssistLoopState(BaseModel):
    loop_id: str | None = Field(default=None, description='null on a dry run.')
    status: LoopStatus
    stop_reason: str | None = Field(
        default=None,
        description=(
            "'done', 'budget_crops', 'budget_requests' or 'budget_time' (completed); "
            "'vlm_failed' (failed)."
        ),
    )
    pause_reason: Literal['user', 'audit', 'budget', 'project_paused', 'vlm_unavailable'] | None = (
        Field(default=None)
    )
    control: AssistLoopControl | None = Field(default=None, description='null on a dry run.')
    iteration: int = 1
    slices: int = Field(default=0, description='Deliveries of the loop job so far.')
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str | None = None
    error: str | None = None
    request: AssistLoopRequest
    pinned: LoopPinned
    plan: LoopPlan
    partition: LoopPartition | None = None
    budget: LoopBudgetView | None = None
    levels: list[LoopLevel] = Field(default_factory=list)
    totals: LoopTotals | None = None
    audit: LoopAuditView | None = None


class AssistLoopList(BaseModel):
    loops: list[AssistLoopState]


class LoopVote(BaseModel):
    class_name: str
    votes: int


class LoopNewClassVote(BaseModel):
    name: str
    votes: int


LoopDecision = Literal['pass', 'split', 'review_leaf', 'review_small', 'stale', 'unvisited']


class LoopNode(BaseModel):
    path: str = Field(
        description='r<display_no> (the cluster card number), then .<child index> per level.'
    )
    level: int
    parent: str | None
    n: int = Field(description='Eligible members at the snapshot.')
    sampled: int
    answered: int
    decision: LoopDecision | None = Field(default=None, description='null while the node is open.')
    reason: str | None = None
    reason_message: str | None = Field(
        default=None, description='A sentence for `reason`; null when it has no wording.'
    )
    class_name: str | None = None
    agree: int | None = None
    bound: float | None = None
    distribution: list[LoopVote] = Field(default_factory=list)
    new_class_proposals: list[LoopNewClassVote] = Field(default_factory=list)
    items_propagated: int = 0


class LoopNodePage(BaseModel):
    nodes: list[LoopNode]
    next_after: str | None = Field(default=None, description='Pass as `after` for the next page.')


class LoopMembersPage(BaseModel):
    crop_ids: list[str]
    next_after: str | None = None


class LoopRollbackRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    node_paths: list[str] | None = Field(
        default=None, description='Only these nodes; null = everything the loop propagated.'
    )
    dry_run: bool = True


class LoopRollbackResponse(BaseModel):
    dry_run: bool
    matched: int = Field(description='Items the loop wrote that no rollback restored yet.')
    restored: int = Field(description='Restored, or that a dry run would restore.')
    kept_changed: int = Field(description='Left as they are: changed by someone since.')
    conflicts: int


__all__ = [
    'AssistLoopBudgetEdit',
    'AssistLoopControl',
    'AssistLoopList',
    'AssistLoopRequest',
    'AssistLoopState',
    'LoopAudit',
    'LoopAuditView',
    'LoopBudget',
    'LoopBudgetLimit',
    'LoopBudgetView',
    'LoopDecision',
    'LoopGate',
    'LoopLevel',
    'LoopMembersPage',
    'LoopNewClassVote',
    'LoopNode',
    'LoopNodePage',
    'LoopPackPin',
    'LoopPartition',
    'LoopPinned',
    'LoopPlan',
    'LoopPoolBudget',
    'LoopRecursion',
    'LoopRequiredSample',
    'LoopRollbackRequest',
    'LoopRollbackResponse',
    'LoopScope',
    'LoopStatus',
    'LoopTotals',
    'LoopUnitBudget',
    'LoopVlmPin',
    'LoopVote',
]
