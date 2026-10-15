"""Wire models of ``/assist/runs``: the request, the run state and the result.

One request shape serves every scope. ``target`` says what is in scope,
``sample`` what the VLM sees, ``actions`` what to do. Cross-field refusals the
types cannot express are :func:`validate_request` (a structured 422).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.domain.assist_sampling import MAX_ASSIST_ITEMS, SAMPLE_HARD_MAX
from src.services.curation.assist.proposal_models import Proposal, ScopeNow
from src.services.curation.item_filter import ItemFilter
from src.services.curation.item_selection import MAX_SELECTION_IDS
from src.services.rejections import rejection


AssistAction = Literal['classify', 'attributes']
RunStatus = Literal[
    'dry_run',
    'queued',
    'running',
    'completed',
    'completed_with_errors',
    'failed',
    'cancelled',
    'interrupted',
]
ACTIVE_STATUSES = frozenset({'queued', 'running'})


class AssistContext(BaseModel):
    """The (sub-)cluster a selection of crops belongs to, so the run's result
    is listed with that scope."""

    model_config = ConfigDict(extra='forbid')

    cluster_id: int
    cluster_subid: str | None = None


class CropsTarget(BaseModel):
    """A single crop (one id) or a user selection (many)."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['crops']
    crop_ids: list[str] = Field(min_length=1, max_length=MAX_SELECTION_IDS)
    context: AssistContext | None = None


class ClusterTarget(BaseModel):
    """A cluster, or one sub-cluster of it, addressed by the stored ids."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['cluster']
    cluster_id: int
    cluster_subid: str | None = None


class FilterTarget(BaseModel):
    """Every item a non-empty item filter matches."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['filter']
    filter: ItemFilter


AssistTarget = Annotated[CropsTarget | ClusterTarget | FilterTarget, Field(discriminator='kind')]


class AssistSample(BaseModel):
    """What the VLM sees. ``all``: every member of the scope. ``representative``:
    a seeded sample whose size grows with the scope (``size`` overrides), stratified
    by sub-cluster when the scope has several."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['all', 'representative'] = 'all'
    size: int | None = Field(default=None, ge=1, le=SAMPLE_HARD_MAX)
    seed: int = 0
    stratify_by: Literal['auto', 'cluster_subid', 'none'] = 'auto'


class AssistRunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    target: AssistTarget
    sample: AssistSample = Field(default_factory=AssistSample)
    actions: list[AssistAction] = Field(min_length=1)
    vlm: str | None = Field(
        default=None, description="A VLM endpoint name; null = the project's active endpoint."
    )
    prompt_pack: str | None = Field(
        default=None, description="A prompt pack name; null = the project's effective default."
    )
    acknowledge_external: bool = False
    min_agreement: float = Field(
        default=0.6, ge=0.5, le=1.0, description="The proposal gate's agreement floor."
    )
    dry_run: bool = True


def validate_request(request: AssistRunRequest) -> None:
    """The refusals the field types cannot express (422 ``invalid_request``)."""
    target = request.target
    if len(set(request.actions)) != len(request.actions):
        raise _invalid('actions must not repeat')
    if isinstance(target, CropsTarget):
        if request.sample.mode != 'all':
            raise _invalid(
                'a crops target is its own sample: sample.mode must be "all"; use a cluster or '
                'filter target for a representative sample'
            )
        if len(set(target.crop_ids)) != len(target.crop_ids):
            raise _invalid('crop_ids must not repeat')
    if isinstance(target, FilterTarget) and target.filter.is_empty():
        raise _invalid(
            'a filter target needs at least one filter field; a whole-project run is the '
            'auto-label job'
        )
    if request.sample.mode == 'all' and request.sample.size is not None:
        raise _invalid('sample.size applies to sample.mode "representative"')


def _invalid(message: str) -> Exception:
    return rejection(422, 'invalid_request', message)


# ---- run state -----------------------------------------------------------------


class StratumInfo(BaseModel):
    key: str
    size: int
    sampled: int


class SkippedCounts(BaseModel):
    """Members of the target the run did not send to the VLM, and why."""

    holdout: int = 0
    excluded: int = 0
    not_found: int = 0
    class_locked: int = Field(
        default=0,
        description='Sent anyway for suggestions; their class is never written (`classify` only).',
    )


class ResolvedScope(BaseModel):
    scope_count: int = Field(description='Members of the target after the visibility rules.')
    scope_digest: str = Field(description='sha256 over the sorted member ids.')
    sampled: int
    sample_seed: int
    strata: list[StratumInfo] = Field(default_factory=list)
    skipped: SkippedCounts = Field(default_factory=SkippedCounts)
    vlm: dict[str, Any] | None = None
    prompt_pack: dict[str, Any] | None = None
    estimated_vlm_calls: int = 0


class AssistProgress(BaseModel):
    processed: int = 0
    total: int = 0


class ClassifyResult(BaseModel):
    predicted: int = 0
    updated: int = 0
    failed: int = 0
    unreadable: int = 0
    locked_suggest_only: int = Field(
        default=0, description='Answers on items whose class is locked: stored as suggestions only.'
    )


class AttributesResult(BaseModel):
    items_written: int = 0
    invalid_values: int = 0


class AssistResult(BaseModel):
    classify: ClassifyResult | None = None
    attributes: AttributesResult | None = None
    proposal: Proposal | None = Field(
        default=None, description='The sampled answers read as a label for the scope.'
    )


class AssistRunState(BaseModel):
    run_id: str | None = None
    status: RunStatus
    created_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    request: AssistRunRequest
    resolved: ResolvedScope
    progress: AssistProgress = Field(default_factory=AssistProgress)
    result: AssistResult | None = None
    scope_now: ScopeNow | None = Field(
        default=None,
        description='Set on reads for a cluster target or a run with a context: the scope '
        'now against the members the run resolved.',
    )


class AssistRunList(BaseModel):
    runs: list[AssistRunState]


__all__ = [
    'ACTIVE_STATUSES',
    'MAX_ASSIST_ITEMS',
    'AssistAction',
    'AssistContext',
    'AssistProgress',
    'AssistResult',
    'AssistRunList',
    'AssistRunRequest',
    'AssistRunState',
    'AssistSample',
    'AssistTarget',
    'AttributesResult',
    'ClassifyResult',
    'ClusterTarget',
    'CropsTarget',
    'FilterTarget',
    'ResolvedScope',
    'RunStatus',
    'SkippedCounts',
    'StratumInfo',
    'validate_request',
]
