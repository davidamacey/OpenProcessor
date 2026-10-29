"""Wire models of the auto-label loop (``/assist/loops``).

The request is the loop's whole configuration: it is validated once at start, stored
normalized in the loop's ``request.json`` and read back by the worker, so what runs is what
was accepted. ``extra='forbid'`` everywhere: a misspelt option is a 422, never ignored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoopScope(BaseModel):
    model_config = ConfigDict(extra='forbid')

    cluster_ids: list[int] | None = Field(
        default=None,
        description='Clusters to work on; null = every candidate cluster of the project.',
    )
    min_root_size: int = Field(
        default=40, ge=1, description='Smaller clusters are left alone (no root).'
    )


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
    vlm_concurrency: int = Field(default=16, ge=1, le=128)
    node_concurrency: int = Field(default=8, ge=1, le=32)
    dry_run: bool = True


__all__ = [
    'AssistLoopRequest',
    'LoopAudit',
    'LoopBudget',
    'LoopGate',
    'LoopRecursion',
    'LoopScope',
]
