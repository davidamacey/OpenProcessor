"""Wire models of the promote routes: ``POST /train/promote/{job_id}``, its job-status
poll and ``/reload_promoted``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PromoteRequest(BaseModel):
    """Body for ``POST {api_prefix}/train/promote/{job_id}``."""

    triton_name: str = Field(
        ...,
        description=(
            'Desired Triton model name (alphanumeric, underscore, hyphen). The model is served as '
            '`<project>__<triton_name>` (no prefix for `default`); the final name is returned as '
            '`triton_name` in the promote result and job status.'
        ),
        min_length=1,
        max_length=64,
    )
    max_batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description='Triton dynamic-batch upper bound. Must match what the '
        'ONNX export was done with (default 8).',
    )
    input_size: int = Field(
        default=640,
        ge=64,
        le=2048,
        description='Square input dim. Must match the ONNX export.',
    )
    fp16: bool = Field(
        default=True,
        description='Use FP16 precision in the JIT-compiled TensorRT engine.',
    )
    overwrite: bool = Field(
        default=False,
        description='If a Triton model with this name exists, delete it first.',
    )
    force: bool = Field(
        default=False,
        description=(
            'Bypass the promote gate (mAP50 ≥ 0.65, no class precision < 0.50, '
            'no class with support < 5). Use only for known-good experimental runs.'
        ),
    )


class PromoteResponse(BaseModel):
    job_id: str
    triton_name: str
    onnx_path: str
    config_path: str
    labels_path: str
    triton_loaded: bool
    force_used: bool = False
    gate_report: dict[str, Any] | None = None
    lineage_stamped: bool = False
    class_remap_source: str = 'none'
    # Truthful static hint, not a live measurement: see
    # TritonPromoter.PromoteResult.cold_start_expected_on_first_inference.
    cold_start_expected_on_first_inference: bool = True


class PromoteJobStatus(BaseModel):
    """A background promote (``POST /train/promote/{job_id}`` without ``wait=true``)."""

    promote_id: str
    job_id: str
    triton_name: str
    status: Literal['queued', 'exporting', 'loading', 'building', 'warming', 'done', 'failed']
    error: str | None = None
    error_status: int | None = Field(
        default=None, description='HTTP status a synchronous promote would have returned'
    )
    result: PromoteResponse | None = None
    started_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    poll_after_s: int | None = None


class ReloadPromotedResponse(BaseModel):
    status: str
    reloaded: list[str] = []
    failed: list[str] = []
