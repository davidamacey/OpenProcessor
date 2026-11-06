"""The performance summary a throughput job returns in ``jobs.result`` (GPU research S-8).

``bulk_ingest_run``, ``reprocess`` and ``directory_ingest`` return one so the scale ladders read
throughput from the job record instead of a side script. The model is validated when a handler
returns it; an unknown decode route or a missing summary fails the job, so a typo cannot reach
a benchmark table. New decode routes are added to :data:`DECODE_ROUTES` with the lane that
creates them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


if TYPE_CHECKING:
    from collections.abc import Mapping

DECODE_ROUTES: tuple[str, ...] = (
    'gpu',
    'gpu_large',
    'gpu_hw_jpeg',
    'cpu_small',
    'cpu_large',
    'cpu_progressive',
    'cpu_decoder',
    'cpu_gpu_memory',
    'cpu_call_failed',
    'cpu_not_image',
)
PERF_SUMMARY_KINDS: frozenset[str] = frozenset({'bulk_ingest_run', 'reprocess', 'directory_ingest'})
RESULT_KEY = 'perf_summary'


class JobPerfSummary(BaseModel):
    model_config = ConfigDict(extra='forbid')

    images: int = Field(ge=0)
    wall_s: float = Field(ge=0)
    images_per_s: float = Field(ge=0)
    decode_routes: dict[str, int]
    stage_seconds: dict[str, float]
    pe_calls_per_image: float = Field(ge=0)
    servers: list[str]

    @field_validator('decode_routes')
    @classmethod
    def _closed_list(cls, routes: dict[str, int]) -> dict[str, int]:
        unknown = sorted(set(routes) - set(DECODE_ROUTES))
        if unknown:
            raise ValueError(f'unknown decode route {unknown}; the closed list is {DECODE_ROUTES}')
        if any(count < 0 for count in routes.values()):
            raise ValueError('decode route counts are not negative')
        return routes


class ResultCheckError(ValueError):
    """A handler's result breaks the rule of its kind."""


def check_result(kind: str, result: Mapping[str, Any] | None) -> None:
    """The kind's result check: the throughput kinds must carry a valid summary."""
    if kind not in PERF_SUMMARY_KINDS:
        return
    if result is None or RESULT_KEY not in result:
        raise ResultCheckError(f'a {kind} result needs a {RESULT_KEY!r}')
    try:
        JobPerfSummary.model_validate(result[RESULT_KEY])
    except ValidationError as exc:
        raise ResultCheckError(f'{RESULT_KEY} is invalid: {exc.errors()[0]["msg"]}') from exc
