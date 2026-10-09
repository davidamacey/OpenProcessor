"""Errors raised by the Triton promote/unload pipeline.

Split out of :mod:`src.services.training.triton_promote`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Errors
# =============================================================================


class PromoteError(RuntimeError):
    """Top of the local error hierarchy. Wraps the underlying cause."""

    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class CheckpointNotFoundError(PromoteError):
    """The job's status.json doesn't point at a valid ONNX export."""

    def __init__(self, job_id: str, expected_path: Path) -> None:
        super().__init__(
            f'job {job_id}: ONNX export not found at {expected_path}; did the '
            'trainer finish its exporting phase?',
            status_code=404,
        )


class ModelNameConflictError(PromoteError):
    """A model with this Triton name already exists; refuse to overwrite."""

    def __init__(self, triton_name: str) -> None:
        super().__init__(
            f'a Triton model named {triton_name!r} already exists; rename '
            'and retry, or delete the existing model first',
            status_code=409,
        )


class TritonLoadError(PromoteError):
    """Triton's load endpoint returned an error."""


class ModelNotPromotedError(PromoteError):
    """No on-disk model repo directory exists for this name.

    Distinguishes "nothing to unload" from a Triton-side failure — the
    caller gets a clean 404 instead of us attempting an unload against a
    name that was never promoted through this service.
    """

    def __init__(self, triton_name: str) -> None:
        super().__init__(
            f'no promoted model directory found for {triton_name!r} under the Triton model repo',
            status_code=404,
        )


class ClassRemapUnreadableError(PromoteError):
    """A class_remap payload was present but unparseable, empty, or
    internally inconsistent. This used to silently fall back to the
    full class registry with only a log line — now a loud 422 instead,
    since a mislabeled ``labels.txt`` is a serving-correctness bug."""

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(
            f'class_remap from {source} is present but unreadable: {reason}',
            status_code=422,
        )


class ClassRemapMissingError(PromoteError):
    """A subset/single_cls run has no resolvable class_remap from any source.

    "None" is illegal for a subset run — falling back to the full
    registry here has, historically, produced a ``labels.txt`` with all
    classes for a model that was actually trained on a handful. Bypass
    only via ``force=true``.
    """

    def __init__(self, job_id: str) -> None:
        super().__init__(
            f'job {job_id!r} was trained with include_classes/single_cls but no '
            'class_remap could be resolved from the manifest or the checkpoint '
            'weights dir; refusing to promote with a mislabeled labels.txt '
            '(pass force=true to bypass — logged distinctly)',
            status_code=422,
        )


class TritonUnloadError(PromoteError):
    """Triton's unload endpoint returned an error, or was unreachable.

    Deliberately fatal (unlike ``_trigger_load``'s fail-soft-on-timeout):
    the caller is about to ``rmtree`` a model directory next, and doing
    that without confirmation Triton actually released the model risks
    Triton's repository index disagreeing with what's on disk. Refuse to
    delete anything unless Triton told us, unambiguously, that it's safe.
    """
