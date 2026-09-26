"""Promote a trained YOLO26 checkpoint into the Triton model repo.

The workflow:

1. The user clicks "Promote to Triton" on a finished training run in
   the labeler frontend's train page.
2. Frontend calls ``POST {api_prefix}/train/promote/{job_id}``.
3. The router calls :func:`promote_yolo26_to_triton` here.
4. We read ``status.json`` for the run's ONNX export (the trainer
   produces ``best.onnx`` alongside ``best.pt`` during the
   ``exporting`` state).
5. Copy the ONNX into the Triton model repo at
   ``models/<triton_name>/1/model.onnx``.
6. Write ``config.pbtxt`` (rendered by
   :mod:`src.services.training.yolo_triton_config`) and
   ``labels.txt``.
7. POST ``/v2/repository/models/<triton_name>/load`` to make the model
   active immediately.

This is intentionally separate from ``src/routers/models.py`` —
that router is YOLO11-specific (NMS plugin, TRT engine builder).
YOLO26's NMS is internal to the forward pass, and Ultralytics' own
guidance is to ship ONNX and let Triton's TensorRT execution
accelerator JIT-compile. We need different scaffolding.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from src.core.logging import get_logger
from src.services.training.yolo_triton_config import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_MAX_BATCH,
    Yolo26TritonConfig,
    render_config,
    render_labels_file,
)


if TYPE_CHECKING:
    from src.services.training.jobs import TrainJobStatus


logger = get_logger(__name__)


# Same default as src/routers/models.py — the Triton model repo mounted
# into the API container. Override at construction time for tests.
#
# This used to be a plain module-level constant, so a deployment
# that mounts the Triton repo somewhere other than /app/models (some
# deployment overlays mount it at /models) silently wrote
# promoted models into a directory Triton never sees, with no error —
# the copy + config-write both "succeed" against a path in the
# container's writable layer. Resolving OP_TRITON_MODEL_REPO here, at
# construction time rather than import time, means the env var set for
# this container is always honored, and tests can still monkeypatch
# os.environ before constructing a TritonPromoter.
DEFAULT_TRITON_MODELS_DIR = Path('/app/models')


def resolve_triton_models_dir() -> Path:
    """``OP_TRITON_MODEL_REPO``, falling back to :data:`DEFAULT_TRITON_MODELS_DIR`."""
    override = os.environ.get('OP_TRITON_MODEL_REPO')
    return Path(override) if override else DEFAULT_TRITON_MODELS_DIR


# Triton's HTTP control endpoint. The yolo-api container shares the
# triton_net network so this resolves through Docker DNS.
DEFAULT_TRITON_HTTP_URL = 'http://triton-server:8000'


def resolve_triton_http_url() -> str:
    """``OP_TRITON_HTTP_URL`` (falling back to the legacy ``TRITON_HTTP_URL``
    name if that's the only one set anywhere in this deployment), else
    :data:`DEFAULT_TRITON_HTTP_URL`.
    """
    return (
        os.environ.get('OP_TRITON_HTTP_URL')
        or os.environ.get('TRITON_HTTP_URL')
        or DEFAULT_TRITON_HTTP_URL
    )


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


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass(frozen=True)
class PromoteResult:
    """Returned to the API caller."""

    job_id: str
    triton_name: str
    onnx_path: str
    config_path: str
    labels_path: str
    triton_loaded: bool
    version: str = '1'
    class_remap_source: str = 'none'
    # Always true for this promoter's onnxruntime+TensorRT-accelerator
    # config.pbtxt (see yolo_triton_config.py): Triton's /load only
    # loads the ONNX graph -- the TensorRT execution accelerator JIT-
    # builds the actual engine on the model's first real inference
    # request, synchronously, on that request's thread. Final E2E run
    # 2026-09-26 measured ~85s for this on a toy single-class model; a
    # caller scripting immediate post-promote verification should expect
    # a slow (not hung) first call and can optionally issue a throwaway
    # warm-up request before treating latency as representative.
    cold_start_expected_on_first_inference: bool = True


@dataclass(frozen=True)
class UnloadResult:
    """Returned to the API caller by :meth:`TritonPromoter.unload`."""

    triton_name: str
    triton_unloaded: bool
    directory_removed: bool


# =============================================================================
# Service
# =============================================================================


class TritonPromoter:
    """Promote a trained YOLO26 checkpoint into Triton.

    Threading: this service is stateless after construction; safe to
    instantiate per-request inside a FastAPI handler (no async lock
    needed). The two filesystem mutations (copy + writes) happen on a
    thread executor so we don't block the event loop on slow NFS.
    """

    def __init__(
        self,
        *,
        triton_models_dir: Path | None = None,
        triton_http_url: str | None = None,
        # 30s was too short for a real TensorRT JIT-build on /load (the
        # engine gets compiled synchronously on first load for a model
        # the .onnx cache doesn't already have a plan for). 300s
        # comfortably covers a cold build for models in this pipeline's
        # size range; the fail-soft-on-connection-error/5xx vs
        # hard-fail-on-4xx distinction in _trigger_load is unchanged.
        http_timeout: float = 300.0,
    ) -> None:
        # Resolved from OP_TRITON_MODEL_REPO / OP_TRITON_HTTP_URL at
        # construction time (not import time), so a caller that doesn't
        # pass these explicitly still gets whatever this container's env
        # actually says, and tests can monkeypatch os.environ per-test.
        self.triton_models_dir = (
            triton_models_dir if triton_models_dir is not None else resolve_triton_models_dir()
        )
        self.triton_http_url = (
            triton_http_url if triton_http_url is not None else resolve_triton_http_url()
        ).rstrip('/')
        self.http_timeout = http_timeout

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def promote(
        self,
        *,
        status: TrainJobStatus,
        triton_name: str,
        class_id_to_name: dict[int, str],
        max_batch_size: int = DEFAULT_MAX_BATCH,
        input_size: int = DEFAULT_INPUT_SIZE,
        fp16: bool = True,
        overwrite: bool = False,
        class_remap: ClassRemapResult | None = None,
    ) -> PromoteResult:
        """Run the promote pipeline end-to-end.

        Args:
            status: The trained run's TrainJobStatus (read by the caller
                from jobs.read_status). We read ``checkpoint_path`` to
                find the ONNX file.
            triton_name: Desired Triton model name (URL-safe;
                lowercase + underscore preferred).
            class_id_to_name: Maps the **trained** class ids (post any
                subset renumber) to their human-readable names. The
                caller looks this up in ``class_remap.json`` next to
                the checkpoint when the run was a subset training, or
                from the full registry for a full-class run.
            max_batch_size: Triton dynamic-batch upper bound. Default 8.
            input_size: Square input dim. Must match the model's export.
            fp16: TensorRT precision mode for the JIT-compiled engine.
            overwrite: If False (default), refuses to clobber an
                existing model of the same name (409). If True,
                deletes the directory first.

        Returns:
            :class:`PromoteResult` with the on-disk paths and load
            status.
        """
        if not triton_name or not triton_name.replace('_', '').replace('-', '').isalnum():
            msg = f'invalid triton_name {triton_name!r}: must be alphanumeric with optional _ or -'
            raise PromoteError(msg, status_code=400)

        onnx_src = self._locate_onnx(status)

        model_dir = self.triton_models_dir / triton_name
        existing_versions = self._existing_version_dirs(model_dir)

        # A model repo dir can exist without any served version yet (e.g.
        # a previous promote attempt that failed before copying weights
        # in, or a bare config-only checkout). Only gate on there being an
        # actual previously-promoted version — that's what `overwrite`
        # historically protected against.
        if existing_versions and not overwrite:
            raise ModelNameConflictError(triton_name)
        if existing_versions and overwrite:
            logger.warning(
                'train_promote_overwrite',
                triton_name=triton_name,
                existing_versions=[str(v) for v in existing_versions],
            )

        # Never rmtree an existing model dir before the new weights are
        # safely copied in — that made every promote destructive
        # in-place with no rollback if the copy failed partway. Triton
        # natively supports multiple numbered version subdirectories and
        # serves the highest by default (`config.pbtxt`'s default
        # `version_policy: { latest: { num_versions: 1 } }`), so we always
        # ADD a new version directory instead of clobbering. Prior
        # versions are left on disk untouched — if the copy or config
        # write fails, the previously-serving version is never touched
        # and Triton keeps serving it without interruption.
        next_version = (max(existing_versions) + 1) if existing_versions else 1
        version_dir = model_dir / str(next_version)
        await asyncio.to_thread(version_dir.mkdir, parents=True, exist_ok=False)

        # Copy ONNX in. shutil.copy2 preserves mtime for traceability;
        # we don't atomic-rename because a half-written file is
        # straight-up rejected by Triton on load anyway, so observable
        # damage is bounded. If the copy fails, roll back the *new*
        # version dir only — pre-existing versions are never touched, so
        # there's nothing to "restore": the model already on disk (if
        # any) was simply never modified.
        onnx_dst = version_dir / 'model.onnx'
        try:
            await asyncio.to_thread(shutil.copy2, onnx_src, onnx_dst)
        except Exception:
            logger.error(
                'train_promote_copy_failed_rolling_back',
                triton_name=triton_name,
                version=str(next_version),
                onnx_src=str(onnx_src),
            )
            await asyncio.to_thread(shutil.rmtree, version_dir, ignore_errors=True)
            raise

        # Render config + labels. These are shared across all versions
        # (Triton's model repo layout keeps config.pbtxt/labels.txt at the
        # model-dir level, not per-version), so a re-promote under the
        # same name updates them too — consistent with the new version
        # actually being what those files describe.
        config_path = model_dir / 'config.pbtxt'
        labels_path = model_dir / 'labels.txt'
        try:
            cfg = Yolo26TritonConfig(
                model_name=triton_name,
                input_size=input_size,
                max_batch_size=max_batch_size,
                fp16=fp16,
            )
            config_text = render_config(cfg)
            labels_text = render_labels_file(class_id_to_name)
            await asyncio.to_thread(config_path.write_text, config_text, encoding='utf-8')
            await asyncio.to_thread(labels_path.write_text, labels_text, encoding='utf-8')
        except Exception:
            logger.error(
                'train_promote_config_write_failed_rolling_back',
                triton_name=triton_name,
                version=str(next_version),
            )
            await asyncio.to_thread(shutil.rmtree, version_dir, ignore_errors=True)
            raise

        # Lineage back-pointer: given only a serving model dir, an
        # operator can trace it back to the run that produced it without
        # needing the API's own job store. Written before the Triton load
        # attempt so it's present even if the load itself times out or
        # 5xx's (fail-soft — the files are still valid and loadable later).
        promote_json_path = model_dir / 'promote.json'
        remap = class_remap or _NONE_REMAP
        backpointer = {
            'job_id': status.job_id,
            'triton_name': triton_name,
            'version': str(next_version),
            'promoted_at': datetime.now(tz=UTC).isoformat(),
            'class_remap': {
                'source': remap.source,
                'n_classes': len(remap.mapping)
                if remap.mapping
                else (1 if remap.single_cls else 0),
                'sha256': None,
            },
        }
        await asyncio.to_thread(
            promote_json_path.write_text,
            json.dumps(backpointer, indent=2, sort_keys=True),
            encoding='utf-8',
        )

        # Copy the resolved remap into the served model dir too, alongside
        # config.pbtxt/labels.txt, so an operator inspecting the served
        # model repo (not the training job store) can still see exactly
        # which original registry ids this model's dense ids map to.
        # Best-effort — a failure here doesn't roll back an otherwise
        # successful promote, but is logged loudly.
        if remap.source != 'none':
            try:
                remap_dest = model_dir / 'class_remap.json'
                remap_payload = {
                    'original_to_new': {str(k): v for k, v in remap.mapping.items()},
                    'single_cls': remap.single_cls,
                    'names': remap.names,
                    'include_classes': remap.include_classes,
                    'source': remap.source,
                    'job_id': status.job_id,
                    'version': str(next_version),
                    'resolved_at': datetime.now(tz=UTC).isoformat(),
                }
                await asyncio.to_thread(
                    remap_dest.write_text,
                    json.dumps(remap_payload, indent=2, sort_keys=True),
                    encoding='utf-8',
                )
            except OSError as exc:
                logger.error(
                    'train_promote_class_remap_copy_to_model_dir_failed',
                    triton_name=triton_name,
                    error=str(exc),
                )

        # Trigger Triton load.
        loaded = await self._trigger_load(triton_name)

        # F-42 (fresh-start E2E findings 2026-09-25, round 2): drop any
        # cached class-name mapping for this model name so the very next
        # detection response reads the labels.txt just written above,
        # not a stale mapping from a prior promote under the same name.
        from src.utils.class_names import invalidate_class_names

        invalidate_class_names(triton_name)

        logger.info(
            'train_promote_ok',
            job_id=status.job_id,
            triton_name=triton_name,
            onnx=str(onnx_dst),
            version=str(next_version),
            loaded=loaded,
        )

        return PromoteResult(
            job_id=status.job_id,
            triton_name=triton_name,
            onnx_path=str(onnx_dst),
            config_path=str(config_path),
            labels_path=str(labels_path),
            triton_loaded=loaded,
            version=str(next_version),
            class_remap_source=remap.source,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _existing_version_dirs(model_dir: Path) -> list[int]:
        """Return the numbered version subdirectories already under ``model_dir``.

        Triton's model repo layout numbers version dirs as plain integers
        (``1``, ``2``, ...); anything else (``config.pbtxt``, a stray
        ``promote.json`` from a prior promote, non-numeric dirs) is
        ignored. Empty list means "no version has ever been promoted here"
        even if the directory itself exists.
        """
        if not model_dir.is_dir():
            return []
        versions = [
            int(entry.name)
            for entry in model_dir.iterdir()
            if entry.is_dir() and entry.name.isdigit()
        ]
        return sorted(versions)

    @staticmethod
    def _locate_onnx(status: TrainJobStatus) -> Path:
        """Resolve the trained ONNX path from the run's status.

        The trainer writes ``best.onnx`` next to ``best.pt`` during the
        ``exporting`` state. ``status.checkpoint_path`` may carry
        either; we coerce to the .onnx form.
        """
        ckpt = getattr(status, 'checkpoint_path', None)
        if not ckpt:
            raise CheckpointNotFoundError(status.job_id, Path('<unknown>'))
        ckpt_path = Path(ckpt)
        if ckpt_path.suffix == '.onnx':
            onnx_path = ckpt_path
        else:
            onnx_path = ckpt_path.with_suffix('.onnx')
        if not onnx_path.is_file():
            raise CheckpointNotFoundError(status.job_id, onnx_path)
        return onnx_path

    async def _trigger_load(self, triton_name: str) -> bool:
        """POST ``/v2/repository/models/<name>/load`` to Triton.

        Returns True on 200, False on non-fatal errors (so the API
        caller can surface the on-disk paths even if Triton's polling
        isn't enabled — a follow-up manual ``model load`` will pick up
        the files we just wrote).
        """
        url = f'{self.triton_http_url}/v2/repository/models/{triton_name}/load'
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                resp = await client.post(url)
        except httpx.HTTPError as exc:
            logger.warning(
                'train_promote_triton_load_unreachable',
                triton_name=triton_name,
                error=str(exc),
            )
            return False
        if resp.status_code == 200:
            return True
        logger.error(
            'train_promote_triton_load_failed',
            triton_name=triton_name,
            status=resp.status_code,
            body=resp.text[:500],
        )
        # 4xx is a load-config error worth surfacing; 5xx may be transient.
        if 400 <= resp.status_code < 500:
            raise TritonLoadError(
                f'triton refused load for {triton_name!r}: {resp.status_code} {resp.text[:200]}',
                status_code=502,
            )
        return False

    # ------------------------------------------------------------------
    # Unload — the mirror of promote(): reuses the same
    # ``/v2/repository/models/<name>/{load,unload}`` Triton control
    # endpoint and the same on-disk model repo layout, so this is the one
    # place either operation is implemented.
    # ------------------------------------------------------------------

    async def unload(self, triton_name: str) -> UnloadResult:
        """Unload ``triton_name`` from Triton and remove its repo directory.

        Caller-side guardrails (active-model pointer, protected-model
        checks) are the router's job, not this method's — this is the
        mechanical "make Triton let go, then delete the files" half
        only. Order matters: we only ``rmtree`` *after* Triton confirms
        the unload succeeded (or that the model wasn't loaded in the
        first place), so we never delete files out from under a model
        Triton still thinks is live.
        """
        model_dir = self.triton_models_dir / triton_name
        if not await asyncio.to_thread(model_dir.is_dir):
            raise ModelNotPromotedError(triton_name)

        unloaded = await self._trigger_unload(triton_name)
        if not unloaded:
            raise TritonUnloadError(
                f'triton did not confirm unload for {triton_name!r}; refusing to '
                'delete its model repo directory',
                status_code=502,
            )

        await asyncio.to_thread(shutil.rmtree, model_dir, ignore_errors=False)
        logger.info('train_unload_ok', triton_name=triton_name, model_dir=str(model_dir))

        # F-42: same cache-invalidation as promote() -- an unloaded
        # model's labels must not linger in memory either (harmless if
        # the name is never reused, but stale otherwise).
        from src.utils.class_names import invalidate_class_names

        invalidate_class_names(triton_name)

        return UnloadResult(
            triton_name=triton_name,
            triton_unloaded=True,
            directory_removed=True,
        )

    async def _trigger_unload(self, triton_name: str) -> bool:
        """POST ``/v2/repository/models/<name>/unload`` to Triton.

        Unlike :meth:`_trigger_load`, a connection failure here returns
        False (not a soft-pass) — see :class:`TritonUnloadError`'s
        docstring for why the caller must treat that as fatal rather than
        proceeding to delete files anyway.
        """
        url = f'{self.triton_http_url}/v2/repository/models/{triton_name}/unload'
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                resp = await client.post(url)
        except httpx.HTTPError as exc:
            logger.warning(
                'train_unload_triton_unreachable',
                triton_name=triton_name,
                error=str(exc),
            )
            return False
        if resp.status_code == 200:
            return True
        logger.error(
            'train_unload_triton_failed',
            triton_name=triton_name,
            status=resp.status_code,
            body=resp.text[:500],
        )
        return False


# =============================================================================
# Public API
# =============================================================================


async def promote_yolo26_to_triton(
    *,
    status: TrainJobStatus,
    triton_name: str,
    class_id_to_name: dict[int, str],
    max_batch_size: int = DEFAULT_MAX_BATCH,
    input_size: int = DEFAULT_INPUT_SIZE,
    fp16: bool = True,
    overwrite: bool = False,
    class_remap: ClassRemapResult | None = None,
    promoter: TritonPromoter | None = None,
) -> PromoteResult:
    """Convenience wrapper. The router uses this; tests pass a custom promoter."""
    p = promoter or TritonPromoter()
    return await p.promote(
        status=status,
        triton_name=triton_name,
        class_id_to_name=class_id_to_name,
        max_batch_size=max_batch_size,
        input_size=input_size,
        fp16=fp16,
        overwrite=overwrite,
        class_remap=class_remap,
    )


async def unload_triton_model(
    triton_name: str,
    *,
    promoter: TritonPromoter | None = None,
) -> UnloadResult:
    """Convenience wrapper. The router uses this; tests pass a custom promoter.

    Caller-side guardrails (active-model pointer, protected-model checks)
    belong in the router, not here — see :meth:`TritonPromoter.unload`.
    """
    p = promoter or TritonPromoter()
    return await p.unload(triton_name)


async def reload_promoted_models(promoter: TritonPromoter | None = None) -> dict[str, Any]:
    """Re-``/load`` every promoted model Triton doesn't report READY.

    Triton in explicit-control mode only loads its ``--load-model`` list
    at startup. A model promoted through this module stays on disk (its
    ``promote.json`` marker is how :func:`src.routers.curation.models.
    _discover_promoted_models` finds it) but drops to UNAVAILABLE after
    any Triton restart until someone POSTs ``/load`` again. Call this
    once at API startup (see ``src/main.py``'s lifespan) so a Triton
    restart doesn't silently strand every previously-promoted model.

    Best-effort throughout: a scan failure, an unreachable Triton, or a
    single model's load failure is logged and folded into the return
    value rather than raised — this must never block API startup.
    """
    p = promoter or TritonPromoter()
    try:
        entries = sorted(p.triton_models_dir.iterdir())
    except OSError as exc:
        logger.warning('reload_promoted_models_scan_failed', error=str(exc))
        return {'status': 'error', 'error': str(exc), 'reloaded': [], 'failed': []}

    promoted_names = [e.name for e in entries if e.is_dir() and (e / 'promote.json').is_file()]
    if not promoted_names:
        return {'status': 'ok', 'reloaded': [], 'failed': []}

    ready_names: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f'{p.triton_http_url}/v2/repository/index')
        if resp.status_code == 200:
            ready_names = {entry['name'] for entry in resp.json() if entry.get('state') == 'READY'}
        else:
            logger.warning('reload_promoted_models_index_failed', status=resp.status_code)
    except httpx.HTTPError as exc:
        logger.warning('reload_promoted_models_index_unreachable', error=str(exc))
        return {'status': 'error', 'error': str(exc), 'reloaded': [], 'failed': []}

    reloaded: list[str] = []
    failed: list[str] = []
    for name in promoted_names:
        if name in ready_names:
            continue
        try:
            ok = await p._trigger_load(name)
        except TritonLoadError as exc:
            logger.warning('reload_promoted_model_failed', name=name, error=str(exc))
            ok = False
        (reloaded if ok else failed).append(name)

    if reloaded or failed:
        logger.info('reload_promoted_models_done', reloaded=reloaded, failed=failed)
    return {'status': 'ok', 'reloaded': reloaded, 'failed': failed}


async def reload_promoted_models_best_effort(*, log_event: str) -> None:
    """:func:`reload_promoted_models`, but never raises and logs for you.

    Shared by ``src.main``'s lifespan startup, its periodic GPU-arbiter
    reconcile tick, and (indirectly, via the plain
    :func:`reload_promoted_models` call) ``POST
    {api_prefix}/train/reload_promoted`` -- a scan failure or unreachable
    Triton is logged and swallowed so it can run unattended on both
    startup and every reconcile tick without ever taking the loop down.
    """
    try:
        reload_result = await reload_promoted_models()
        if reload_result.get('reloaded') or reload_result.get('failed'):
            logger.info(
                log_event,
                reloaded=reload_result.get('reloaded'),
                failed=reload_result.get('failed'),
            )
    except Exception as exc:
        logger.warning('promoted_models_reload_skipped', error=str(exc))


__all__ = [
    'DEFAULT_TRITON_HTTP_URL',
    'DEFAULT_TRITON_MODELS_DIR',
    'CheckpointNotFoundError',
    'ClassRemapMissingError',
    'ClassRemapResult',
    'ClassRemapUnreadableError',
    'ModelNameConflictError',
    'ModelNotPromotedError',
    'PromoteError',
    'PromoteResult',
    'TritonLoadError',
    'TritonPromoter',
    'TritonUnloadError',
    'UnloadResult',
    'build_class_id_to_name',
    'promote_yolo26_to_triton',
    'reload_promoted_models',
    'reload_promoted_models_best_effort',
    'resolve_class_remap',
    'resolve_triton_http_url',
    'resolve_triton_models_dir',
    'unload_triton_model',
]


# =============================================================================
# class_remap resolution
# =============================================================================


@dataclass(frozen=True)
class ClassRemapResult:
    """A resolved, validated class_remap payload.

    ``mapping`` is ``{original_registry_class_id: new_dense_class_id}`` —
    the same contract the trainer's subset-dataset builder has always
    produced, just parsed into a typed object instead of a bare dict.
    ``source`` records which of the two resolution paths won
    (``'manifest'`` / ``'weights_dir'``), surfaced to the caller as
    ``class_remap_source`` so a promote result is traceable.
    """

    mapping: dict[int, int]
    names: list[str] | None
    single_cls: bool
    include_classes: list[int] | None
    source: str  # 'manifest' | 'weights_dir' | 'none'


_NONE_REMAP = ClassRemapResult(
    mapping={}, names=None, single_cls=False, include_classes=None, source='none'
)


def _parse_class_remap_payload(raw: Any, *, source: str) -> ClassRemapResult:
    """Parse a class_remap payload, real shape first.

    The trainer's subset-dataset builder writes ``{original_to_new,
    new_to_original, single_cls, names, include_classes}`` — a previous
    parser here only understood a flat ``{orig: new}`` dict or a
    ``{'mapping': {...}}`` wrapper, so ``int(k)`` failed on every real key
    and this always silently degraded to "no remap". Old flat/``mapping``
    shapes are still accepted for backward compat with anything that wrote
    them directly. Raises :class:`ClassRemapUnreadableError` — never
    returns ``None`` — on anything unparseable or empty.
    """
    if not isinstance(raw, dict):
        raise ClassRemapUnreadableError(source, f'not a JSON object: {type(raw).__name__}')

    if 'original_to_new' in raw:
        raw_mapping = raw.get('original_to_new')
        names = raw.get('names')
        single_cls = bool(raw.get('single_cls', False))
        include_classes = raw.get('include_classes')
    elif isinstance(raw.get('mapping'), dict):
        raw_mapping = raw['mapping']
        names = raw.get('names')
        single_cls = bool(raw.get('single_cls', False))
        include_classes = raw.get('include_classes')
    else:
        # Legacy flat {orig: new} shape.
        raw_mapping = raw
        names = None
        single_cls = False
        include_classes = None

    if not isinstance(raw_mapping, dict):
        raise ClassRemapUnreadableError(source, "no 'original_to_new'/'mapping' dict found")

    mapping: dict[int, int] = {}
    for k, v in raw_mapping.items():
        try:
            mapping[int(k)] = int(v)
        except (TypeError, ValueError) as exc:
            raise ClassRemapUnreadableError(source, f'non-integer key/value {k!r}: {v!r}') from exc

    if not mapping:
        raise ClassRemapUnreadableError(source, 'mapping is empty')

    return ClassRemapResult(
        mapping=mapping,
        names=list(names) if isinstance(names, list) else None,
        single_cls=single_cls,
        include_classes=[int(c) for c in include_classes]
        if isinstance(include_classes, list)
        else None,
        source=source,
    )


def resolve_class_remap(
    *,
    job_id: str,
    checkpoint_path: Path,
    manifest: dict[str, Any] | None,
) -> ClassRemapResult:
    """Resolve a job's class_remap, manifest first, then the weights dir.

    Resolution order:
        (a) ``manifest['lineage']['class_remap']`` — works for every
            already-completed run, since the trainer has always captured
            this pre-``rmtree`` at manifest-write time.
        (b) ``<checkpoint_path's dir>/class_remap.json`` — written by the
            trainer directly into the weights dir for runs after this fix.
        (c) neither present: :data:`_NONE_REMAP` (source='none') — legal
            only for a full-class run; the caller enforces that.

    A payload that exists but fails to parse is a loud failure
    (:class:`ClassRemapUnreadableError`), never a silent fall-through to
    the next source — an unreadable-but-present remap for a subset run is
    exactly the bug this rewrite fixes.
    """
    lineage = (manifest or {}).get('lineage') or {}
    manifest_remap = lineage.get('class_remap')
    if manifest_remap is not None:
        try:
            return _parse_class_remap_payload(manifest_remap, source='manifest')
        except ClassRemapUnreadableError:
            logger.error('train_promote_class_remap_manifest_unreadable', job_id=job_id)
            raise

    weights_dir_path = checkpoint_path.parent / 'class_remap.json'
    if weights_dir_path.is_file():
        try:
            raw = json.loads(weights_dir_path.read_text(encoding='utf-8'))
        except OSError as exc:
            logger.error(
                'train_promote_class_remap_weights_dir_unreadable', job_id=job_id, error=str(exc)
            )
            raise ClassRemapUnreadableError('weights_dir', str(exc)) from exc
        except ValueError as exc:
            logger.error(
                'train_promote_class_remap_weights_dir_unreadable', job_id=job_id, error=str(exc)
            )
            raise ClassRemapUnreadableError('weights_dir', f'invalid JSON: {exc}') from exc
        try:
            return _parse_class_remap_payload(raw, source='weights_dir')
        except ClassRemapUnreadableError:
            logger.error('train_promote_class_remap_weights_dir_unreadable', job_id=job_id)
            raise

    return _NONE_REMAP


def build_class_id_to_name(
    *,
    remap: ClassRemapResult,
    full_registry: dict[int, str],
) -> dict[int, str]:
    """Resolve the **post-training** class_id → name map for ``labels.txt``.

    For a full-class run (``remap.source == 'none'``), returns
    ``full_registry`` as-is. For a subset run, applies the remap to
    renumber to ``0..N-1``. ``single_cls`` collapses to a single-line
    ``labels.txt``.
    """
    if remap.source == 'none':
        return dict(full_registry)
    if remap.single_cls:
        single_name = remap.names[0] if remap.names else 'object'
        return {0: single_name}
    out: dict[int, str] = {}
    for orig_id, new_id in remap.mapping.items():
        registry_name = full_registry.get(orig_id)
        if registry_name is None:
            continue
        out[new_id] = registry_name
    return out
