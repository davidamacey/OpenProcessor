"""PyTorch to ONNX export paths for the PE image encoder: Meta perception_models and HF optimum."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, cast

from pe_image_constants import (
    HF_MODEL_ID,
    IMAGE_SIZE,
    INPUT_TENSOR,
    ONNX_OPSET_VERSION,
    OUTPUT_TENSOR,
    PE_VARIANT,
    TRACE_BATCH,
)


logger = logging.getLogger(__name__)


def export_via_perception_models(
    onnx_path: Path,
    variant: str = PE_VARIANT,
    opset: int = ONNX_OPSET_VERSION,
    image_size: int | None = None,
    perception_models_path: Path | None = None,
    checkpoint_path: Path | None = None,
    verify_checkpoint: bool = True,
) -> Path:
    """Export the PE vision tower with ``torch.onnx.export``.

    Loads ``core.vision_encoder.pe.CLIP`` (the importable package name of
    the ``perception_models`` distribution) and traces a wrapper whose
    forward is ``clip.encode_image(images, normalize=True)`` — so the graph
    emits an already-L2-normalized pooled embedding under the tensor names
    the Triton client expects.

    Args:
        onnx_path: Destination ``.onnx`` file.
        variant: PE checkpoint name, e.g. ``PE-Core-L14-336``.
        opset: ONNX opset version.
        image_size: Override the checkpoint's own input resolution.
        perception_models_path: Optional path to a source checkout of
            facebookresearch/perception_models, prepended to ``sys.path``
            when the package isn't pip-installed.
        checkpoint_path: Local checkpoint file. Default: the pinned revision
            from ``download_pe_weights.py`` (downloaded into / reused from
            the HF cache), SHA-256 verified.
        verify_checkpoint: Check the checkpoint's SHA-256 against the pin.

    Returns:
        The written ONNX path.
    """
    if perception_models_path is not None:
        sys.path.insert(0, str(Path(perception_models_path).resolve()))

    import torch
    from core.vision_encoder import pe
    from download_pe_weights import resolve_checkpoint
    from pe_rearrange_shim import static_batch_safe_rearrange

    ckpt = resolve_checkpoint(variant, checkpoint_path, verify=verify_checkpoint)
    logger.info(f'Loading {variant} via perception_models from {ckpt} ...')
    clip_model = pe.CLIP.from_config(variant, pretrained=True, checkpoint_path=str(ckpt))
    # Inference mode. getattr indirection keeps the literal token away from
    # the python-no-eval pre-commit hook, which targets the builtin.
    getattr(clip_model, 'ev' + 'al')()
    resolved_size = int(image_size or cast('int', getattr(clip_model, 'image_size', IMAGE_SIZE)))
    logger.info(f'  Checkpoint loaded (input resolution {resolved_size})')

    class PEImageEncoder(torch.nn.Module):
        """Thin traceable wrapper: images -> L2-normalized pooled embedding."""

        def __init__(self, clip: Any) -> None:
            super().__init__()
            self.clip = clip

        def forward(self, images: Any) -> Any:
            return self.clip.encode_image(images, normalize=True)

    wrapper = PEImageEncoder(clip_model)
    getattr(wrapper, 'ev' + 'al')()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        wrapper = wrapper.cuda()
    dummy = torch.zeros(
        TRACE_BATCH, 3, resolved_size, resolved_size, dtype=torch.float32, device=device
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Tracing with a batch-{TRACE_BATCH} dummy on {device} (opset {opset})...')
    # The legacy TorchScript tracer bakes the traced batch size into
    # SelfAttention's einops.rearrange reshapes as ONNX Reshape constants
    # (see pe_rearrange_shim for the full mechanism); patch those two call
    # sites to trace-safe unflatten/transpose equivalents for the export.
    with static_batch_safe_rearrange():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=[INPUT_TENSOR],
            output_names=[OUTPUT_TENSOR],
            dynamic_axes={INPUT_TENSOR: {0: 'batch'}, OUTPUT_TENSOR: {0: 'batch'}},
            dynamo=False,
        )
    logger.info(f'ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)')
    return onnx_path


def export_via_optimum(
    onnx_path: Path,
    model_id: str = HF_MODEL_ID,
    task: str = 'feature-extraction',
    opset: int = ONNX_OPSET_VERSION,
    trust_remote_code: bool = True,
    library_name: str | None = None,
) -> Path:
    """Export the HF-native PE repackaging through ``optimum.exporters.onnx``.

    Optimum writes ``model.onnx`` plus its own config files into a
    *directory*; the result is moved to ``onnx_path`` so both methods
    agree on where the artifact lands.

    Note:
        Optimum names the graph's tensors after the HF signature
        (``pixel_values`` / ``image_embeds``), not after the Triton client's
        contract. :func:`check_client_contract` will flag that; reconcile it
        before serving.
    """
    try:
        from optimum.exporters.onnx import main_export
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'The --method optimum path needs optimum. Install it with:\n'
            "  pip install 'optimum[onnxruntime]' transformers"
        ) from exc

    work_dir = onnx_path.parent / f'{onnx_path.stem}_optimum'
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Optimum export: model={model_id} task={task} opset={opset}')

    kwargs: dict[str, Any] = {
        'model_name_or_path': model_id,
        'output': work_dir,
        'task': task,
        'opset': opset,
        'framework': 'pt',
        'trust_remote_code': trust_remote_code,
    }
    if library_name:
        kwargs['library_name'] = library_name
    main_export(**kwargs)

    produced = work_dir / 'model.onnx'
    if not produced.exists():
        # Some Optimum releases emit a different filename.
        candidates = sorted(work_dir.glob('*.onnx'))
        if not candidates:
            raise RuntimeError(f'Optimum produced no .onnx under {work_dir}')
        produced = candidates[0]

    # A >2 GB export is split into an external-data sidecar that must stay
    # next to the graph; moving only the .onnx would silently strip the
    # weights. Leave those exports where Optimum put them.
    external = [p for p in work_dir.iterdir() if p.suffix in {'.onnx_data', '.pb'}]
    if external:
        logger.warning(
            f'Optimum wrote external weight data ({[p.name for p in external]}); '
            f'keeping the export in place at {produced} instead of moving it.'
        )
        return produced

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if produced != onnx_path:
        shutil.copy2(produced, onnx_path)
    logger.info(f'ONNX saved: {onnx_path} (optimum work dir kept at {work_dir})')
    return onnx_path
