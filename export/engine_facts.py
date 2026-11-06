"""The facts an export job records about each engine it builds (research GPU S-1..S-4, S-13).

* the **declared** input contract of a detector engine (what a caller must send), checked
  against the **built** plan: a plan is strongly typed, so its input is the ONNX input
  (``export/check_input_dtype.py``), and a plan whose dtype or size differs from the contract
  fails the job with ``contract_mismatch`` instead of loading and answering wrongly;
* the output contract, read from the plan;
* hashes of the plan and of the ONNX it was built from;
* the device memory the engine takes, measured by loading it once into Triton and reading
  Triton's own ``nv_gpu_memory_used_bytes`` before and after
  (https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html).

Nothing here imports TensorRT at module import; :func:`read_plan` imports it when called.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.schemas.models import EngineInputContract, EngineOutputContract


if TYPE_CHECKING:
    from pathlib import Path


IMG_SIZE = 640
CHANNELS = 3
PRECISION = 'fp16'  # the recipe of every detector build: ModelOpt AutoCast bakes FP16 into the ONNX
_HASH_CHUNK = 1 << 20
_TRT_DTYPES = {
    'UINT8': 'uint8',
    'HALF': 'fp16',
    'FLOAT': 'fp32',
    'INT32': 'int32',
    'INT64': 'int64',
}


class ContractMismatchError(RuntimeError):
    """The built plan does not match the contract its model declares; the job fails closed."""

    def __init__(self, detail: str) -> None:
        super().__init__(f'contract_mismatch: {detail}')


@dataclass(frozen=True)
class PlanTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]  # -1 marks a dynamic dimension


@dataclass(frozen=True)
class PlanFacts:
    """What a serialized engine says about its own inputs and outputs."""

    inputs: tuple[PlanTensor, ...]
    outputs: tuple[PlanTensor, ...]
    trt_version: str
    gpu_arch: str


def declared_input(size: int = IMG_SIZE) -> EngineInputContract:
    """The detector input: the letterboxed uint8 RGB image of ``src/services/detection/geometry.py``
    (PIL bilinear resize, fill 114, centred with the floor of half the pad), scaled in the graph."""
    return EngineInputContract(
        dtype='uint8',
        layout='NCHW',
        size=(size, size),
        color='RGB',
        normalize='in_graph',
        resize='letterbox',
        interpolation='pil_bilinear',
        fill=114,
        pad_split='floor_half',
    )


def declared_output_kind(engine_name: str, head_kind: str) -> str:
    """``end2end_boxes`` when NMS is in the graph (the NMS-free head, or ``*_trt_end2end``)."""
    if head_kind == 'one2one' or engine_name.endswith('_trt_end2end'):
        return 'end2end_boxes'
    return 'anchors_raw'


def read_plan(plan_path: Path) -> PlanFacts:
    """Deserialize ``plan_path`` and describe its I/O tensors (TensorRT Python API)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan_path.read_bytes())
    if engine is None:
        raise RuntimeError(f'could not deserialize engine {plan_path}')
    inputs: list[PlanTensor] = []
    outputs: list[PlanTensor] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        tensor = PlanTensor(
            name,
            _TRT_DTYPES.get(engine.get_tensor_dtype(name).name, engine.get_tensor_dtype(name).name),
            tuple(engine.get_tensor_shape(name)),
        )
        (inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(
            tensor
        )
    return PlanFacts(tuple(inputs), tuple(outputs), trt.__version__, _gpu_arch())


def _gpu_arch() -> str:
    import pynvml

    pynvml.nvmlInit()
    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(
            pynvml.nvmlDeviceGetHandleByIndex(0)
        )
    finally:
        pynvml.nvmlShutdown()
    return f'sm_{major}{minor}'


def check_contract(plan: PlanFacts, declared: EngineInputContract) -> int:
    """Raise :class:`ContractMismatchError` unless the plan's one input is the declared one.

    Returns the input size (the side of the square input) for the registry row.
    """
    if len(plan.inputs) != 1:
        raise ContractMismatchError(f'expected one input tensor, the plan has {len(plan.inputs)}')
    tensor = plan.inputs[0]
    if tensor.dtype != declared.dtype:
        raise ContractMismatchError(
            f'input {tensor.name} is {tensor.dtype}, the contract declares {declared.dtype}'
        )
    want = declared.size
    if not isinstance(want, tuple):
        raise ContractMismatchError('the declared input size is a range, not a fixed square')
    side = (want[0], want[1])
    if len(tensor.shape) != 4 or tensor.shape[1] != CHANNELS or tensor.shape[2:] != side:
        raise ContractMismatchError(
            f'input {tensor.name} has shape {list(tensor.shape)}, the contract declares '
            f'[-1, {CHANNELS}, {side[0]}, {side[1]}]'
        )
    return tensor.shape[2]


def output_contract(plan: PlanFacts, kind: str) -> EngineOutputContract:
    return EngineOutputContract(
        kind=kind,
        shapes=[list(t.shape) for t in plan.outputs],
        dtypes=[t.dtype for t in plan.outputs],
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def build_stamp(max_batch: int, opt_batch: int, size: int, flags: str, dtype: str) -> str:
    """The same text ``scripts/lib/model_stamps.sh`` writes for the other engines."""
    return f'min1/opt{opt_batch}/max{max_batch} {size}x{size} {flags} strongly-typed input={dtype}'


def total_gpu_memory_mib(metrics_text: str) -> float:
    """Sum of Triton's ``nv_gpu_memory_used_bytes`` over its cards, in MiB."""
    from prometheus_client.parser import text_string_to_metric_families

    total = 0.0
    for family in text_string_to_metric_families(metrics_text):
        if family.name == 'nv_gpu_memory_used_bytes':
            total += sum(sample.value for sample in family.samples)
    return total / (1024 * 1024)


def measured_mib(before: float, after: float) -> int | None:
    """The memory one load added; ``None`` when it did not grow (already resident)."""
    delta = after - before
    return round(delta) if delta > 0 else None


def contracts_for(
    plan: PlanFacts, *, engine_name: str, head_kind: str
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Check ``plan`` against the declared contract and return the two jsonb values and the size."""
    declared = declared_input()
    size = check_contract(plan, declared)
    out = output_contract(plan, declared_output_kind(engine_name, head_kind))
    return declared.model_dump(mode='json'), out.model_dump(mode='json'), size
