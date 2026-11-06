"""The shipped Triton models as ready engines in the registry, with their contracts.

``db-migrate`` seeds one ``models`` row (origin ``builtin``, no project) and one ``model_engines``
row per model the repository ships (research GPU S-1, S-2, S-4), so a consumer reads what an
engine takes and returns from the registry instead of from its name. The semantics a
``config.pbtxt`` cannot say (resize, colour order, normalisation) are written here once, next to
the code that produces those inputs, and ``tests/promote/test_builtin_engine_contracts.py`` checks
dtype, dimensions, batch size and outputs against the shipped ``config.pbtxt`` so the two cannot
drift apart. Only engines get a row (``platform:`` models); the python-backend models
(``gpu_image_prep``, ``ocr_pipeline``) run no engine of their own.

Output shapes exclude the batch dimension, as in ``config.pbtxt``. A row seeded here records
``trt_version`` and ``gpu_arch`` as ``unrecorded`` and a null ``built_at``: it states the shipped
contract, not a measured build. ``device_mib_measured`` carries the model's own
``GPU_MIB_PER_INSTANCE`` where it declares one, until an export or a load measures it. The first
export of the same model writes a row with the real build keys next to this one (lane 5B owns
what the readers do with two rows of one name).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from src.db.models.training import Model, ModelEngine
from src.domain.storage_keys import make_key
from src.schemas.models import EngineInputContract, EngineOutputContract, NormalizeSpec, SizeRange


if TYPE_CHECKING:
    from sqlalchemy.orm import Session

MODELS_DIR = Path(__file__).resolve().parents[3] / 'models'
UNRECORDED = 'unrecorded'
CREATED_BY = 'seed'


class TokenInputContract(BaseModel):
    """A text encoder's input: token ids, not pixels."""

    model_config = ConfigDict(extra='forbid', frozen=True)
    kind: Literal['tokens'] = 'tokens'
    dtype: Literal['int64'] = 'int64'
    length: int | None  # ``None``: any length up to the tokenizer's context


@dataclass(frozen=True)
class Shipped:
    name: str
    family: str
    task: str
    precision: Literal['fp32', 'fp16', 'int8']
    input_size: int
    input_contract: EngineInputContract | TokenInputContract
    output_contract: EngineOutputContract


_YOLO_INPUT = EngineInputContract(
    dtype='uint8',
    layout='NCHW',
    size=(640, 640),
    color='RGB',
    normalize='in_graph',
    resize='letterbox',
    interpolation='pil_bilinear',
    fill=114,
    pad_split='floor_half',
)
_FACE_NORMALIZE = NormalizeSpec(mean=(127.5, 127.5, 127.5), std=(128.0, 128.0, 128.0), scale=1.0)
_OCR_NORMALIZE = NormalizeSpec(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5), scale=1.0)
_OCR_SIDE = SizeRange(min=32, opt=640, max=960)


def _embedding(dim: int) -> EngineOutputContract:
    return EngineOutputContract(kind='embedding', shapes=[[dim]], dtypes=['fp32'])


SHIPPED: tuple[Shipped, ...] = (
    Shipped(
        'yolov11_small_trt_end2end',
        'yolo11',
        'detect',
        'fp16',
        640,
        _YOLO_INPUT,
        EngineOutputContract(
            kind='end2end_boxes',
            shapes=[[1], [300, 4], [300], [300]],
            dtypes=['int32', 'fp32', 'fp16', 'int32'],
        ),
    ),
    Shipped(
        'yolo26_small_trt',
        'yolo26',
        'detect',
        'fp16',
        640,
        _YOLO_INPUT,
        EngineOutputContract(kind='end2end_boxes', shapes=[[300, 6]], dtypes=['fp32']),
    ),
    Shipped(
        'scrfd_10g_bnkps',
        'scrfd',
        'face_detect',
        'fp16',
        640,
        EngineInputContract(
            dtype='fp32',
            layout='NCHW',
            size=(640, 640),
            color='RGB',
            normalize=_FACE_NORMALIZE,
            resize='letterbox',
            interpolation='bilinear',
            fill=0,
        ),
        EngineOutputContract(
            kind='anchors_raw',
            shapes=[
                [12800, 1],
                [3200, 1],
                [800, 1],
                [12800, 4],
                [3200, 4],
                [800, 4],
                [12800, 10],
                [3200, 10],
                [800, 10],
            ],
            dtypes=['fp32'] * 9,
        ),
    ),
    Shipped(
        'arcface_w600k_r50',
        'arcface',
        'face_embed',
        'fp16',
        112,
        EngineInputContract(
            dtype='fp32',
            layout='NCHW',
            size=(112, 112),
            color='RGB',
            normalize=_FACE_NORMALIZE,
            resize='stretch',
            interpolation='bilinear',
        ),
        _embedding(512),
    ),
    Shipped(
        'mobileclip2_s2_image_encoder',
        'mobileclip2',
        'image_embed',
        'fp16',
        256,
        EngineInputContract(
            dtype='fp32',
            layout='NCHW',
            size=(256, 256),
            color='RGB',
            normalize=NormalizeSpec(
                mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), scale=0.00392156862745098
            ),
            resize='short_edge_center_crop',
            interpolation='bilinear',
        ),
        _embedding(512),
    ),
    Shipped(
        'mobileclip2_s2_text_encoder',
        'mobileclip2',
        'text_embed',
        'fp16',
        77,
        TokenInputContract(length=77),
        _embedding(512),
    ),
    Shipped(
        'pe_image_encoder',
        'pe_core',
        'image_embed',
        'fp16',
        336,
        EngineInputContract(
            dtype='uint8',
            layout='NCHW',
            size=(336, 336),
            color='RGB',
            normalize='in_graph',
            resize='short_edge_center_crop',
            interpolation='pil_bilinear',
        ),
        _embedding(1024),
    ),
    Shipped(
        'pe_text_encoder',
        'pe_core',
        'text_embed',
        'fp32',
        32,
        TokenInputContract(length=None),
        _embedding(1024),
    ),
    Shipped(
        'paddleocr_det_trt',
        'paddleocr',
        'ocr_detect',
        'fp16',
        960,
        EngineInputContract(
            dtype='fp32',
            layout='NCHW',
            size=(_OCR_SIDE, _OCR_SIDE),
            color='RGB',
            normalize=_OCR_NORMALIZE,
            resize='stretch',
            interpolation='bilinear',
        ),
        EngineOutputContract(kind='anchors_raw', shapes=[[1, -1, -1]], dtypes=['fp32']),
    ),
    Shipped(
        'paddleocr_rec_trt',
        'paddleocr',
        'ocr_recognize',
        'fp16',
        48,
        EngineInputContract(
            dtype='fp32',
            layout='NCHW',
            size=(48, SizeRange(min=32, opt=320, max=960)),
            color='RGB',
            normalize=_OCR_NORMALIZE,
            resize='native_width',
            interpolation='bilinear',
        ),
        EngineOutputContract(kind='ctc_logits', shapes=[[-1, 18385]], dtypes=['fp32']),
    ),
)


@dataclass(frozen=True)
class PbtxtFacts:
    max_batch: int
    device_mib: int | None


def read_pbtxt(models_dir: Path, name: str) -> PbtxtFacts:
    """``max_batch_size`` and ``GPU_MIB_PER_INSTANCE`` of the shipped ``config.pbtxt``."""
    text = (models_dir / name / 'config.pbtxt').read_text(encoding='utf-8')
    batch = re.search(r'^\s*max_batch_size:\s*(\d+)', text, re.MULTILINE)
    if batch is None:
        raise ValueError(f'{name}: config.pbtxt has no max_batch_size')
    mib = re.search(r'key:\s*"GPU_MIB_PER_INSTANCE"\s*value:\s*\{\s*string_value:\s*"(\d+)"', text)
    return PbtxtFacts(int(batch.group(1)), None if mib is None else int(mib.group(1)))


def _labels(models_dir: Path, name: str) -> list[str]:
    path = models_dir / name / 'labels.txt'
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _dump(contract: BaseModel) -> Any:
    return contract.model_dump(mode='json')


def seed_builtin_engines(session: Session, models_dir: Path = MODELS_DIR) -> list[str]:
    """Insert the missing builtin model and engine rows; return the Triton names inserted.

    Runs inside the caller's deployment transaction and never commits. Re-running inserts
    nothing; a row is never rewritten, so a measured value survives a later seed.
    """
    inserted: list[str] = []
    for shipped in SHIPPED:
        facts = read_pbtxt(models_dir, shipped.name)
        model = session.execute(
            select(Model).where(
                Model.project_id.is_(None), Model.name == shipped.name, Model.version == 1
            )
        ).scalar_one_or_none()
        if model is None:
            model = Model(
                project_id=None,
                name=shipped.name,
                version=1,
                origin='builtin',
                family=shipped.family,
                task=shipped.task,
                labels=_labels(models_dir, shipped.name),
                artifact_storage_key=make_key('models', shipped.name),
                created_by=CREATED_BY,
            )
            session.add(model)
            session.flush()
        engine = session.execute(
            select(ModelEngine.id).where(
                ModelEngine.triton_model_name == shipped.name,
                ModelEngine.trt_version == UNRECORDED,
                ModelEngine.gpu_arch == UNRECORDED,
                ModelEngine.build_profile == 'default',
            )
        ).first()
        if engine is not None:
            continue
        session.add(
            ModelEngine(
                model_id=model.id,
                triton_model_name=shipped.name,
                precision=shipped.precision,
                input_size=shipped.input_size,
                input_contract=_dump(shipped.input_contract),
                output_contract=_dump(shipped.output_contract),
                build_profile='default',
                build_settings={'source': 'seeded from the shipped config.pbtxt'},
                build_stamp='shipped',
                device_mib_measured=facts.device_mib,
                max_batch=facts.max_batch,
                trt_version=UNRECORDED,
                gpu_arch=UNRECORDED,
                status='ready',
            )
        )
        inserted.append(shipped.name)
    session.flush()
    return inserted
