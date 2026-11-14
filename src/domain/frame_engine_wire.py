"""Frame engine protocol, version 1: the tensor names and the request plan, shared by the engine
and the API (docs/design/v070_rebuild/13_backend_speed_plan.md, D1 "Engine protocol, version 1").

One request per image over KServe v2 ``ModelInfer`` to the model ``frame_engine``. Inputs:
``image`` (the encoded file, one BYTES element), ``plan`` (one JSON string, see :class:`EnginePlan`)
and the optional ``boxes`` (stored ``bbox_norm`` rows, FP64 ``[n, 4]``: the caller supplies the
detections and the detector is not called; FP64 because the stored values are Python floats and
the rectangle each one cuts must not move by a rounded pixel). Outputs: ``status``,
``decode_route`` and ``branch_status`` always, ``size`` once the image decoded, then each branch's
tensors (:data:`BRANCH_OUTPUTS`).

``branch_status`` is one JSON string, an object with exactly one entry per requested branch (and
``derivatives`` when the plan names renditions): ``{"status": "ok" | "not_run"}`` or
``{"status": "error", "reason": "<reason>"}``. Reasons: ``unsupported_branch`` (in the plan
vocabulary, not built in this engine), ``model_error`` (the model call failed),
``dependency_failed`` (a branch this one needs failed), ``image_failed`` (``status`` is not
``ok``: nothing ran).

The branch names are :data:`src.domain.processing_plan.BRANCHES`; the engine runs the subset in
:data:`ENGINE_BRANCHES`. ``derivatives`` is a plan field, not a branch, and is reported in
``branch_status`` under its own name. Pure rules and JSON, standard library only, so the engine
image carries this module without the API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from src.domain.decode_route import GPU
from src.domain.processing_plan import BRANCHES, CONDITIONAL, Branch


if TYPE_CHECKING:
    from collections.abc import Mapping

MODEL_NAME = 'frame_engine'
PROTOCOL_VERSION = 1

# Inputs
IMAGE = 'image'
PLAN = 'plan'
BOXES = 'boxes'

# Outputs, with their KServe datatypes
STATUS = 'status'
DECODE_ROUTE = 'decode_route'
BRANCH_STATUS = 'branch_status'
SIZE = 'size'
DET_BOXES = 'det_boxes'
DET_SCORES = 'det_scores'
DET_CLASSES = 'det_classes'
LETTERBOX = 'letterbox'
WHOLE_EMBEDDING = 'whole_embedding'
CROP_EMBEDDINGS = 'crop_embeddings'
CROP_EMBEDDED = 'crop_embedded'
BLUR_SUMS = 'blur_sums'
CROP_JPEGS = 'crop_jpegs'
FACE_BOXES = 'face_boxes'
FACE_SCORES = 'face_scores'
FACE_LANDMARKS = 'face_landmarks'
FACE_BLUR_SUMS = 'face_blur_sums'
FACE_OVERFLOW = 'face_overflow'
FACE_EMBEDDINGS = 'face_embeddings'
FACE_EMBEDDED = 'face_embedded'
OCR_QUADS = 'ocr_quads'
OCR_TEXTS = 'ocr_texts'
OCR_SCORES = 'ocr_scores'
OCR_CHAR_MIN = 'ocr_char_min'
OCR_DET_SCORES = 'ocr_det_scores'
OCR_OVERFLOW = 'ocr_overflow'
DERIVATIVE_JPEGS = 'derivative_jpegs'

INPUT_DATATYPES: Mapping[str, str] = {IMAGE: 'BYTES', PLAN: 'BYTES', BOXES: 'FP64'}
OUTPUT_DATATYPES: Mapping[str, str] = {
    STATUS: 'BYTES',
    DECODE_ROUTE: 'BYTES',
    BRANCH_STATUS: 'BYTES',
    SIZE: 'INT32',
    DET_BOXES: 'FP32',
    DET_SCORES: 'FP32',
    DET_CLASSES: 'INT32',
    LETTERBOX: 'FP64',
    WHOLE_EMBEDDING: 'FP32',
    CROP_EMBEDDINGS: 'FP32',
    CROP_EMBEDDED: 'BOOL',
    BLUR_SUMS: 'INT64',
    CROP_JPEGS: 'BYTES',
    FACE_BOXES: 'FP32',
    FACE_SCORES: 'FP32',
    FACE_LANDMARKS: 'FP32',
    FACE_BLUR_SUMS: 'INT64',
    FACE_OVERFLOW: 'INT32',
    FACE_EMBEDDINGS: 'FP32',
    FACE_EMBEDDED: 'BOOL',
    OCR_QUADS: 'FP32',
    OCR_TEXTS: 'BYTES',
    OCR_SCORES: 'FP32',
    OCR_CHAR_MIN: 'FP32',
    OCR_DET_SCORES: 'FP32',
    OCR_OVERFLOW: 'INT32',
    DERIVATIVE_JPEGS: 'BYTES',
}

#: ``status``: ``ok`` or one of :data:`src.domain.decode_route.PREP_STATUSES`.
STATUS_OK = 'ok'

#: ``decode_route``: the GPU decoder produced the pixels, or nothing was decoded. The vocabulary
#: grows when the engine also decodes on the CPU (the images the GPU route does not take).
DECODED_GPU = GPU
DECODED_NONE = 'none'

EngineBranch = Literal['detect', 'embed_frame', 'embed_crops', 'faces', 'face_embed', 'ocr']
ENGINE_BRANCHES: tuple[EngineBranch, ...] = cast(
    'tuple[EngineBranch, ...]',
    tuple(b for b in BRANCHES if b not in ('near_duplicates', 'region')),
)
#: Branches this engine runs; the others in the plan vocabulary answer ``unsupported_branch``.
BUILT_BRANCHES: frozenset[str] = frozenset(
    {'detect', 'embed_frame', 'embed_crops', 'faces', 'face_embed', 'derivatives', 'ocr'}
)
DERIVATIVES = 'derivatives'

BRANCH_OUTPUTS: Mapping[str, tuple[str, ...]] = {
    'detect': (DET_BOXES, DET_SCORES, DET_CLASSES, LETTERBOX, BLUR_SUMS),
    'embed_frame': (WHOLE_EMBEDDING,),
    'embed_crops': (CROP_EMBEDDINGS, CROP_EMBEDDED),
    # boxes [n, 4] and landmarks [n, 10] (five x, y) in source pixels of the oriented frame, best
    # score first; blur sums [n, 3] as ``blur_sums``; overflow [1]: faces past ``max_faces``
    'faces': (FACE_BOXES, FACE_SCORES, FACE_LANDMARKS, FACE_BLUR_SUMS, FACE_OVERFLOW),
    # vectors [m, dim] of the faces marked in ``face_embedded`` [n] (a degenerate landmark set is not)
    'face_embed': (FACE_EMBEDDINGS, FACE_EMBEDDED),
    # One row per text line, in reading order: ``ocr_quads`` [n, 8] (x, y of the top-left,
    # top-right, bottom-right, bottom-left corner, pixels of the oriented frame), ``ocr_texts`` [n],
    # ``ocr_scores`` [n] (mean probability of the decoded characters), ``ocr_char_min`` [n] (the
    # lowest), ``ocr_det_scores`` [n] (the detector's mean map value in the box); ``ocr_overflow``
    # [1]: lines past ``max_lines`` that were detected and dropped (the lowest detector scores).
    'ocr': (OCR_QUADS, OCR_TEXTS, OCR_SCORES, OCR_CHAR_MIN, OCR_DET_SCORES, OCR_OVERFLOW),
}
CROPS_OUTPUTS: tuple[str, ...] = (CROP_JPEGS,)

CropsMode = Literal['inline', 'none']
#: Shorter side to ``size``, centre crop, antialiased bilinear (Pillow ``BILINEAR``):
#: ``src.utils.pe_preprocess.whole_frame_rgb_from_image``.
FRAME_RECIPES = ('whole_frame',)
#: The crop rectangle, shorter side to ``size``, centre crop, plain bilinear:
#: ``src.utils.pe_preprocess.resize_crop_rgb``.
CROP_RECIPES = ('resize_crop',)

BranchState = Literal['ok', 'error', 'not_run']
UNSUPPORTED_BRANCH = 'unsupported_branch'
MODEL_ERROR = 'model_error'
DEPENDENCY_FAILED = 'dependency_failed'
IMAGE_FAILED = 'image_failed'
PlanFault = Literal[
    'bad_json',
    'bad_version',
    'unknown_field',
    'missing_field',
    'bad_field',
    'empty_plan',
    'unknown_branch',
    'duplicate_branch',
    'missing_dependency',
    'condition_on_unconditional',
    'boxes_with_detect',
    'unused_boxes',
]

_TOP_FIELDS = frozenset(
    {
        'version',
        'branches',
        'detector',
        'embed',
        'embed_classes',
        'if_detected',
        'faces',
        'ocr',
        'crops',
        DERIVATIVES,
    }
)


RENDITION_FORMATS = ('jpeg',)
RENDITION_SIDE_RANGE = (16, 4096)
DEFAULT_RENDITION_QUALITY = 85


@dataclass(frozen=True)
class RenditionSpec:
    """One stored rendition: the long side fits ``side`` (Pillow ``thumbnail``, never enlarged)."""

    name: str
    side: int
    quality: int = DEFAULT_RENDITION_QUALITY
    format: str = RENDITION_FORMATS[0]

    def to_json(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'side': self.side,
            'quality': self.quality,
            'format': self.format,
        }


class EnginePlanError(ValueError):
    """A plan the engine refuses as a whole (gRPC ``INVALID_ARGUMENT``)."""

    def __init__(self, fault: PlanFault, detail: str = '') -> None:
        self.fault = fault
        self.detail = detail
        super().__init__(f'{fault}: {detail}' if detail else fault)


@dataclass(frozen=True)
class DetectorSpec:
    """A fused end-to-end detector (``output0`` rows ``x1, y1, x2, y2, score, class``)."""

    model: str
    input_size: int
    letterbox_fill: int


@dataclass(frozen=True)
class EmbedSpec:
    model: str
    size: int
    batch: int
    frame_recipe: str = FRAME_RECIPES[0]
    crop_recipe: str = CROP_RECIPES[0]


@dataclass(frozen=True)
class FaceSpec:
    """The ``faces`` and ``face_embed`` branches: a SCRFD build (uint8 input ``images``, the nine
    stride outputs) at ``input_size``, an ArcFace build (uint8 input ``images``) called in chunks
    of ``batch`` aligned faces. ``coarse`` adds a pass of the whole frame in that square (0: none);
    ``tile_side`` adds tiles of the frame scaled to that longer side with ``overlap`` input pixels
    shared (0: none). ``score_min`` and ``nms`` are the detector thresholds; ``max_faces`` caps the
    faces kept per photo (the rest are counted in ``face_overflow``)."""

    detector: str
    input_size: int
    recogniser: str
    batch: int
    score_min: float
    nms: float
    max_faces: int
    coarse: int = 0
    tile_side: int = 0
    overlap: int = 0


@dataclass(frozen=True)
class OcrSpec:
    """The ``ocr`` branch: a detector build (uint8 BGR ``x``, normalisation in the graph, map
    ``fetch_name_0``) and a recogniser build (uint8 BGR ``x``, arg-max in the graph, ``indices`` and
    ``scores``), by model name.

    The detector reads the frame resized as PaddleX ``DetResizeForTest`` does: ``det_limit`` ``min``
    raises the shorter side to ``det_side`` (never shrinks), ``max`` caps the longer side at
    ``det_side`` (never enlarges); both end at ``det_max_side`` and a multiple of 32.
    ``det_antialias`` filters the shrink (the reference's ``cv2.resize`` does not). ``thresh``,
    ``box_thresh`` and ``unclip_ratio`` are the DB post-process's. Lines are recognised in batches of
    ``batch`` by PaddleOCR's recipe, each line sampled ``taps`` x ``taps`` times per recogniser pixel
    at most (1 reads the pixel centre, as ``cv2.resize`` does); at most ``max_lines`` are kept.
    """

    detector: str
    recogniser: str
    batch: int
    max_lines: int
    det_side: int
    det_limit: str
    det_max_side: int
    det_antialias: bool
    thresh: float
    box_thresh: float
    unclip_ratio: float
    taps: int


@dataclass(frozen=True)
class EnginePlan:
    branches: tuple[EngineBranch, ...]
    detector: DetectorSpec | None = None
    embed: EmbedSpec | None = None
    #: Detector class ids that get a crop vector; ``None`` means every detection.
    embed_classes: frozenset[int] | None = None
    crops: CropsMode = 'none'
    if_detected: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    faces: FaceSpec | None = None
    ocr: OcrSpec | None = None
    derivatives: tuple[RenditionSpec, ...] = ()

    def runs(self, branch: str) -> bool:
        return branch in self.branches

    def reported(self) -> tuple[str, ...]:
        """The names ``branch_status`` carries for this plan, one entry each."""
        return (*self.branches, DERIVATIVES) if self.derivatives else self.branches

    def unsupported(self) -> tuple[str, ...]:
        """Requested names this engine does not build, each answered ``unsupported_branch``."""
        return tuple(b for b in self.reported() if b not in BUILT_BRANCHES)

    def to_json(self) -> str:
        body: dict[str, Any] = {
            'version': PROTOCOL_VERSION,
            'branches': list(self.branches),
            'crops': self.crops,
        }
        if self.detector is not None:
            d = self.detector
            body['detector'] = {
                'model': d.model,
                'input_size': d.input_size,
                'letterbox_fill': d.letterbox_fill,
            }
        if self.embed is not None:
            e = self.embed
            body['embed'] = {
                'model': e.model,
                'size': e.size,
                'batch': e.batch,
                'frame_recipe': e.frame_recipe,
                'crop_recipe': e.crop_recipe,
            }
        if self.embed_classes is not None:
            body['embed_classes'] = sorted(self.embed_classes)
        if self.if_detected:
            body['if_detected'] = {k: list(v) for k, v in self.if_detected.items()}
        if self.faces is not None:
            f = self.faces
            body['faces'] = {
                'detector': f.detector,
                'input_size': f.input_size,
                'recogniser': f.recogniser,
                'batch': f.batch,
                'score_min': f.score_min,
                'nms': f.nms,
                'max_faces': f.max_faces,
                'coarse': f.coarse,
                'tile_side': f.tile_side,
                'overlap': f.overlap,
            }
        if self.ocr is not None:
            o = self.ocr
            body['ocr'] = {
                'detector': o.detector,
                'recogniser': o.recogniser,
                'batch': o.batch,
                'max_lines': o.max_lines,
                'det_side': o.det_side,
                'det_limit': o.det_limit,
                'det_max_side': o.det_max_side,
                'det_antialias': o.det_antialias,
                'thresh': o.thresh,
                'box_thresh': o.box_thresh,
                'unclip_ratio': o.unclip_ratio,
                'taps': o.taps,
            }
        if self.derivatives:
            body[DERIVATIVES] = [r.to_json() for r in self.derivatives]
        return json.dumps(body, sort_keys=True)


def _obj(raw: Any, name: str, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise EnginePlanError('bad_field', name)
    extra = set(raw) - keys
    if extra:
        raise EnginePlanError('unknown_field', f'{name}.{sorted(extra)[0]}')
    return raw


def _int(raw: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or not lo <= raw <= hi:
        raise EnginePlanError('bad_field', name)
    return raw


def _float(raw: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not lo <= raw <= hi:
        raise EnginePlanError('bad_field', name)
    return float(raw)


def _str(raw: Any, name: str, choices: tuple[str, ...] | None = None) -> str:
    if not isinstance(raw, str) or not raw or (choices is not None and raw not in choices):
        raise EnginePlanError('bad_field', name)
    return raw


def _need(body: dict[str, Any], key: str, name: str) -> Any:
    if key not in body:
        raise EnginePlanError('missing_field', name)
    return body[key]


def _renditions(raw: list[Any]) -> tuple[RenditionSpec, ...]:
    specs: list[RenditionSpec] = []
    for i, entry in enumerate(raw):
        where = f'{DERIVATIVES}[{i}]'
        r = _obj(entry, where, frozenset({'name', 'side', 'quality', 'format'}))
        name = _str(_need(r, 'name', f'{where}.name'), f'{where}.name')
        if name in {s.name for s in specs}:
            raise EnginePlanError('bad_field', f'{where}.name')
        specs.append(
            RenditionSpec(
                name=name,
                side=_int(
                    _need(r, 'side', f'{where}.side'), f'{where}.side', *RENDITION_SIDE_RANGE
                ),
                quality=_int(
                    r.get('quality', DEFAULT_RENDITION_QUALITY), f'{where}.quality', 1, 100
                ),
                format=_str(
                    r.get('format', RENDITION_FORMATS[0]), f'{where}.format', RENDITION_FORMATS
                ),
            )
        )
    return tuple(specs)


def _detector(raw: Any) -> DetectorSpec:
    d = _obj(raw, 'detector', frozenset({'model', 'input_size', 'letterbox_fill'}))
    return DetectorSpec(
        model=_str(_need(d, 'model', 'detector.model'), 'detector.model'),
        input_size=_int(
            _need(d, 'input_size', 'detector.input_size'), 'detector.input_size', 32, 4096
        ),
        letterbox_fill=_int(
            _need(d, 'letterbox_fill', 'detector.letterbox_fill'), 'detector.letterbox_fill', 0, 255
        ),
    )


def _embed(raw: Any) -> EmbedSpec:
    e = _obj(raw, 'embed', frozenset({'model', 'size', 'batch', 'frame_recipe', 'crop_recipe'}))
    return EmbedSpec(
        model=_str(_need(e, 'model', 'embed.model'), 'embed.model'),
        size=_int(_need(e, 'size', 'embed.size'), 'embed.size', 16, 4096),
        batch=_int(_need(e, 'batch', 'embed.batch'), 'embed.batch', 1, 1024),
        frame_recipe=_str(
            e.get('frame_recipe', FRAME_RECIPES[0]), 'embed.frame_recipe', FRAME_RECIPES
        ),
        crop_recipe=_str(e.get('crop_recipe', CROP_RECIPES[0]), 'embed.crop_recipe', CROP_RECIPES),
    )


_FACE_FIELDS = frozenset(
    {'detector', 'input_size', 'recogniser', 'batch', 'score_min', 'nms', 'max_faces', 'coarse',
     'tile_side', 'overlap'}
)  # fmt: skip


def _faces(raw: Any) -> FaceSpec:
    f = _obj(raw, 'faces', _FACE_FIELDS)
    net = _int(_need(f, 'input_size', 'faces.input_size'), 'faces.input_size', 32, 4096)
    if net % 32:
        raise EnginePlanError('bad_field', 'faces.input_size')
    coarse = _int(f.get('coarse', 0), 'faces.coarse', 0, net)
    if coarse and coarse < 32:
        raise EnginePlanError('bad_field', 'faces.coarse')
    return FaceSpec(
        detector=_str(_need(f, 'detector', 'faces.detector'), 'faces.detector'),
        input_size=net,
        recogniser=_str(_need(f, 'recogniser', 'faces.recogniser'), 'faces.recogniser'),
        batch=_int(_need(f, 'batch', 'faces.batch'), 'faces.batch', 1, 1024),
        score_min=_float(_need(f, 'score_min', 'faces.score_min'), 'faces.score_min', 0.0, 1.0),
        nms=_float(_need(f, 'nms', 'faces.nms'), 'faces.nms', 0.0, 1.0),
        max_faces=_int(_need(f, 'max_faces', 'faces.max_faces'), 'faces.max_faces', 1, 100_000),
        coarse=coarse,
        tile_side=_int(f.get('tile_side', 0), 'faces.tile_side', 0, 65_535),
        overlap=_int(f.get('overlap', 0), 'faces.overlap', 0, net - 1),
    )


_OCR_FIELDS = frozenset(
    {'detector', 'recogniser', 'batch', 'max_lines', 'det_side', 'det_limit', 'det_max_side',
     'det_antialias', 'thresh', 'box_thresh', 'unclip_ratio', 'taps'}
)  # fmt: skip
DET_LIMITS = ('min', 'max')
# The limits of the two engines' optimisation profiles (export_paddleocr_rec.py, export_paddleocr_det.py;
# a test pins them): a plan the wire accepts is a call the engines accept.
OCR_MAX_BATCH = 16
OCR_MAX_DET_SIDE = 1920


def _ocr(raw: Any) -> OcrSpec:
    o = _obj(raw, 'ocr', _OCR_FIELDS)
    antialias = o.get('det_antialias', False)
    if not isinstance(antialias, bool):
        raise EnginePlanError('bad_field', 'ocr.det_antialias')
    return OcrSpec(
        detector=_str(_need(o, 'detector', 'ocr.detector'), 'ocr.detector'),
        recogniser=_str(_need(o, 'recogniser', 'ocr.recogniser'), 'ocr.recogniser'),
        batch=_int(_need(o, 'batch', 'ocr.batch'), 'ocr.batch', 1, OCR_MAX_BATCH),
        max_lines=_int(_need(o, 'max_lines', 'ocr.max_lines'), 'ocr.max_lines', 1, 10_000),
        det_side=_int(_need(o, 'det_side', 'ocr.det_side'), 'ocr.det_side', 32, 4096),
        det_limit=_str(o.get('det_limit', 'max'), 'ocr.det_limit', DET_LIMITS),
        det_max_side=_int(
            o.get('det_max_side', OCR_MAX_DET_SIDE), 'ocr.det_max_side', 32, OCR_MAX_DET_SIDE
        ),
        det_antialias=antialias,
        thresh=_float(o.get('thresh', 0.2), 'ocr.thresh', 0.0, 1.0),
        box_thresh=_float(o.get('box_thresh', 0.45), 'ocr.box_thresh', 0.0, 1.0),
        unclip_ratio=_float(o.get('unclip_ratio', 1.4), 'ocr.unclip_ratio', 0.0, 10.0),
        taps=_int(o.get('taps', 1), 'ocr.taps', 1, 8),
    )


def _class_ids(raw: Any, name: str) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise EnginePlanError('bad_field', name)
    return tuple(_int(v, name, 0, 1_000_000) for v in raw)


def _branches(raw: Any) -> tuple[EngineBranch, ...]:
    if not isinstance(raw, list):
        raise EnginePlanError('bad_field', 'branches')
    if not raw:
        raise EnginePlanError('empty_plan')
    for name in raw:
        if name not in ENGINE_BRANCHES:
            raise EnginePlanError('unknown_branch', str(name))
    if len(set(raw)) != len(raw):
        raise EnginePlanError('duplicate_branch', next(n for n in raw if raw.count(n) > 1))
    return tuple(b for b in ENGINE_BRANCHES if b in raw)


def _check_settings(plan: EnginePlan, branch: str, settings: object) -> None:
    """A branch's settings object comes with the branch and only with it."""
    if plan.runs(branch) and settings is None:
        raise EnginePlanError('missing_field', branch)
    if settings is not None and not plan.runs(branch):
        raise EnginePlanError('bad_field', branch)


def _check_dependencies(plan: EnginePlan, *, has_boxes: bool) -> None:
    runs = plan.runs
    if has_boxes and runs('detect'):
        raise EnginePlanError('boxes_with_detect')
    if has_boxes and not (runs('embed_crops') or plan.crops == 'inline'):
        raise EnginePlanError('unused_boxes')
    rects = runs('detect') or has_boxes
    if runs('embed_crops') and not rects:
        raise EnginePlanError('missing_dependency', 'detect')
    if plan.crops == 'inline' and not rects:
        raise EnginePlanError('missing_dependency', 'detect')
    if runs('face_embed') and not runs('faces'):
        raise EnginePlanError('missing_dependency', 'faces')
    _check_settings(plan, 'faces', plan.faces)
    _check_settings(plan, 'ocr', plan.ocr)
    if runs('detect') and plan.detector is None:
        raise EnginePlanError('missing_field', 'detector')
    if (runs('embed_frame') or runs('embed_crops')) and plan.embed is None:
        raise EnginePlanError('missing_field', 'embed')
    if plan.embed_classes is not None and (not runs('embed_crops') or has_boxes):
        raise EnginePlanError('bad_field', 'embed_classes')
    for name in plan.if_detected:
        if name not in CONDITIONAL or not runs(name):
            raise EnginePlanError('condition_on_unconditional', name)
        if not runs('detect'):
            raise EnginePlanError('missing_dependency', 'detect')


def parse_plan(text: str | bytes, *, has_boxes: bool) -> EnginePlan:
    """The plan in ``text`` exactly as written, or :class:`EnginePlanError`.

    ``has_boxes`` says whether the request carries the ``boxes`` input: stored boxes stand in for
    the detector and so exclude ``detect``.
    """
    try:
        body = json.loads(text)
    except (ValueError, UnicodeDecodeError) as exc:
        raise EnginePlanError('bad_json', str(exc)) from exc
    body = _obj(body, 'plan', _TOP_FIELDS)
    if body.get('version') != PROTOCOL_VERSION:
        raise EnginePlanError('bad_version', repr(body.get('version')))
    branches = _branches(_need(body, 'branches', 'branches'))
    crops = _str(body.get('crops', 'none'), 'crops', ('inline', 'none'))
    raw_classes = body.get('embed_classes')
    raw_cond = body.get('if_detected', {})
    if not isinstance(raw_cond, dict):
        raise EnginePlanError('bad_field', 'if_detected')
    derivatives = body.get(DERIVATIVES, [])
    if not isinstance(derivatives, list):
        raise EnginePlanError('bad_field', DERIVATIVES)
    plan = EnginePlan(
        branches=branches,
        detector=_detector(body['detector']) if 'detector' in body else None,
        embed=_embed(body['embed']) if 'embed' in body else None,
        embed_classes=(
            None if raw_classes is None else frozenset(_class_ids(raw_classes, 'embed_classes'))
        ),
        crops=cast('CropsMode', crops),
        if_detected={str(k): _class_ids(v, f'if_detected.{k}') for k, v in raw_cond.items()},
        faces=_faces(body['faces']) if 'faces' in body else None,
        ocr=_ocr(body['ocr']) if 'ocr' in body else None,
        derivatives=_renditions(derivatives),
    )
    _check_dependencies(plan, has_boxes=has_boxes)
    return plan


def encode_branch_status(states: Mapping[str, tuple[BranchState, str]]) -> str:
    """``{name: {"status": ..., "reason": ...}}``; ``reason`` only for ``error``."""
    out: dict[str, dict[str, str]] = {}
    for name, (state, reason) in states.items():
        out[name] = {'status': state, 'reason': reason} if state == 'error' else {'status': state}
    return json.dumps(out, sort_keys=True)


def decode_branch_status(text: str | bytes) -> dict[str, tuple[BranchState, str]]:
    """The inverse of :func:`encode_branch_status`; raises ``ValueError`` on anything else."""
    body = json.loads(text)
    if not isinstance(body, dict):
        raise ValueError('branch_status is not an object')
    out: dict[str, tuple[BranchState, str]] = {}
    for name, entry in body.items():
        state = entry.get('status') if isinstance(entry, dict) else None
        if state not in ('ok', 'error', 'not_run'):
            raise ValueError(f'branch_status {name!r}: bad status {state!r}')
        out[name] = (cast('BranchState', state), str(entry.get('reason', '')))
    return out


def expected_outputs(plan: EnginePlan) -> tuple[str, ...]:
    """The branch tensors an ``ok`` answer to ``plan`` carries, besides the four always present."""
    names: list[str] = []
    for branch in plan.branches:
        names.extend(BRANCH_OUTPUTS.get(branch, ()))
    if plan.crops == 'inline':
        names.extend(CROPS_OUTPUTS)
    if plan.derivatives:
        names.append(DERIVATIVE_JPEGS)
    return tuple(names)


__all__ = [
    'BUILT_BRANCHES',
    'ENGINE_BRANCHES',
    'MODEL_NAME',
    'PROTOCOL_VERSION',
    'Branch',
    'DetectorSpec',
    'EmbedSpec',
    'EnginePlan',
    'EnginePlanError',
    'FaceSpec',
    'OcrSpec',
    'RenditionSpec',
    'decode_branch_status',
    'encode_branch_status',
    'expected_outputs',
    'parse_plan',
]
