"""The plan texts of ``contracts/json/frame_engine_plan_cases.json`` (plan 17, section 3, "Contract").

Each case is the exact text a client may send as the ``plan`` input and whether the request carries
``boxes``. The outcome is never written here: the generator runs ``frame_engine_wire.parse_plan`` on
every case and records its answer (the plan it parsed to, or the fault), so the Python parser is the
definition and the native parser is held to it case by case.
"""

from __future__ import annotations

import json
from typing import Any


DETECTOR = {'model': 'det', 'input_size': 640, 'letterbox_fill': 114}
EMBED = {'model': 'pe', 'size': 336}
CROP_EMBED = {'model': 'pe_crop', 'size': 224, 'batch': 32}
FACES = {
    'detector': 'scrfd',
    'input_size': 640,
    'recogniser': 'arcface',
    'batch': 32,
    'score_min': 0.5,
    'nms': 0.4,
    'max_faces': 128,
}
OCR = {
    'detector': 'ocr_det',
    'recogniser': 'ocr_rec',
    'batch': 16,
    'max_lines': 128,
    'det_side': 960,
}

#: The ingest plan as the API builds it (``ingest_engine_plan``), written by hand with sorted keys.
INGEST_TEXT = (
    '{"branches": ["detect", "embed_frame", "embed_crops"], '
    '"crop_embed": {"batch": 32, "model": "pe_crop", "size": 224}, "crops": "inline", '
    '"detector": {"input_size": 640, "letterbox_fill": 114, "model": "det"}, '
    '"embed": {"model": "pe", "size": 336}, "version": 1}'
)


def _plan(**over: Any) -> str:
    """The ingest plan with ``over`` applied; a value of ``None`` removes the key. Each branch's
    settings object comes with its branch unless the case sets it."""
    body: dict[str, Any] = {
        'version': 1,
        'branches': ['detect', 'embed_frame', 'embed_crops'],
        'detector': DETECTOR,
        'crops': 'inline',
    }
    body.update(over)
    branches = body['branches'] or []
    if 'embed' not in over and 'embed_frame' in branches:
        body['embed'] = EMBED
    if 'crop_embed' not in over and 'embed_crops' in branches:
        body['crop_embed'] = CROP_EMBED
    return json.dumps({k: v for k, v in body.items() if v is not None})


def _edit(old: str, new: str) -> str:
    if old not in INGEST_TEXT:
        raise ValueError(f'{old!r} is not in the ingest text')
    return INGEST_TEXT.replace(old, new, 1)


def _valid() -> list[tuple[str, str, bool]]:
    lean = {'branches': ['detect', 'embed_frame'], 'crops': 'none'}
    boxes_only = {'branches': ['embed_frame', 'embed_crops'], 'detector': None, 'crops': 'none'}
    faces_only = {'branches': ['faces', 'face_embed'], 'detector': None, 'embed': None,
                  'crops': 'none', 'faces': FACES}  # fmt: skip
    return [
        ('ingest_plan_as_the_api_sends_it', INGEST_TEXT, False),
        ('ingest_plan_built_here', _plan(), False),
        ('frame_plan', _plan(**lean), False),
        (
            'frame_plan_with_class_filter',
            _plan(branches=['detect', 'embed_crops'], embed=None, embed_classes=[2, 0, 5]),
            False,
        ),
        ('crops_inline_without_embedding', _plan(branches=['detect'], crops='inline'), False),
        ('stored_boxes_embedded', _plan(**boxes_only), True),
        ('stored_boxes_crops_inline', _plan(branches=['embed_frame'], detector=None), True),
        (
            'defaults_are_filled_in',
            _plan(
                embed={'model': 'pe', 'size': 336},
                crop_embed={'model': 'c', 'size': 224, 'batch': 1},
            ),
            False,
        ),
        (
            'crop_recipe_and_fill_given',
            _plan(crop_embed={**CROP_EMBED, 'crop_recipe': 'letterbox', 'fill': 0}),
            False,
        ),
        ('frame_recipe_given', _plan(embed={**EMBED, 'frame_recipe': 'whole_frame'}), False),
        (
            'integers_where_floats_are_allowed',
            _plan(
                branches=['faces'],
                detector=None,
                embed=None,
                crops='none',
                faces={**FACES, 'score_min': 1, 'nms': 0},
            ),
            False,
        ),
        (
            'branch_order_is_the_vocabulary_order',
            _plan(branches=['embed_crops', 'embed_frame', 'detect']),
            False,
        ),
        ('faces_with_embedding', _plan(**faces_only), False),
        (
            'faces_lean_defaults',
            _plan(branches=['faces'], detector=None, embed=None, crops='none', faces=FACES),
            False,
        ),
        (
            'faces_full',
            _plan(
                branches=['faces', 'face_embed'],
                detector=None,
                embed=None,
                crops='none',
                faces={**FACES, 'coarse': 256, 'tile_side': 1920, 'overlap': 160},
            ),
            False,
        ),
        (
            'ocr_lean',
            _plan(branches=['ocr'], detector=None, embed=None, crops='none', ocr=OCR),
            False,
        ),
        (
            'ocr_full',
            _plan(
                branches=['ocr'],
                detector=None,
                embed=None,
                crops='none',
                ocr={
                    **OCR,
                    'det_limit': 'min',
                    'det_max_side': 1920,
                    'det_antialias': True,
                    'thresh': 0.3,
                    'box_thresh': 0.5,
                    'unclip_ratio': 1.6,
                    'taps': 2,
                },
            ),
            False,
        ),
        (
            'conditional_branch_after_detect',
            _plan(
                branches=['detect', 'ocr'],
                embed=None,
                crops='none',
                ocr=OCR,
                if_detected={'ocr': [2, 7]},
            ),
            False,
        ),
        (
            'renditions',
            _plan(
                branches=['detect'],
                crops='none',
                derivatives=[
                    {'name': 'thumb', 'side': 256},
                    {'name': 'preview', 'side': 1024, 'quality': 70, 'format': 'jpeg'},
                ],
            ),
            False,
        ),
        (
            'renditions_alone_with_faces',
            _plan(
                branches=['faces'],
                detector=None,
                embed=None,
                crops='none',
                faces=FACES,
                derivatives=[{'name': 't', 'side': 16}],
            ),
            False,
        ),
        (
            'bytes_input_is_text_with_unicode_names',
            _plan(detector={**DETECTOR, 'model': 'détecteur'}),
            False,
        ),
    ]


def _faults() -> list[tuple[str, str, bool]]:
    d = DETECTOR
    return [
        ('not_json', '{', False),
        ('empty_text', '', False),
        ('not_an_object', '[]', False),
        ('null_plan', 'null', False),
        ('version_two', _plan(version=2), False),
        ('version_missing', _plan(version=None), False),
        ('version_string', _plan(version='1'), False),
        ('version_null', '{"version": null}', False),
        ('unknown_top_field', _plan(extra=1), False),
        ('branches_missing', _plan(branches=None), False),
        ('branches_not_a_list', _plan(branches='detect'), False),
        ('empty_plan', _plan(branches=[]), False),
        ('unknown_branch', _plan(branches=['detect', 'region']), False),
        ('unknown_branch_non_string', _plan(branches=['detect', 3]), False),
        ('duplicate_branch', _plan(branches=['detect', 'detect']), False),
        ('crops_without_detect', _plan(branches=['embed_crops'], crops='none'), False),
        ('crops_inline_without_detect', _plan(branches=['embed_frame'], crops='inline'), False),
        ('face_embed_without_faces', _plan(branches=['face_embed']), False),
        ('detector_missing', _plan(detector=None), False),
        ('embed_missing', _plan(embed=None), False),
        ('crop_embed_missing', _plan(crop_embed=None), False),
        (
            'faces_missing',
            _plan(branches=['faces'], detector=None, embed=None, crops='none'),
            False,
        ),
        ('ocr_missing', _plan(branches=['ocr'], detector=None, embed=None, crops='none'), False),
        ('detector_size_zero', _plan(detector={**d, 'input_size': 0}), False),
        ('detector_fill_over_255', _plan(detector={**d, 'letterbox_fill': 256}), False),
        ('detector_unknown_field', _plan(detector={**d, 'nms': 1}), False),
        ('detector_not_an_object', _plan(detector='det'), False),
        ('embed_unknown_field', _plan(embed={**EMBED, 'batch': 32}), False),
        (
            'embed_crop_recipe_not_allowed',
            _plan(embed={**EMBED, 'crop_recipe': 'letterbox'}),
            False,
        ),
        ('embed_frame_recipe_unknown', _plan(embed={**EMBED, 'frame_recipe': 'centre'}), False),
        ('crop_batch_is_a_bool', _plan(crop_embed={**CROP_EMBED, 'batch': True}), False),
        ('crop_batch_is_a_float', _plan(crop_embed={**CROP_EMBED, 'batch': 32.0}), False),
        ('crop_size_below_range', _plan(crop_embed={**CROP_EMBED, 'size': 8}), False),
        ('crop_fill_over_255', _plan(crop_embed={**CROP_EMBED, 'fill': 256}), False),
        (
            'crop_recipe_resize_crop',
            _plan(crop_embed={**CROP_EMBED, 'crop_recipe': 'resize_crop'}),
            False,
        ),
        (
            'crop_recipe_centre_crop',
            _plan(crop_embed={**CROP_EMBED, 'crop_recipe': 'centre_crop'}),
            False,
        ),
        ('crop_embed_unknown_field', _plan(crop_embed={**CROP_EMBED, 'extra': 1}), False),
        (
            'crop_embed_without_its_branch',
            _plan(branches=['detect', 'embed_frame'], crop_embed=CROP_EMBED, crops='none'),
            False,
        ),
        ('embed_without_its_branch', _plan(branches=['detect', 'embed_crops'], embed=EMBED), False),
        ('crops_value_unknown', _plan(crops='thumb'), False),
        ('crops_value_empty', _plan(crops=''), False),
        ('embed_classes_not_a_list', _plan(embed_classes=['car']), False),
        ('embed_classes_not_integers', _plan(embed_classes=[1.5]), False),
        ('embed_classes_negative', _plan(embed_classes=[-1]), False),
        (
            'embed_classes_without_crops',
            _plan(branches=['detect', 'embed_frame'], embed_classes=[2]),
            False,
        ),
        (
            'embed_classes_with_stored_boxes',
            _plan(
                branches=['embed_frame', 'embed_crops'],
                detector=None,
                crops='none',
                embed_classes=[1],
            ),
            True,
        ),
        ('if_detected_not_an_object', _plan(if_detected=[1]), False),
        ('condition_on_embed_crops', _plan(if_detected={'embed_crops': [2]}), False),
        ('condition_on_ocr_not_in_plan', _plan(if_detected={'ocr': [2]}), False),
        (
            'condition_without_detect',
            _plan(
                branches=['ocr'],
                detector=None,
                embed=None,
                crops='none',
                ocr=OCR,
                if_detected={'ocr': [2]},
            ),
            False,
        ),
        ('derivatives_not_a_list', _plan(derivatives={'thumb': 256}), False),
        ('rendition_side_out_of_range', _plan(derivatives=[{'name': 't', 'side': 8}]), False),
        (
            'rendition_format_unknown',
            _plan(derivatives=[{'name': 't', 'side': 64, 'format': 'png'}]),
            False,
        ),
        (
            'faces_input_not_a_multiple_of_32',
            _plan(
                branches=['faces'],
                detector=None,
                embed=None,
                crops='none',
                faces={**FACES, 'input_size': 650},
            ),
            False,
        ),
        (
            'faces_score_out_of_range',
            _plan(
                branches=['faces'],
                detector=None,
                embed=None,
                crops='none',
                faces={**FACES, 'score_min': 1.5},
            ),
            False,
        ),
        (
            'faces_unknown_field',
            _plan(
                branches=['faces'], detector=None, embed=None, crops='none', faces={**FACES, 'x': 1}
            ),
            False,
        ),
        (
            'ocr_batch_over_profile',
            _plan(
                branches=['ocr'], detector=None, embed=None, crops='none', ocr={**OCR, 'batch': 17}
            ),
            False,
        ),
        (
            'ocr_limit_unknown',
            _plan(
                branches=['ocr'],
                detector=None,
                embed=None,
                crops='none',
                ocr={**OCR, 'det_limit': 'mid'},
            ),
            False,
        ),
        (
            'ocr_antialias_not_a_bool',
            _plan(
                branches=['ocr'],
                detector=None,
                embed=None,
                crops='none',
                ocr={**OCR, 'det_antialias': 1},
            ),
            False,
        ),
        ('boxes_with_detect', _plan(), True),
        ('boxes_unused', _plan(branches=['embed_frame'], detector=None, crops='none'), True),
        (
            'detect_missing_for_crops_with_no_boxes',
            _plan(branches=['embed_frame', 'embed_crops'], detector=None, crops='none'),
            False,
        ),
    ]


def _set(text: str, path: tuple[str | int, ...], value: Any) -> str:
    """``text`` with the JSON value at ``path`` replaced."""
    body = json.loads(text)
    node = body
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return json.dumps(body)


def _boundaries() -> list[tuple[str, str, bool]]:
    """Each numeric plan field at, just below and just above both ends of its range, so the native
    parser's ranges are held to the Python ones without a table of its own to trust: the generator
    writes only the inputs, the recorded answers are ``parse_plan``'s."""
    ingest = _plan()
    faces = _plan(branches=['faces'], detector=None, embed=None, crops='none', faces=FACES)
    ocr = _plan(branches=['ocr'], detector=None, embed=None, crops='none', ocr=OCR)
    renditions = _plan(derivatives=[{'name': 't', 'side': 64, 'quality': 85}])
    classes = _plan(embed_classes=[0])
    conditional = _plan(
        branches=['detect', 'ocr'], embed=None, crops='none', ocr=OCR, if_detected={'ocr': [2]}
    )
    ints: list[tuple[str, str, tuple[str | int, ...], int, int]] = [
        ('detector_input_size', ingest, ('detector', 'input_size'), 32, 4096),
        ('detector_fill', ingest, ('detector', 'letterbox_fill'), 0, 255),
        ('embed_size', ingest, ('embed', 'size'), 16, 4096),
        ('crop_size', ingest, ('crop_embed', 'size'), 16, 4096),
        ('crop_batch', ingest, ('crop_embed', 'batch'), 1, 1024),
        ('crop_fill', ingest, ('crop_embed', 'fill'), 0, 255),
        ('faces_input_size', faces, ('faces', 'input_size'), 32, 4096),
        ('faces_batch', faces, ('faces', 'batch'), 1, 1024),
        ('faces_max', faces, ('faces', 'max_faces'), 1, 100_000),
        ('faces_coarse', faces, ('faces', 'coarse'), 0, 640),
        ('faces_tile', faces, ('faces', 'tile_side'), 0, 65_535),
        ('faces_overlap', faces, ('faces', 'overlap'), 0, 639),
        ('ocr_batch', ocr, ('ocr', 'batch'), 1, 16),
        ('ocr_lines', ocr, ('ocr', 'max_lines'), 1, 10_000),
        ('ocr_side', ocr, ('ocr', 'det_side'), 32, 4096),
        ('ocr_max_side', ocr, ('ocr', 'det_max_side'), 32, 1920),
        ('ocr_taps', ocr, ('ocr', 'taps'), 1, 8),
        ('rendition_side', renditions, ('derivatives', 0, 'side'), 16, 4096),
        ('rendition_quality', renditions, ('derivatives', 0, 'quality'), 1, 100),
        ('embed_class_id', classes, ('embed_classes', 0), 0, 1_000_000),
        ('if_detected_id', conditional, ('if_detected', 'ocr', 0), 0, 1_000_000),
    ]
    floats: list[tuple[str, str, tuple[str | int, ...], float, float]] = [
        ('faces_score_min', faces, ('faces', 'score_min'), 0.0, 1.0),
        ('faces_nms', faces, ('faces', 'nms'), 0.0, 1.0),
        ('ocr_thresh', ocr, ('ocr', 'thresh'), 0.0, 1.0),
        ('ocr_box_thresh', ocr, ('ocr', 'box_thresh'), 0.0, 1.0),
        ('ocr_unclip', ocr, ('ocr', 'unclip_ratio'), 0.0, 10.0),
    ]
    cases: list[tuple[str, str, bool]] = []
    for name, base, path, lo, hi in ints:
        for label, value in (('lo', lo), ('below', lo - 1), ('hi', hi), ('above', hi + 1)):
            cases.append((f'range_{name}_{label}', _set(base, path, value), False))
    for fname, fbase, fpath, lo_f, hi_f in floats:
        for label, fvalue in (
            ('lo', lo_f),
            ('below', lo_f - 0.5),
            ('hi', hi_f),
            ('above', hi_f + 0.5),
        ):
            cases.append((f'range_{fname}_{label}', _set(fbase, fpath, fvalue), False))
    # an integer where a float is allowed, and the huge and the exponent spellings
    cases.append(('range_faces_score_min_integer', _set(faces, ('faces', 'score_min'), 1), False))
    cases.append(('range_big_integer', _set(ingest, ('detector', 'input_size'), 10**30), False))
    cases.append(('range_big_version', _set(ingest, ('version',), 10**30), False))
    cases.append(('range_float_version_text', _edit('"version": 1', '"version": 1.50'), False))
    return cases


def _quirks() -> list[tuple[str, str, bool]]:
    """The four ways Python's ``json`` and ``==`` accept what another parser would not (plan 17,
    review finding m4). The API never sends one."""
    cases: list[tuple[str, str, bool]] = [
        ('quirk_version_true', _edit('"version": 1', '"version": true'), False),
        ('quirk_version_one_point_zero', _edit('"version": 1', '"version": 1.0'), False),
        ('quirk_version_exponent', _edit('"version": 1', '"version": 1e0'), False),
        ('quirk_version_false', _edit('"version": 1', '"version": false'), False),
    ]
    for label, literal in (
        ('nan', 'NaN'),
        ('infinity', 'Infinity'),
        ('minus_infinity', '-Infinity'),
    ):
        cases.append(
            (f'quirk_{label}_version', _edit('"version": 1', f'"version": {literal}'), False)
        )
        cases.append((f'quirk_{label}_size', _edit('"size": 336', f'"size": {literal}'), False))
        cases.append(
            (
                f'quirk_{label}_fill',
                _edit('"letterbox_fill": 114', f'"letterbox_fill": {literal}'),
                False,
            )
        )
    cases += [
        ('quirk_duplicate_version', _edit('"version": 1', '"version": 1, "version": 1'), False),
        (
            'quirk_duplicate_crops_differing',
            _edit('"crops": "inline"', '"crops": "none", "crops": "inline"'),
            False,
        ),
        ('quirk_duplicate_nested', _edit('"size": 336', '"size": 336, "size": 224'), False),
        (
            'quirk_duplicate_in_detector',
            _edit('"model": "det"', '"model": "det", "model": "x"'),
            False,
        ),
        ('same_key_in_different_objects_is_fine', INGEST_TEXT, False),
    ]
    return cases


def plan_cases() -> list[tuple[str, str, bool]]:
    """``(id, plan text, has_boxes)`` for every case, in a fixed order; ids are unique."""
    cases = [*_valid(), *_faults(), *_boundaries(), *_quirks()]
    ids = [c[0] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate case id')
    return cases
