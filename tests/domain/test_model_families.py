"""The detection-head registry: what export a head needs, in one stdlib-only module.

``src/domain/model_families.py`` is read by the API's upload validator and by the trainer image
(which carries ``src/domain`` and nothing of the web stack), so it imports the standard library
only.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.domain.model_families import export_kwargs, head_kind_of


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_a_head_with_the_one_to_one_branch_is_one2one() -> None:
    assert head_kind_of(SimpleNamespace(one2one_cv2=[], cv2=[])) == 'one2one'


@pytest.mark.parametrize(
    'head',
    [
        SimpleNamespace(cv2=[], cv3=[]),
        # Ultralytics 8.4 exposes a `one2one` property on heads that have no one-to-one branch
        # as well as on those that do, so only `one2one_cv2` tells the two apart.
        SimpleNamespace(one2one={'box_head': None}),
    ],
)
def test_any_other_head_needs_nms(head: object) -> None:
    assert head_kind_of(head) == 'nms'


def test_the_one_to_one_export_selects_the_nms_free_head() -> None:
    assert export_kwargs('one2one', 8) == {'nms': False, 'batch': 8}


def test_the_nms_export_embeds_class_agnostic_nms_at_the_served_thresholds() -> None:
    assert export_kwargs('nms', 64) == {
        'nms': True,
        'agnostic_nms': True,
        'iou': 0.7,
        'conf': 0.25,
        'max_det': 300,
        'batch': 64,
    }


def test_the_export_batch_is_the_serving_batch() -> None:
    assert export_kwargs('nms', 17)['batch'] == 17
    assert export_kwargs('one2one', 5)['batch'] == 5


def test_an_unknown_head_kind_is_refused() -> None:
    with pytest.raises(ValueError, match='onnx_objectness'):
        export_kwargs('onnx_objectness', 8)


def test_each_call_returns_a_fresh_dict() -> None:
    first = export_kwargs('nms', 8)
    first['conf'] = 0.9
    assert export_kwargs('nms', 8)['conf'] == 0.25


def test_the_module_imports_with_only_the_standard_library(tmp_path: Path) -> None:
    """The trainer image copies ``src/domain`` and has no other part of ``src``."""
    package = tmp_path / 'src' / 'domain'
    package.mkdir(parents=True)
    shutil.copy(REPO_ROOT / 'src' / 'domain' / '__init__.py', package / '__init__.py')
    shutil.copy(REPO_ROOT / 'src' / 'domain' / 'model_families.py', package / 'model_families.py')
    probe = (
        'import sys; sys.path.insert(0, sys.argv[1]); before = set(sys.modules);'
        'import src.domain.model_families as m;'
        "new = {n.split('.')[0] for n in set(sys.modules) - before};"
        "print(m.head_kind_of(object()), sorted(new - set(sys.stdlib_module_names) - {'src'}))"
    )
    result = subprocess.run(  # nosec B603 - the test interpreter, fixed script, tmp_path argument
        [sys.executable, '-I', '-c', probe, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == 'nms []'


def test_every_family_has_a_spec_with_a_known_head_kind_and_decoder() -> None:
    from src.domain.model_families import FAMILIES

    assert [spec.name for spec in FAMILIES] == [
        'yolov5u',
        'yolov8',
        'yolov9',
        'yolov10',
        'yolo11',
        'yolo12',
        'yolo26',
        'yolov5_original',
    ]
    for spec in FAMILIES:
        assert spec.head_kind in {'one2one', 'nms', 'onnx_objectness'}
        assert spec.decoder in {'fused', 'objectness'}
        assert spec.pt_supported == (spec.name != 'yolov5_original')


def test_the_one_to_one_families_are_exactly_yolov10_and_yolo26() -> None:
    from src.domain.model_families import FAMILIES

    assert {s.name for s in FAMILIES if s.head_kind == 'one2one'} == {'yolov10', 'yolo26'}
    assert {s.name for s in FAMILIES if s.head_kind == 'onnx_objectness'} == {'yolov5_original'}


def test_the_probe_route_accepts_the_architectures_it_always_accepted() -> None:
    from src.domain.model_families import PROBE_ARCHITECTURES, probe_decoder

    assert PROBE_ARCHITECTURES == ('yolo11', 'yolo26', 'yolov5_objectness')
    assert probe_decoder('yolo11') == probe_decoder('yolo26') == 'fused'
    assert probe_decoder('yolov5_objectness') == 'objectness'
    with pytest.raises(ValueError, match='yolov8'):
        probe_decoder('yolov8')


def test_no_other_module_defines_an_architecture_tuple() -> None:
    """The registry is the one place that names the families (``rg "('yolo11', 'yolo26'" src``)."""
    import ast

    offenders: list[str] = []
    for top in ('src', 'export', 'docker', 'scripts'):
        for path in sorted((REPO_ROOT / top).rglob('*.py')):
            if path == REPO_ROOT / 'src' / 'domain' / 'model_families.py':
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                    names = {
                        e.value
                        for e in node.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    }
                    if {'yolo11', 'yolo26'} <= names:
                        offenders.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno}')
    assert not offenders, offenders


def test_the_export_job_reads_its_head_kinds_from_the_registry() -> None:
    from src.domain.model_families import HEAD_KINDS

    assert HEAD_KINDS == ('one2one', 'nms')
