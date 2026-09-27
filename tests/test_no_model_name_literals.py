"""Guard: Triton model-name literals stay out of src/ and scripts/ except
where a single settings default (or the one probe-architecture registry)
lives (task #63 fix wave, model_name_audit_2026-09-25).

Two checks:

1. None of the audited Triton model-name string literals appear as a
   real AST string constant anywhere under ``src/`` or ``scripts/``
   *outside* ``src/config/`` (docstrings/comments excluded -- prose
   mentions aren't a second source of truth). Every call site must read
   the name from ``TritonModelConfig`` (or, for PE, the module constants
   that themselves resolve from ``TritonModelConfig``).
2. Exactly one probe-architecture tuple exists
   (``src.services.curation.probe_models.PROBE_ARCHITECTURES``); every
   other module that needs the list imports it rather than redeclaring
   its own copy.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Model-name literals audited in model_name_audit_2026-09-26.md. Their
# only legitimate home is a default value inside src/config/.
_LITERALS = (
    'arcface_w600k_r50',
    'mobileclip2_s2_image_encoder',
    'mobileclip2_s2_text_encoder',
    'ocr_pipeline',
    'scrfd_10g_bnkps',
    'pe_image_encoder',
    'pe_text_encoder',
)

# Roots to scan. src/config/ is exempt (that's where the settings
# defaults live); everything else under src/ and scripts/ is scanned.
_SCAN_ROOTS = ('src', 'scripts')
_EXEMPT_DIRS = ('src/config',)

_PROBE_ARCH_TUPLE = ('yolo11', 'yolo26', 'yolov5_objectness')


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _iter_py_files() -> list[Path]:
    files: list[Path] = []
    for root in _SCAN_ROOTS:
        for path in (REPO_ROOT / root).rglob('*.py'):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if any(rel.startswith(exempt) for exempt in _EXEMPT_DIRS):
                continue
            files.append(path)
    return files


def _literal_hits(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    docstring_ids = _docstring_ids(tree)
    return [
        f'{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.value!r}'
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in _LITERALS
        and id(node) not in docstring_ids
    ]


def test_no_model_name_literals_outside_settings() -> None:
    all_hits: list[str] = []
    for path in _iter_py_files():
        all_hits.extend(_literal_hits(path))

    assert not all_hits, (
        'Triton model-name literals must be read from '
        'src.config.settings.TritonModelConfig, not hardcoded. Found:\n' + '\n'.join(all_hits)
    )


def test_exactly_one_probe_architectures_tuple() -> None:
    """Every module scanned should reference the tuple's *values* through
    at most one literal tuple declaration (the registry itself); any
    other file redeclaring ``('yolo11', 'yolo26', 'yolov5_objectness')``
    (in that order) is a duplicate the audit asked to collapse."""
    declarations: list[str] = []
    for path in _iter_py_files():
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Tuple):
                values = [
                    elt.value
                    for elt in node.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
                if tuple(values) == _PROBE_ARCH_TUPLE:
                    declarations.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno}')

    assert len(declarations) == 1, (
        'expected exactly one PROBE_ARCHITECTURES declaration '
        '(src/services/curation/probe_models.py); every other consumer must '
        'import it. Found:\n' + '\n'.join(declarations)
    )
    assert 'probe_models.py' in declarations[0]


def test_probe_architectures_is_imported_by_its_consumers() -> None:
    from src.services.curation import probe_models

    assert probe_models.PROBE_ARCHITECTURES == _PROBE_ARCH_TUPLE

    import scripts.curation.run_probe as run_probe_mod

    assert run_probe_mod.ARCHITECTURES is probe_models.PROBE_ARCHITECTURES

    from src.services.curation import probe_predictions

    assert probe_predictions.PROBE_ARCHITECTURES is probe_models.PROBE_ARCHITECTURES

    from src.routers.curation import probe as probe_router_mod

    assert probe_router_mod.PROBE_ARCHITECTURES is probe_models.PROBE_ARCHITECTURES


def test_segmenter_client_source_name_has_no_default() -> None:
    """scripts/curation/worker/client.py:SegmenterClient -- source_name
    must be required (no ``source_name='sam3'`` default); callers pass
    the profile's ``segmenter_name`` explicitly."""
    import inspect

    from scripts.curation.worker.client import SegmenterClient

    sig = inspect.signature(SegmenterClient.__init__)
    param = sig.parameters['source_name']
    assert param.default is inspect.Parameter.empty


def test_class_names_has_no_stock_coco_special_case() -> None:
    """src/utils/class_names.py must not carry a special-cased model-name
    set that borrows another model's (COCO's) vocabulary."""
    import src.utils.class_names as class_names_mod

    assert not hasattr(class_names_mod, '_STOCK_COCO_MODEL_NAMES')
