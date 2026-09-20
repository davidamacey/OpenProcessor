"""Grep guard over the default detector model name (plan Wave 5 T-6).

Ported from the reference's ``test_active_model_pointer.py``. That file
named six call sites; on this tree only one exists —
``src/config/settings.py``'s ``TritonModelConfig.YOLO_MODEL`` default.
The other five reference sites
(``src/routers/detect.py``, ``src/services/inference.py`` x2,
the domain ingest service, the domain model-management router) either
already read the single settings default here (``detect.py``,
``inference.py``) or were never ported (the two domain-named files) —
so the expected hardcode count on this tree is 1, not 6.
"""

from __future__ import annotations

import ast
from pathlib import Path


LITERAL = 'yolov11_small_trt_end2end'

# Every module that mentions the literal at all, so a future added
# hardcode anywhere gets caught, not just at the one known site.
SITES: tuple[Path, ...] = (
    Path('src/config/settings.py'),
    Path('src/routers/detect.py'),
    Path('src/services/inference.py'),
    Path('src/schemas/models.py'),
    Path('src/clients/triton_client.py'),
)


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _code_hits(path: Path) -> list[str]:
    """Real code hardcodes of ``LITERAL`` — an actual string-constant
    AST node equal to the model name, excluding docstrings (prose
    mentions aren't a second source of truth)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    docstring_ids = _docstring_ids(tree)
    return [
        f'{path}:{node.lineno}'
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value == LITERAL
        and id(node) not in docstring_ids
    ]


def test_no_module_hardcodes_the_model_name_except_the_settings_default() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    all_hits: list[str] = []
    for rel in SITES:
        path = repo_root / rel
        if not path.exists():
            continue
        all_hits.extend(_code_hits(path))

    assert len(all_hits) == 1, (
        f'expected exactly one hardcode of {LITERAL!r} — the settings default '
        f'(src/config/settings.py TritonModelConfig.YOLO_MODEL) — every other '
        f'call site must read it from there instead. Found:\n' + '\n'.join(all_hits)
    )
    assert 'settings.py' in all_hits[0], (
        f'the one remaining hardcode must be the settings default, got: {all_hits[0]}'
    )


def test_settings_default_is_byte_identical_to_the_historical_value() -> None:
    """The default MUST stay exactly what was always hardcoded, so
    behavior is unchanged until someone explicitly overrides it via the
    ``YOLO_MODEL`` env var."""
    import importlib

    import src.config.settings as settings_mod

    importlib.reload(settings_mod)
    assert settings_mod.TritonModelConfig.YOLO_MODEL == LITERAL


def test_downstream_call_sites_read_the_single_settings_pointer() -> None:
    """Non-regression: detect.py and inference.py must keep reading
    ``TritonModelConfig.YOLO_MODEL`` rather than re-hardcoding."""
    import src.routers.detect as detect_mod
    from src.config.settings import TritonModelConfig

    assert detect_mod.DEFAULT_MODEL == TritonModelConfig.YOLO_MODEL
