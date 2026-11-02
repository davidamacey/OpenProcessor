"""A 500/503 response body never carries exception text.

The cause is logged server-side (``src/routers/_errors.py``); the caller gets a fixed message.
This scans every router for an ``HTTPException`` with a 5xx status whose ``detail`` is built from
a value (f-string interpolation or ``str(...)``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROUTERS = Path(__file__).resolve().parents[1] / 'src' / 'routers'

EXC_NAMES = {'e', 'ex', 'exc', 'err', 'error', 'exception'}
# The detail is the fixed, curated text of ExportWorkerUnavailableError (it names the worker
# service the operator must start), not a caught arbitrary exception.
ALLOWED = {('models_export.py', 'str')}


def _is_http_exception(call: ast.Call) -> bool:
    func = call.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', '')
    return name == 'HTTPException'


def _is_5xx(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and 500 <= node.value <= 599
    return isinstance(node, ast.Attribute) and node.attr.startswith('HTTP_5')


def _carries_error(expr: ast.expr) -> bool:
    """The interpolated value is a caught exception or an upstream ``error`` field."""
    return any(
        (isinstance(n, ast.Name) and n.id in EXC_NAMES)
        or (isinstance(n, ast.Constant) and n.value == 'error')
        for n in ast.walk(expr)
    )


def _offences(path: Path) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if not isinstance(node, ast.Call) or not _is_http_exception(node):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        if 'status_code' not in kw or not _is_5xx(kw['status_code']) or 'detail' not in kw:
            continue
        detail = kw['detail']
        if isinstance(detail, ast.JoinedStr) and any(
            isinstance(v, ast.FormattedValue) and _carries_error(v.value) for v in detail.values
        ):
            kind = 'fstring'
        elif isinstance(detail, ast.Call) and getattr(detail.func, 'id', '') == 'str':
            kind = 'str'
        else:
            continue
        if (path.name, kind) not in ALLOWED:
            found.append(f'{path.relative_to(ROUTERS)}:{node.lineno}')
    return found


@pytest.mark.parametrize(
    'path', sorted(ROUTERS.rglob('*.py')), ids=lambda p: str(p.relative_to(ROUTERS))
)
def test_no_exception_text_in_5xx_detail(path: Path) -> None:
    assert _offences(path) == []


def test_server_error_hides_the_cause_and_keeps_the_status() -> None:
    from src.routers._errors import server_error

    err = server_error(503, 'opensearch unavailable', RuntimeError('10.0.0.5:9200 refused'))
    assert err.status_code == 503
    assert err.detail == 'opensearch unavailable'
