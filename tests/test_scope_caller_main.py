"""A module run with ``python -m`` is judged by its real name, not ``__main__``.

``python -m scripts.db.migrate`` sets ``__name__`` to ``'__main__'`` and keeps the import name in
``__spec__.name`` (Python docs, ``__main__`` and ``runpy``), so the scope allowlist must read the
spec or every ``-m`` entry point is refused.
"""

from __future__ import annotations

from importlib.machinery import ModuleSpec
from types import SimpleNamespace

from src.db.session import _caller_is_allowed


def _frame(globals_: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(f_globals=globals_, f_code=SimpleNamespace(co_filename='/app/x.py'))


def test_a_module_run_with_dash_m_is_allowed_by_its_spec_name() -> None:
    frame = _frame({'__name__': '__main__', '__spec__': ModuleSpec('scripts.db.migrate', None)})
    assert _caller_is_allowed(frame, ('scripts.db',))  # type: ignore[arg-type]


def test_a_dash_m_module_outside_the_allowlist_is_still_refused() -> None:
    frame = _frame({'__name__': '__main__', '__spec__': ModuleSpec('scripts.other.tool', None)})
    assert not _caller_is_allowed(frame, ('scripts.db',))  # type: ignore[arg-type]


def test_a_script_run_by_path_has_no_spec_and_is_refused() -> None:
    frame = _frame({'__name__': '__main__', '__spec__': None})
    assert not _caller_is_allowed(frame, ('scripts.db',))  # type: ignore[arg-type]
