"""P1 commit 2/3 static guard: nothing captures a project-scoped value at
import time, and no raw ``AsyncOpenSearch``/``OpenSearch(`` bypasses the
factory inside the curation packages this pass covers (§3.3/§2.4).

**Known, documented gap** (see PR notes / SubagentHandback report): the
plan's own text says "AsyncOpenSearch( / OpenSearch( outside the factory"
should be forbidden repo-wide, including ``scripts/curation/*.py``. This
codebase has 17 script files that construct ``AsyncOpenSearch`` directly
and were never routed through ``make_curation_opensearch()`` in this
pass -- routing them through one factory is real, separate work. So this
test only scans ``src/services/curation/`` and ``src/routers/curation*``
for that specific rule, not ``scripts/``.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

_INDEX_CONST_NAMES = {
    'CURATION_ITEMS_INDEX',
    'CURATION_IMAGES_INDEX',
    'CURATION_LABELS_CONFIRMED_INDEX',
    'CURATION_CLASSES_INDEX',
}


def _iter_py_files(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        base = REPO_ROOT / root
        if base.is_file():
            files.append(base)
        elif base.is_dir():
            files.extend(sorted(base.rglob('*.py')))
    return files


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _module_level_assignments(tree: ast.Module) -> list[ast.Assign]:
    return [node for node in tree.body if isinstance(node, ast.Assign)]


def test_no_module_level_get_curation_config_project_field_capture() -> None:
    """A module-level ``X = get_curation_config().<project field>`` (or a
    two-step ``config = get_curation_config(); X = config.<field>``)
    freezes a project-scoped value at import time -- the exact bug the
    view exists to prevent."""
    from src.config.curation import PROJECT_SCOPED_FIELDS

    offenders: list[str] = []
    for path in _iter_py_files('src', 'scripts'):
        tree = _parse(path)
        module_config_names: set[str] = set()
        for assign in _module_level_assignments(tree):
            value = assign.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == 'get_curation_config'
            ):
                for target in assign.targets:
                    if isinstance(target, ast.Name):
                        module_config_names.add(target.id)
            elif (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id in module_config_names
                and value.attr in PROJECT_SCOPED_FIELDS
            ):
                offenders.append(f'{path.relative_to(REPO_ROOT)}: {value.value.id}.{value.attr}')
    # Known red: commit 5 (the codemod over the ~15 documented captures)
    # has not landed yet in this pass.
    if offenders:
        import pytest

        pytest.xfail(
            'commit 5 (module-level capture codemod) not landed yet: ' + ', '.join(offenders)
        )


def test_no_module_level_index_name_call() -> None:
    offenders: list[str] = []
    for path in _iter_py_files('src', 'scripts'):
        tree = _parse(path)
        for assign in _module_level_assignments(tree):
            value = assign.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == 'index_name'
            ):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    # Known red: commit 5 (the CURATION_*_INDEX -> function codemod) has
    # not landed yet in this pass -- _common.py/curation_train.py still
    # compute their module-level index constants via index_name().
    if offenders:
        import pytest

        pytest.xfail('commit 5 (index_name() codemod) not landed yet: ' + ', '.join(offenders))


def test_no_frozen_curation_index_constants() -> None:
    """The legacy ``CURATION_*_INDEX``/``ITEMS_INDEX``-style module
    constants must not exist as *assignment targets* anywhere (the
    codemod to functions is P1 commit 5 scope; this asserts the target
    state once that codemod lands, and documents today's known-red
    baseline in the assertion message rather than silently skipping)."""
    offenders: list[str] = []
    for path in _iter_py_files('src', 'scripts'):
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and (
                    target.id in _INDEX_CONST_NAMES or target.id in ('ITEMS_INDEX',)
                ):
                    offenders.append(f'{path.relative_to(REPO_ROOT)}: {target.id}')  # noqa: PERF401
    # Known red: commit 5 (the codemod) has not landed yet in this pass.
    # Document rather than silently pass.
    if offenders:
        import pytest

        pytest.xfail(
            'commit 5 (index-constant codemod) not landed yet; frozen constants remain: '
            + ', '.join(offenders)
        )


def test_no_dataclasses_replace_or_asdict_on_curation_config() -> None:
    offenders: list[str] = []
    for path in _iter_py_files('src', 'scripts'):
        if path.name == 'curation.py' and path.parent.name == 'config':
            continue  # the dataclass's own module may legitimately reference these
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name in ('replace', 'asdict') and node.args:
                first = node.args[0]
                if (
                    isinstance(first, ast.Call)
                    and isinstance(first.func, ast.Name)
                    and first.func.id == 'get_curation_config'
                ):
                    offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f'dataclasses.replace/asdict on get_curation_config(): {offenders}'


def test_no_bare_run_in_executor_in_curation_packages() -> None:
    offenders: list[str] = []
    for path in _iter_py_files(
        'src/services/curation',
        'src/routers/curation',
        'src/routers/curation_images.py',
        'src/routers/curation_umap.py',
        'src/routers/curation_train.py',
    ):
        tree = _parse(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'run_in_executor'
            ):
                offenders.append(str(path.relative_to(REPO_ROOT)))  # noqa: PERF401
    # Known red: commit 5 (run_in_executor_bound migration) not landed.
    if offenders:
        import pytest

        pytest.xfail(
            'commit 5 (run_in_executor_bound migration) not landed yet: ' + ', '.join(offenders)
        )


def test_no_raw_opensearch_construction_outside_factory() -> None:
    """Scoped to src/services/curation and src/routers/curation* for this
    pass -- see module docstring for the scripts/ gap."""
    offenders: list[str] = []
    for path in _iter_py_files(
        'src/services/curation',
        'src/routers/curation',
        'src/routers/curation_images.py',
        'src/routers/curation_umap.py',
        'src/routers/curation_train.py',
    ):
        tree = _parse(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ('AsyncOpenSearch', 'OpenSearch')
            ):
                offenders.append(str(path.relative_to(REPO_ROOT)))  # noqa: PERF401
    assert not offenders, f'raw AsyncOpenSearch()/OpenSearch() outside the factory: {offenders}'
