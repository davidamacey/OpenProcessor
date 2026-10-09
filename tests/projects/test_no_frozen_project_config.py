"""Static guards for project isolation (projects_plan.md §3.3/§2.4):

- nothing captures a project-scoped value at import time (a captured
  string would pin every request to one project's index);
- no bare ``run_in_executor`` in the curation request code (it drops the
  bound project on the thread hop);
- no ``AsyncOpenSearch(...)`` / ``OpenSearch(...)`` construction anywhere
  in ``src/`` or ``scripts/`` outside the guarded factories;
- every ``scripts/curation`` entry point takes ``--project`` and binds it.
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
    'ITEMS_INDEX',
    'UMAP_STATE_INDEX',
    'UMAP_VIZ_STATE_INDEX',
}

# The only places allowed to construct an OpenSearch client: the shared
# API client wrapper (guarded by make_curation_opensearch) and the guard
# module's own factories.
_CLIENT_FACTORIES = frozenset({'src/clients/opensearch.py', 'src/services/projects/guard.py'})

_CURATION_REQUEST_CODE = (
    'src/services/curation',
    'src/routers/curation',
    'src/routers/curation_images.py',
    'src/routers/curation_umap.py',
    'src/routers/curation_train',
    'src/services/training/preflight_checks.py',
    'src/services/training/promote_gate.py',
)


def _iter_py_files(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        base = REPO_ROOT / root
        if base.is_file():
            files.append(base)
        elif base.is_dir():
            files.extend(sorted(base.rglob('*.py')))
    return files


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _import_time_nodes(tree: ast.Module) -> list[ast.AST]:
    """Everything evaluated when the module is imported: module and class
    bodies, decorators and default argument values -- never a function
    body."""
    out: list[ast.AST] = []

    def _visit(body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                out.extend(stmt.args.defaults)
                out.extend(d for d in stmt.args.kw_defaults if d is not None)
                out.extend(stmt.decorator_list)
            elif isinstance(stmt, ast.ClassDef):
                out.extend(stmt.decorator_list)
                _visit(stmt.body)
            elif isinstance(stmt, ast.If | ast.Try):
                _visit(stmt.body)
                _visit(stmt.orelse)
                for handler in getattr(stmt, 'handlers', []):
                    _visit(handler.body)
            else:
                out.append(stmt)

    _visit(tree.body)
    return out


def test_no_import_time_read_of_a_project_scoped_field() -> None:
    """``X = get_curation_config().items_index``, ``config.export_root`` as a
    default argument, and friends freeze one project's value at import."""
    from src.config.curation import PROJECT_SCOPED_FIELDS

    offenders: list[str] = []
    for path in _iter_py_files('src', 'scripts'):
        tree = _parse(path)
        config_names = {'config', 'cfg', '_config', '_cfg', 'CFG'}
        for node in _import_time_nodes(tree):
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Attribute)
                    and sub.attr in PROJECT_SCOPED_FIELDS
                    and (
                        (isinstance(sub.value, ast.Name) and sub.value.id in config_names)
                        or (
                            isinstance(sub.value, ast.Call)
                            and isinstance(sub.value.func, ast.Name)
                            and sub.value.func.id == 'get_curation_config'
                        )
                    )
                ):
                    offenders.append(f'{_rel(path)}:{sub.lineno} {ast.unparse(sub)}')  # noqa: PERF401
    assert offenders == []


def test_no_module_level_index_name_call() -> None:
    offenders = [
        f'{_rel(path)}:{sub.lineno}'
        for path in _iter_py_files('src', 'scripts')
        for node in _import_time_nodes(_parse(path))
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id in ('index_name', 'idx')
    ]
    assert offenders == []


def test_no_frozen_index_name_constants() -> None:
    offenders: list[str] = []
    for path in _iter_py_files('src', 'scripts'):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Assign):
                offenders.extend(
                    f'{_rel(path)}:{node.lineno} {target.id}'
                    for target in node.targets
                    if isinstance(target, ast.Name) and target.id in _INDEX_CONST_NAMES
                )
    assert offenders == []


def test_no_dataclasses_replace_or_asdict_on_curation_config() -> None:
    offenders: list[str] = []
    for path in _iter_py_files('src', 'scripts'):
        if path.name == 'curation.py' and path.parent.name == 'config':
            continue  # the dataclass's own module may legitimately reference these
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
            first = node.args[0]
            if (
                name in ('replace', 'asdict')
                and isinstance(first, ast.Call)
                and isinstance(first.func, ast.Name)
                and first.func.id == 'get_curation_config'
            ):
                offenders.append(_rel(path))
    assert not offenders, f'dataclasses.replace/asdict on get_curation_config(): {offenders}'


def test_no_bare_run_in_executor_in_curation_request_code() -> None:
    """Request code runs in a per-request context; ``run_in_executor``
    would drop it. (Scripts bind the whole process, so their thread hops
    keep the binding.)"""
    offenders = [
        f'{_rel(path)}:{node.lineno}'
        for path in _iter_py_files(*_CURATION_REQUEST_CODE)
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'run_in_executor'
    ]
    assert offenders == []


def test_no_raw_opensearch_construction_outside_the_factories() -> None:
    offenders: list[str] = []
    for path in _iter_py_files('src', 'scripts'):
        if _rel(path) in _CLIENT_FACTORIES:
            continue
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
            if name in ('AsyncOpenSearch', 'OpenSearch'):
                offenders.append(f'{_rel(path)}:{node.lineno}')
    assert not offenders, f'raw AsyncOpenSearch()/OpenSearch() outside the factory: {offenders}'


def test_every_curation_script_takes_and_binds_a_project() -> None:
    offenders: list[str] = []
    for path in _iter_py_files('scripts/curation'):
        text = path.read_text(encoding='utf-8')
        if 'argparse.ArgumentParser(' not in text or "__name__ == '__main__'" not in text:
            continue
        if path.parent.name == 'bakeoff':
            continue  # the evaluator container's own tools; no curation config
        if 'add_project_argument(' not in text:
            offenders.append(f'{_rel(path)}: no --project')
        if 'bind_script_project(' not in text:
            offenders.append(f'{_rel(path)}: --project never bound')
    assert offenders == []


def test_no_index_name_literals() -> None:
    """A literal index name (``'op_items'``, ``'op_prj_default__items'``
    and friends) pins code to one project's index; every index name comes
    from the bound project (``index_name(cfg, role)``). Only
    ``src/config/curation.py`` spells the ``default`` names, as the shape of
    an explicitly constructed ``CurationConfig``."""
    import re

    from src.config.curation import IndexRole

    roles = '|'.join(role.value for role in IndexRole)
    pattern = re.compile(rf'^op_(prj_[a-z0-9-]+__)?({roles}|curation_settings|clusters)$')
    offenders = [
        f'{_rel(path)}:{node.lineno} {node.value!r}'
        for path in _iter_py_files('src', 'scripts', 'docker')
        if _rel(path) != 'src/config/curation.py'
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and pattern.match(node.value)
    ]
    assert offenders == []
