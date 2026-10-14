"""check_layering.py: import-direction contracts, one permanent reasoned exception, no baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_layering as guard
import pytest


if TYPE_CHECKING:
    from conftest import ScratchRepo


_FACTORY = 'src/clients/opensearch/client.py'
_FACTORY_SOURCE = (
    'from src.services.projects.guard import g\nfrom src.services.projects.registry import r\n'
)


def run(repo: ScratchRepo) -> int:
    # The permanent exception must still have its import, or the guard reports it stale.
    if not (repo.root / _FACTORY).exists():
        repo.write(_FACTORY, _FACTORY_SOURCE)
    return guard.main(['--root', str(repo.root)])


def test_a_function_level_router_import_in_a_service_fails(scratch_repo: ScratchRepo) -> None:
    scratch_repo.write(
        'src/services/_probe.py',
        'def f():\n    from src.routers.curation import _common\n    return _common\n',
    )
    assert run(scratch_repo) == 1


@pytest.mark.parametrize(
    ('path', 'source'),
    [
        ('src/services/a.py', 'import src.routers.curation\n'),
        ('src/clients/a.py', 'from src.routers import models\n'),
        ('src/core/a.py', 'from src.routers.x import y\n'),
        ('src/config/a.py', 'from src.routers.x import y\n'),
        ('src/utils/a.py', 'from src.routers.x import y\n'),
        ('src/schemas/a.py', 'from src.routers.x import y\n'),
        ('scripts/curation/a.py', 'from src.routers.x import y\n'),
        ('src/clients/b.py', 'from src.services.x import y\n'),
        ('src/clients/c.py', 'import src.services\n'),
        ('src/services/d.py', 'from fastapi import HTTPException\n'),
        ('src/clients/d.py', 'import starlette.responses\n'),
        ('src/config/d.py', 'from fastapi import Depends\n'),
        ('src/utils/d.py', 'from starlette.requests import Request\n'),
        ('src/schemas/d.py', 'from fastapi import UploadFile\n'),
        ('src/config/e.py', 'from src.services.x import y\n'),
        ('src/utils/e.py', 'from src.clients.x import y\n'),
        ('src/schemas/e.py', 'from src.clients import x\n'),
        ('src/domain/a.py', 'from src.routers.x import y\n'),
        ('src/domain/b.py', 'from fastapi import HTTPException\n'),
        ('src/domain/c.py', 'from src.services.x import y\n'),
        ('src/domain/d.py', 'import src.clients.x\n'),
    ],
)
def test_each_contract_is_enforced(scratch_repo: ScratchRepo, path: str, source: str) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo) == 1


@pytest.mark.parametrize(
    ('path', 'source'),
    [
        ('src/routers/a.py', 'from fastapi import APIRouter\nfrom src.services.x import y\n'),
        ('src/routers/b.py', 'from src.routers.c import d\n'),
        ('src/services/a.py', 'from src.clients.x import y\nfrom src.config import z\n'),
        ('src/clients/a.py', 'from src.config import z\nfrom src.utils.q import r\n'),
        ('src/main.py', 'from src.routers.a import router\nfrom fastapi import FastAPI\n'),
        ('scripts/codegen/a.py', 'from src.routers.a import router\n'),
        ('scripts/curation/a.py', 'from src.services.x import y\nfrom src.clients.z import w\n'),
        ('tests/a.py', 'from src.routers.a import router\nfrom fastapi import FastAPI\n'),
        ('src/services/b.py', 'import fastapi_utils_not_fastapi\n'),
        ('src/domain/ok.py', 'from src.config import z\nfrom src.utils.q import r\n'),
        ('src/clients/ok.py', 'from src.domain.x import y\n'),
        ('src/services/ok.py', 'from src.domain.x import y\n'),
    ],
)
def test_allowed_directions_pass(scratch_repo: ScratchRepo, path: str, source: str) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo) == 0


def test_relative_imports_are_resolved(scratch_repo: ScratchRepo) -> None:
    scratch_repo.write('src/services/pkg/a.py', 'from ...routers import models\n')
    assert run(scratch_repo) == 1


def test_the_permanent_exception_passes_and_a_second_one_in_the_same_file_fails(
    scratch_repo: ScratchRepo,
) -> None:
    scratch_repo.write(_FACTORY, _FACTORY_SOURCE)
    assert run(scratch_repo) == 0
    scratch_repo.write(_FACTORY, _FACTORY_SOURCE + 'from src.services.projects.other import o\n')
    assert run(scratch_repo) == 1


def test_the_exception_does_not_cover_another_file(scratch_repo: ScratchRepo) -> None:
    scratch_repo.write(_FACTORY, _FACTORY_SOURCE)
    scratch_repo.write(
        'src/clients/opensearch/other.py', 'from src.services.projects.guard import g\n'
    )
    assert run(scratch_repo) == 1


def test_a_stale_exception_fails(scratch_repo: ScratchRepo) -> None:
    scratch_repo.write(_FACTORY, 'from src.services.projects.guard import g\n')
    assert run(scratch_repo) == 1


def test_a_syntax_error_in_a_scanned_file_fails_closed(scratch_repo: ScratchRepo) -> None:
    scratch_repo.write('src/services/a.py', 'def broken(:\n')
    assert run(scratch_repo) == 1


def test_the_real_repo_passes() -> None:
    assert guard.main([]) == 0
