"""check_layering.py: import-direction contracts, with a shrink-only baseline of today's findings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_layering as guard
import pytest


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from conftest import ScratchRepo


def run(repo: ScratchRepo, baseline: Path, *extra: str) -> int:
    return guard.main(['--root', str(repo.root), '--baseline', str(baseline), *extra])


def test_a_function_level_router_import_in_a_service_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write(
        'src/services/_probe.py',
        'def f():\n    from src.routers.curation import _common\n    return _common\n',
    )
    assert run(scratch_repo, baseline_file('')) == 1


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
    ],
)
def test_each_contract_is_enforced(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str, source: str
) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo, baseline_file('')) == 1


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
    ],
)
def test_allowed_directions_pass(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path], path: str, source: str
) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo, baseline_file('')) == 0


def test_relative_imports_are_resolved(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/services/pkg/a.py', 'from ...routers import models\n')
    assert run(scratch_repo, baseline_file('')) == 1


def test_a_baselined_import_may_stay_but_a_second_one_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/services/a.py', 'from src.routers.x import y\n')
    base = baseline_file('src/services/a.py -> src.routers.x\t1\tmoves in PR 4\n')
    assert run(scratch_repo, base) == 0
    scratch_repo.write(
        'src/services/a.py', 'from src.routers.x import y\nfrom src.routers.x import z\n'
    )
    assert run(scratch_repo, base) == 1


def test_a_stale_entry_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/services/a.py', 'x = 1\n')
    assert (
        run(scratch_repo, baseline_file('src/services/a.py -> src.routers.x\t1\tmoves in PR 4\n'))
        == 1
    )


def test_a_syntax_error_in_a_scanned_file_fails_closed(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/services/a.py', 'def broken(:\n')
    assert run(scratch_repo, baseline_file('')) == 1


def test_the_real_repo_passes() -> None:
    assert guard.main([]) == 0
