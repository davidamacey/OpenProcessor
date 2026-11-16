"""check_layering.py: import-direction contracts, one permanent reasoned exception, no baseline."""

from __future__ import annotations

import json
from pathlib import Path
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


@pytest.mark.parametrize(
    ('path', 'source'),
    [
        # the database layer stays below services, clients and the web
        ('src/db/a.py', 'from src.services.x import y\n'),
        ('src/db/a.py', 'import src.clients\n'),
        ('src/db/a.py', 'from src.routers import x\n'),
        ('src/db/a.py', 'from fastapi import Depends\n'),
        ('src/db/models/a.py', 'import starlette.requests\n'),
        # pure layers never reach the database
        ('src/domain/a.py', 'from src.db import engine\n'),
        ('src/config/a.py', 'import src.db.models\n'),
        ('src/utils/a.py', 'from src.db.session import project_tx\n'),
        ('src/schemas/a.py', 'from src.db import base\n'),
        # scripts and everything else may use only the runtime leaf of the task layer
        ('scripts/curation/a.py', 'from src.tasks import app\n'),
        ('scripts/curation/a.py', 'import src.tasks.families_a\n'),
        # the task layer carries no web stack
        ('src/tasks/a.py', 'from fastapi import FastAPI\n'),
        ('src/tasks/families_a.py', 'import starlette\n'),
        ('src/tasks/a.py', 'from src.routers.x import y\n'),
        # the runtime leaf imports only db, config, domain, utils (even inside a function)
        ('src/tasks/runtime/a.py', 'from src.services.x import y\n'),
        ('src/tasks/runtime/a.py', 'def f():\n    from src.services import x\n    return x\n'),
        ('src/tasks/runtime/a.py', 'from src.clients.task_queue import enqueue\n'),
        ('src/tasks/runtime/a.py', 'from src.tasks.families_a import run\n'),
        ('src/tasks/runtime/a.py', 'from src.core.logging import get_logger\n'),
        # a service, trainer, exporter or bake-off module reaches no task module but the runtime
        ('src/services/jobs/a.py', 'from src.tasks.families_a import run\n'),
        ('docker/trainer/a.py', 'from src.tasks.families_a import run\n'),
        ('export/a.py', 'import src.tasks.app\n'),
        ('scripts/curation/bakeoff/a.py', 'from src.tasks.families_a import run\n'),
        # handlers go through services, not repositories
        ('src/routers/a.py', 'from src.db.repos import jobs\n'),
        ('src/routers/a.py', 'import src.db.repos.jobs\n'),
        # the frame engine's image carries src.domain and src.utils only; nothing imports it
        ('frame_engine/a.py', 'from src.config.settings import Settings\n'),
        ('frame_engine/a.py', 'from src.services.curation import wire\n'),
        ('src/services/curation/a.py', 'from frame_engine.engine import FrameEngine\n'),
        ('scripts/bench/a.py', 'import frame_engine.front\n'),
    ],
)
def test_the_control_plane_contracts_are_enforced(
    scratch_repo: ScratchRepo, path: str, source: str
) -> None:
    scratch_repo.write(path, source)
    assert run(scratch_repo) == 1


@pytest.mark.parametrize(
    ('path', 'source'),
    [
        ('src/db/a.py', 'from sqlalchemy import select\nfrom src.config.database import X\n'),
        ('src/db/a.py', 'from src.domain.job_states import STATUSES\nfrom src.utils.q import r\n'),
        ('src/core/a.py', 'from src.db.session import project_tx\nfrom fastapi import Depends\n'),
        ('src/services/a.py', 'from src.db.repos import jobs\n'),
        ('src/routers/a.py', 'from src.db.session import project_tx\n'),
        ('src/tasks/a.py', 'from src.services.x import y\nfrom src.db import engine\n'),
        (
            'src/tasks/runtime/a.py',
            'from src.db.repos.jobs import claim\nfrom src.config import x\n',
        ),
        ('src/tasks/runtime/a.py', 'from src.domain import job_states\nfrom src.utils import q\n'),
        ('src/tasks/runtime/a.py', 'from src.tasks.runtime import job_task\nimport redis\n'),
        ('src/tasks/runtime/a.py', 'from . import heartbeat\n'),
        ('src/services/jobs/a.py', 'from src.tasks.runtime.job_task import JobTask\n'),
        ('docker/trainer/a.py', 'from src.tasks.runtime import gpu_app\n'),
        ('export/a.py', 'from src.tasks.runtime.gpu_app import make_app\n'),
        ('scripts/curation/bakeoff/a.py', 'from src.tasks.runtime import gpu_app\n'),
        ('scripts/db/a.py', 'from src.db import roles\n'),
        ('src/tasks/families_a.py', 'from src.tasks.runtime import x\nfrom src.tasks import app\n'),
        (
            'frame_engine/a.py',
            'from src.domain import frame_engine_wire\nfrom src.utils import q\n',
        ),
        ('frame_engine/a.py', 'from frame_engine.slab import SlabAllocator\nimport cupy\n'),
    ],
)
def test_the_allowed_control_plane_directions_pass(
    scratch_repo: ScratchRepo, path: str, source: str
) -> None:
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


def test_only_the_harness_engine_stand_in_may_import_the_engine(scratch_repo: ScratchRepo) -> None:
    scratch_repo.write('docker/test/fake_frame_engine.py', 'from frame_engine.front import F\n')
    assert run(scratch_repo) == 0
    scratch_repo.write('docker/test/fake_other.py', 'from frame_engine.front import F\n')
    assert run(scratch_repo) == 1


# The Rust crate rule: plan 18a section 2.2, read from `cargo metadata --no-deps` output.
_METADATA = Path(__file__).parent / 'fixtures' / 'native_metadata.json'


def _metadata() -> dict:
    return json.loads(_METADATA.read_text(encoding='utf-8'))


def _add_dependency(metadata: dict, crate: str, dependency: str, kind: str | None = None) -> None:
    package = next(p for p in metadata['packages'] if p['name'] == crate)
    package['dependencies'].append(
        {'name': dependency, 'kind': kind, 'path': f'/w/native/{dependency}'}
    )


def test_the_fixture_workspace_follows_the_crate_table() -> None:
    assert guard.crate_violations(_metadata()) == []


@pytest.mark.parametrize(
    ('crate', 'dependency', 'kind'),
    [
        ('op_config', 'op_db', None),
        ('op_db', 'op_jobs', None),
        ('op_config', 'op_db', 'dev'),
        ('op_jobs', 'op_storage', 'build'),
        ('frame_engine', 'op_config', None),
        ('op_jobs', 'frame_engine', None),
    ],
)
def test_a_crate_dependency_outside_the_table_fails(
    crate: str, dependency: str, kind: str | None
) -> None:
    metadata = _metadata()
    _add_dependency(metadata, crate, dependency, kind)
    problems = guard.crate_violations(metadata)
    assert len(problems) == 1
    assert crate in problems[0]
    assert dependency in problems[0]


def test_a_workspace_crate_without_a_table_row_fails() -> None:
    metadata = _metadata()
    metadata['packages'].append({'name': 'op_unlisted', 'dependencies': []})
    assert guard.crate_violations(metadata) == [
        'op_unlisted: a workspace crate with no row in CRATE_DEPENDENCIES'
    ]


def test_an_external_package_of_the_same_name_is_not_a_workspace_dependency() -> None:
    metadata = _metadata()
    package = next(p for p in metadata['packages'] if p['name'] == 'op_config')
    package['dependencies'].append({'name': 'op_db', 'kind': None, 'path': None})
    assert guard.crate_violations(metadata) == []


def test_the_crate_rule_reads_a_metadata_file(tmp_path: Path) -> None:
    assert guard.main(['--crates', '--metadata', str(_METADATA)]) == 0
    metadata = _metadata()
    _add_dependency(metadata, 'op_config', 'op_db')
    bad = tmp_path / 'metadata.json'
    bad.write_text(json.dumps(metadata), encoding='utf-8')
    assert guard.main(['--crates', '--metadata', str(bad)]) == 1
