"""check_vulture.py: unused code is rejected unless it is in the shrink-only baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import check_vulture as guard


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from conftest import ScratchRepo


def run(repo: ScratchRepo, baseline: Path, *extra: str) -> int:
    return guard.main(['--root', str(repo.root), '--baseline', str(baseline), *extra])


def test_an_unused_function_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/_probe.py', 'def never_called():\n    return 1\n')
    assert run(scratch_repo, baseline_file('')) == 1


def test_a_function_used_in_another_file_passes(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/a.py', 'def helper():\n    return 1\n')
    scratch_repo.write('src/b.py', 'from src.a import helper\n\nprint(helper())\n')
    assert run(scratch_repo, baseline_file('')) == 0


def test_route_handlers_and_validators_are_not_dead(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write(
        'src/routers/r.py',
        'router = object()\n\n\n@router.get("/x")\nasync def get_x():\n    return 1\n\n\n'
        '@field_validator("a")\ndef check_a(v):\n    return v\n',
    )
    # `router` is used by the decorator; the handler and validator are registered by decorators.
    assert run(scratch_repo, baseline_file('')) == 0


def test_declared_class_fields_are_data_not_dead_code(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write(
        'src/m.py',
        'class Wire:\n    never_read: int = 0\n    other: str\n',
    )
    scratch_repo.write('src/use.py', 'from src.m import Wire\n\nprint(Wire)\n')
    assert run(scratch_repo, baseline_file('')) == 0


def test_a_whitelist_entry_silences_a_reviewed_false_positive(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/_probe.py', 'def called_by_name():\n    return 1\n')
    scratch_repo.write(
        'scripts/codegen/vulture_whitelist.py',
        'from src._probe import called_by_name\n\ncalled_by_name  # dispatched by name from a config file\n',
    )
    assert run(scratch_repo, baseline_file('')) == 0


def test_a_baselined_finding_may_stay_and_a_new_one_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/_probe.py', 'def old_dead():\n    return 1\n')
    base = baseline_file('src/_probe.py::function old_dead\t1\tdelete in PR 2\n')
    assert run(scratch_repo, base) == 0
    scratch_repo.write(
        'src/_probe.py', 'def old_dead():\n    return 1\n\n\ndef new_dead():\n    return 2\n'
    )
    assert run(scratch_repo, base) == 1


def test_a_stale_entry_fails(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/_probe.py', 'X = 1\n')
    scratch_repo.write('src/use.py', 'from src._probe import X\n\nprint(X)\n')
    assert (
        run(scratch_repo, baseline_file('src/_probe.py::function gone\t1\tdelete in PR 2\n')) == 1
    )


def test_third_party_patches_are_excluded(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write('src/ultralytics_patches/p.py', 'def dead():\n    return 1\n')
    assert run(scratch_repo, baseline_file('')) == 0


def test_the_real_repo_passes() -> None:
    assert guard.main([]) == 0


def test_handlers_registered_on_an_app_factory_variable_are_not_dead(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write(
        'src/main.py',
        'def create_app():\n    application = object()\n\n'
        '    @application.middleware("http")\n    async def timing(request, call_next):\n'
        '        return await call_next(request)\n\n    return application\n\n\ncreate_app()\n',
    )
    assert run(scratch_repo, baseline_file('')) == 0


def test_a_whitelisted_attribute_name_is_a_use(
    scratch_repo: ScratchRepo, baseline_file: Callable[[str], Path]
) -> None:
    scratch_repo.write(
        'src/m.py', 'class H:\n    def do_GET(self):\n        return 1\n\n\nprint(H)\n'
    )
    scratch_repo.write(
        'scripts/codegen/vulture_whitelist.py',
        'from types import SimpleNamespace\n\n_ = SimpleNamespace()\n_.do_GET  # http.server dispatch\n',
    )
    assert run(scratch_repo, baseline_file('')) == 0
