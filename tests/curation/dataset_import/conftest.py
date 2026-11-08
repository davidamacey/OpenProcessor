from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.db_fixture import import_database, world  # noqa: F401 - fixtures


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_region_profile_registry() -> Iterator[None]:
    """``activate_region_profile`` registers into the process-wide profile
    registry; leaving it set makes every later test on the worker see an
    active region profile (the no-profile gating tests then fail)."""
    yield
    from src.services.detection import profile_registry

    profile_registry._reset_registry_for_tests()


@pytest.fixture(autouse=True)
def _import_database(request: pytest.FixtureRequest) -> None:
    """A ``db`` test of this directory runs on the Harness database (``db_fixture``)."""
    if request.node.get_closest_marker('db') is not None:
        request.getfixturevalue('import_database')
