"""The test databases' password files live under pytest's base temp, so a killed run's leftovers are pruned.

pytest keeps only the last ``tmp_path_retention_count`` (default 3) base temp directories
(https://docs.pytest.org/en/stable/how-to/tmp_path.html); a ``tempfile`` directory left by a killed
worker stays in /tmp forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase


pytestmark = [pytest.mark.db]


def test_password_files_live_under_the_pytest_base_temp(
    pg: PgDatabase, tmp_path_factory: pytest.TempPathFactory
) -> None:
    secrets = pg.settings.app_password_file.parent
    assert secrets.is_relative_to(tmp_path_factory.getbasetemp())
    assert {p.name for p in secrets.iterdir()} >= {
        'app_password',
        'owner_password',
        'super_password',
    }
