"""Database-tier helpers shared by the config-store tests: the tests/db fixtures live one directory over."""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'db'))


@pytest.fixture(autouse=True)
def _forget_project_ids():
    """A module's database is fresh, so a slug cached from the previous module names no row."""
    from src.services.config_store import bound_project

    bound_project._IDS.clear()
    yield
    bound_project._IDS.clear()
