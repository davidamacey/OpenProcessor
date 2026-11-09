"""A 0.6.0 settings document that still stores ``defaults.cluster`` loads with the key dropped."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from src.services.upgrade.from_060.parse_core import parse_settings
from src.services.upgrade.from_060.snapshot import Snapshot
from src.services.upgrade.from_060.sources import SETTINGS_DOC_ID


def test_stored_cluster_default_is_dropped() -> None:
    snap = Snapshot(
        docs={
            'op_settings': {
                SETTINGS_DOC_ID: {'defaults': {'cluster': 'ivf', 'sort': 'atypicality'}}
            }
        }
    )
    project = cast('Any', SimpleNamespace(legacy_indexes={'settings': 'op_settings'}))
    settings, _ = parse_settings(snap, project)
    assert settings is not None
    assert settings.defaults == {'sort': 'atypicality'}
