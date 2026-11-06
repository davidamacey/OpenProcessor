"""Index bodies for every role other than ``images`` and ``items``."""

from __future__ import annotations

from typing import Any

from src.clients.curation_opensearch.base import _plain_settings


def _leases_body() -> dict[str, Any]:
    """The job lease documents (``src/clients/job_lease.py``): one ``lease`` object per document,
    stored and never searched, so its fields are not mapped."""
    return {
        'settings': _plain_settings(),
        'mappings': {
            'dynamic': False,
            'properties': {'lease': {'type': 'object', 'enabled': False}},
        },
    }
