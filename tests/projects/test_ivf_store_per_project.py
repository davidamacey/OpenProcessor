"""IVF centroid store must resolve a per-project directory.

Before the fix, ``IVFCentroidStore.__init__`` fell back to module-level
constants (``IVF_STORE_DIR`` et al.) frozen at import time from
``get_curation_config().state_dir`` -- a global, unscoped field shared by
every project. Two projects' residual-clustering centroids/metadata/gate
files collided in the same directory. The fix resolves the directory
fresh on every construction from the CURRENTLY bound project's
``project_state_dir``.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import Any

import pytest

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore


pytestmark = pytest.mark.unbound


def _record(slug: str, tmp_path: Any) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    resources = resources_for_new(slug, base_curation_config())
    resources = dataclasses.replace(resources, project_state_dir=tmp_path / slug / 'state')
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )


def test_store_dir_resolves_per_currently_bound_project(tmp_path: Any) -> None:
    alpha, beta = _record('alpha', tmp_path), _record('beta', tmp_path)

    with bind_project(alpha):
        a = IVFCentroidStore()
    with bind_project(beta):
        b = IVFCentroidStore()

    assert a._dir == alpha.resources.project_state_dir / 'ivf_residuals'
    assert b._dir == beta.resources.project_state_dir / 'ivf_residuals'
    assert a._dir != b._dir
    assert str(alpha.resources.project_state_dir) in str(a._dir)
    assert str(beta.resources.project_state_dir) in str(b._dir)


def test_written_files_never_cross_contaminate(tmp_path: Any) -> None:
    alpha, beta = _record('alpha', tmp_path), _record('beta', tmp_path)

    with bind_project(alpha):
        a = IVFCentroidStore()
        a._dir.mkdir(parents=True, exist_ok=True)
        (a._dir / 'centroids.faiss').write_text('alpha-centroids')
        (a._dir / 'metadata.json').write_text('{"project": "alpha"}')
        (a._dir / 'gate.json').write_text('{"project": "alpha"}')

    with bind_project(beta):
        b = IVFCentroidStore()
        b._dir.mkdir(parents=True, exist_ok=True)
        (b._dir / 'metadata.json').write_text('{"project": "beta"}')

    # alpha's writes never landed under beta's dir, and vice versa.
    assert not (b._dir / 'centroids.faiss').exists()
    assert not (b._dir / 'gate.json').exists()
    assert (a._dir / 'metadata.json').read_text() == '{"project": "alpha"}'
    assert (b._dir / 'metadata.json').read_text() == '{"project": "beta"}'
    assert a._dir.resolve() != b._dir.resolve()
