"""Physical directories of the named storage roots (02c section 5), from the existing settings.

No new variable: each root reads where today's code reads it. The ``projects`` root holds
``<slug>/...`` under ``OP_PROJECTS_DATA_ROOT``; the ``cache`` root is the crop cache
(``OP_CROP_CACHE_DIR``); ``upgrade`` lives under ``OP_STATE_DIR``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config.curation_accessors import get_curation_config
from src.config.env import env_path
from src.config.projects import projects_data_root


if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class StorageRootPaths:
    source: Path
    projects: Path
    models: Path
    cache: Path
    upgrade: Path


def storage_root_paths() -> StorageRootPaths:
    cfg = get_curation_config()
    return StorageRootPaths(
        source=cfg.source_root,
        projects=projects_data_root(),
        models=env_path('OP_TRITON_MODEL_REPO', '/app/models'),
        cache=cfg.crop_cache_dir,
        upgrade=cfg.state_dir / 'upgrade',
    )
