"""Physical directories of the named storage roots (02c section 5), from the existing settings.

No new variable: each root reads where today's code reads it. The ``projects`` root holds
``<slug>/...`` under ``OP_PROJECTS_DATA_ROOT``; the ``cache`` root is the crop cache
(``OP_CROP_CACHE_DIR``); ``upgrade`` lives under ``OP_STATE_DIR``. ``bakeoff_results`` is the per-project
bake-off results directory where it already is (``<state>/projects/<slug>/bakeoff_jobs/out``): key
``bakeoff_results/<slug>/<job_id>``. Uploaded models keep the key ``models/pytorch_models/<file>``;
that sub-folder is the separate ``pytorch_models`` mount, not a folder of the models root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.curation_accessors import get_curation_config
from src.config.env import env_path
from src.config.projects import projects_data_root


PYTORCH_MODELS_DIR = Path('/app/pytorch_models')


@dataclass(frozen=True)
class StorageRootPaths:
    source: Path
    projects: Path
    models: Path
    cache: Path
    upgrade: Path
    bakeoff_results: Path
    pytorch_models: Path


def storage_root_paths() -> StorageRootPaths:
    cfg = get_curation_config()
    return StorageRootPaths(
        source=cfg.source_root,
        projects=projects_data_root(),
        models=env_path('OP_TRITON_MODEL_REPO', '/app/models'),
        cache=cfg.crop_cache_dir,
        upgrade=cfg.state_dir / 'upgrade',
        bakeoff_results=cfg.state_dir / 'projects',
        pytorch_models=PYTORCH_MODELS_DIR,
    )
