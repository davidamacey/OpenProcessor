"""Copy the image's model repository seed into the installed model repository.

The installer's preflight runs this inside the API image:

    python /app/export/sync_model_seed.py /opt/openprocessor/model_repo_seed /app/models

Files the install lacks are added. A model directory that carries a ``MODEL_VERSION`` file (one
integer, raised whenever the shipped model code or config changes in a way the old files cannot
serve) is replaced file by file when the seed's version is newer than the installed one; an install
without the file counts as version 0. Built engines and other files the seed does not ship are never
touched. Profile-managed config values (instance count, GPU id) are rewritten right afterwards by
the installer's profile step, so a replaced ``config.pbtxt`` does not keep the seed's defaults.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


VERSION_FILE = 'MODEL_VERSION'


def _version(model_dir: Path) -> int:
    try:
        return int((model_dir / VERSION_FILE).read_text().strip())
    except (OSError, ValueError):
        return 0


def _copy_tree(src: Path, dst: Path, *, overwrite: bool) -> None:
    for path in sorted(src.rglob('*')):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif overwrite or not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def sync(seed: Path, dest: Path) -> list[str]:
    """Merge the seed into dest and return the models whose files were replaced."""
    dest.mkdir(parents=True, exist_ok=True)
    updated: list[str] = []
    for model in sorted(p for p in seed.iterdir() if p.is_dir()):
        target = dest / model.name
        replace = (model / VERSION_FILE).is_file() and _version(model) > _version(target)
        _copy_tree(model, target, overwrite=replace)
        if replace:
            updated.append(model.name)
    for path in sorted(p for p in seed.iterdir() if p.is_file()):
        if not (dest / path.name).exists():
            shutil.copy2(path, dest / path.name)
    return updated


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print('usage: sync_model_seed.py SEED_DIR MODELS_DIR', file=sys.stderr)
        return 2
    updated = sync(Path(argv[1]), Path(argv[2]))
    if updated:
        print(f'updated model files: {", ".join(updated)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
