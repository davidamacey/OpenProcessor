"""Ultralytics is a dependency at one exact release, never vendored or patched.

The export used a vendored fork (a directory of patches applied to ``ultralytics`` at import time) to
embed TensorRT's EfficientNMS plugin. The documented ``nms`` export argument does the same job in
the official package (https://docs.ultralytics.com/modes/export/), so the fork, its licence lines
and the plugin are gone. Plans and the changelog keep the history, so ``docs/`` and
``CHANGELOG.md`` are not scanned.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORK_DIR = 'ultralytics' + '_patches'
PLUGIN = 'EfficientNMS' + '_TRT'
SCANNED_SUFFIXES = {'.py', '.md', '.mdx', '.toml', '.txt', '.yml', '.yaml', '.sh', '.json', '.cfg'}
SCANNED_NAMES = {'LICENSE', 'Makefile', 'Dockerfile', '.gitignore'}
# The plugin's operator name is also read by the API's adapters (their own change).
PLUGIN_SCOPE = ('export/', 'scripts/', 'docker/', 'tests/')


def _tracked() -> list[str]:
    out = subprocess.run(
        ['git', 'ls-files'], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    return [
        name
        for name in out.splitlines()
        if not name.startswith('docs/')
        and name not in {'CHANGELOG.md', 'tests/test_no_vendored_ultralytics.py'}
        and (Path(name).suffix in SCANNED_SUFFIXES or Path(name).name in SCANNED_NAMES)
        and (ROOT / name).is_file()
    ]


def test_the_vendored_fork_directory_is_gone() -> None:
    assert not (ROOT / 'src' / FORK_DIR).exists()
    assert not list(
        subprocess.run(
            ['git', 'ls-files', f'src/{FORK_DIR}'],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )


def test_nothing_tracked_names_the_fork() -> None:
    hits = [n for n in _tracked() if FORK_DIR in (ROOT / n).read_text(errors='replace')]
    assert not hits, hits


def test_the_export_side_never_names_the_plugin() -> None:
    hits = [
        n
        for n in _tracked()
        if n.startswith(PLUGIN_SCOPE) and PLUGIN in (ROOT / n).read_text(errors='replace')
    ]
    assert not hits, hits
