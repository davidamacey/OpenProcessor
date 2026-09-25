"""F-29 (fresh-start E2E findings 2026-09-25): bind-mount source dirs must
pre-exist, host-user-owned, in a fresh clone.

``yolo-api`` bind-mounts ``./test_images`` (docker-compose.yml). On a
fresh clone that host directory doesn't exist, so Docker auto-creates it
**root-owned** on the very first ``docker compose up``, and ``make
download-test-images`` (running as the host user) then fails to write
into it. Shipping ``test_images/.gitkeep`` means ``git clone`` itself
creates the directory, owned by whoever ran the clone, before compose
ever gets a chance to.

This only works because ``.gitignore`` excludes ``test_images/*``
(directory contents), not ``test_images/`` (the directory itself) --
git can never re-include a file whose parent directory is excluded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_check_ignore(path: str) -> bool:
    """True if `path` is gitignored (exit 0), False if not (exit 1)."""
    result = subprocess.run(
        ['git', 'check-ignore', '--quiet', path],
        check=False,
        cwd=REPO_ROOT,
        timeout=10,
    )
    return result.returncode == 0


def test_test_images_gitkeep_exists_on_disk() -> None:
    assert (REPO_ROOT / 'test_images' / '.gitkeep').exists()


def test_test_images_gitkeep_is_not_gitignored() -> None:
    assert not _git_check_ignore('test_images/.gitkeep')


def test_test_images_gitkeep_is_actually_tracked_by_git() -> None:
    result = subprocess.run(
        ['git', 'ls-files', '--error-unmatch', 'test_images/.gitkeep'],
        check=False,
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, 'test_images/.gitkeep is not tracked by git'


def test_downloaded_test_images_are_still_gitignored() -> None:
    """The placeholder must not accidentally un-ignore real test assets."""
    assert _git_check_ignore('test_images/bus.jpg')
