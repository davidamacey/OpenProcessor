"""F-29/F-71 (fresh-start E2E findings 2026-09-25): bind-mount source dirs
must exist, host-user-owned, before compose ever runs -- without
breaking `git pull`.

``yolo-api`` bind-mounts ``./test_images``. On a fresh clone that host
directory doesn't exist, so Docker auto-creates it **root-owned** on the
very first ``docker compose up``, and ``make download-test-images``
(running as the host user) then fails to write into it.

F-29's first fix shipped a tracked ``test_images/.gitkeep`` placeholder
so ``git clone`` itself created the directory, host-user-owned, before
compose ever ran. That backfired as F-71: on an EXISTING install whose
``test_images/`` was already root-owned (from before this fix ever
shipped), ``git pull`` needs write access to materialize the new
tracked file there -- permission denied, and the pull aborted
half-applied (HEAD unmoved, working tree partially modified).

Fixed properly instead: no file is tracked inside ``test_images/`` at
all (it's fully gitignored again), and ``make up`` / ``make
up-monitoring`` depend on ``ensure-host-bind-mount-dirs``, a Makefile
target that ``mkdir -p``s every bind-mount source directory as the
invoking host user *before* compose ever runs. A `git pull` never
touches that directory, so it can never be blocked by its ownership.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (REPO_ROOT / 'Makefile').read_text()


def _git_check_ignore(path: str) -> bool:
    """True if `path` is gitignored (exit 0), False if not (exit 1)."""
    result = subprocess.run(
        ['git', 'check-ignore', '--quiet', path],
        check=False,
        cwd=REPO_ROOT,
        timeout=10,
    )
    return result.returncode == 0


def test_no_file_is_tracked_inside_test_images() -> None:
    """The F-71 fix: nothing inside test_images/ may be a tracked file --
    that's exactly what made `git pull` require write access there."""
    result = subprocess.run(
        ['git', 'ls-files', 'test_images/'],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == '', (
        f'test_images/ has tracked files (breaks git pull on a root-owned '
        f'existing install, F-71): {result.stdout!r}'
    )


def test_test_images_is_fully_gitignored() -> None:
    assert _git_check_ignore('test_images/bus.jpg')
    assert _git_check_ignore('test_images/anything.jpg')


def test_ensure_host_bind_mount_dirs_target_exists() -> None:
    assert '.PHONY: ensure-host-bind-mount-dirs' in MAKEFILE
    assert 'ensure-host-bind-mount-dirs:' in MAKEFILE


def test_up_targets_depend_on_ensure_host_bind_mount_dirs() -> None:
    """F-71: `up` and `up-monitoring` must create the bind-mount source
    dirs (as the host user) BEFORE compose runs and could auto-create
    them root-owned instead."""
    assert 'up: ensure-host-bind-mount-dirs' in MAKEFILE
    assert 'up-monitoring: ensure-host-bind-mount-dirs' in MAKEFILE


def test_ensure_host_bind_mount_dirs_creates_test_images(tmp_path: Path) -> None:
    """Actually run the target's mkdir step (isolated to a scratch CWD --
    no real docker compose call) and confirm it creates test_images/ as
    the current (host) user, not root."""
    scratch = tmp_path / 'repo_root'
    scratch.mkdir()
    result = subprocess.run(
        ['bash', '-c', 'mkdir -p test_images data/source cache/huggingface cache/vllm'],
        check=True,
        cwd=scratch,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    created = scratch / 'test_images'
    assert created.is_dir()
    assert created.stat().st_uid == Path(scratch).stat().st_uid


def test_recovery_step_is_documented_for_an_already_root_owned_install() -> None:
    gitignore = (REPO_ROOT / '.gitignore').read_text()
    assert 'chown' in gitignore, (
        'F-71: an install whose test_images/ is already root-owned from '
        'before this fix needs a documented one-time recovery step'
    )
