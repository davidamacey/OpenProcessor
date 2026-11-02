"""--image-tag mode: every image resolves to ${OP_IMAGE_REPO}/<image>:<tag>, never a digest."""

from __future__ import annotations

from typing import TYPE_CHECKING

from _setup_dry_run_helpers import configure, env_file
from test_lifecycle import install


if TYPE_CHECKING:
    from installer_harness import Shimmed

BUILD_ENVS = (
    'OP_API_IMAGE',
    'OP_EXPORT_IMAGE',
    'OP_EVALUATOR_IMAGE',
    'OP_TRITON_IMAGE',
    'OP_SEGMENTER_IMAGE',
    'OP_TRAINER_IMAGE',
    'OP_CROPWRIGHT_IMAGE',
)


def _local_images(shimmed: Shimmed, repo: str, tag: str) -> None:
    images = shimmed.state / 'images'
    images.mkdir(exist_ok=True)
    for name in ('openprocessor', 'openprocessor-export', 'openprocessor-triton'):
        (images / f'{repo}_{name}_{tag}').write_text(f'{repo}/{name}@sha256:{"0" * 64}\n')


def test_tag_mode_clears_every_build_image_pin_left_by_a_lock_install(shimmed: Shimmed) -> None:
    locked = configure(shimmed)
    assert locked.returncode == 0, locked.stderr
    pinned = env_file(shimmed)
    assert all('@sha256:' in pinned[k] for k in BUILD_ENVS), pinned
    _local_images(shimmed, 'opinst', 'localtest')
    result = configure(shimmed, '--image-tag', 'localtest', OP_IMAGE_REPO='opinst')
    assert result.returncode == 0, result.stderr
    env = env_file(shimmed)
    assert {k: env[k] for k in BUILD_ENVS} == dict.fromkeys(BUILD_ENVS, '')


def test_repair_of_a_tag_install_keeps_the_image_repo(shimmed: Shimmed) -> None:
    # --repair reuses the installed tag but not the repo prefix, so a local build under
    # another repo was looked up (and pulled) as davidamacey/<image>:<tag>.
    _local_images(shimmed, 'opinst', 'localtest')
    first = install(shimmed, '--image-tag', 'localtest', OP_IMAGE_REPO='opinst')
    assert first.returncode == 0, first.stderr[-1500:]
    repaired = shimmed.run(['--repair', '--unattended', '--dir', 'inst'])
    assert repaired.returncode == 0, repaired.stderr[-1500:]
    env = env_file(shimmed)
    assert env['OP_IMAGE_REPO'] == 'opinst'
    assert env['OP_IMAGE_TAG'] == 'localtest'
    assert {k: env[k] for k in BUILD_ENVS} == dict.fromkeys(BUILD_ENVS, '')
