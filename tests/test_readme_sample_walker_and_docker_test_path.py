"""F-21 (fresh-start E2E findings 2026-09-25): README command blocks must
actually work against this repo's real compose/scripts/env, not a
plausible-looking but broken variant.

Two concrete breakages fixed here:

1. **Sample walker 200/200 failed.** `make sample-coco-readme` writes to
   ``data/samples/coco_va_readme/``, but the compose mount for ingest is
   ``${OP_SOURCE_ROOT_HOST:-./data/source}:/data/source``. README's old
   ``--root data/samples/coco_va_readme/images`` (a container-relative
   ``/app/data/samples/...`` path) is outside that mount entirely, so
   every image 404s ``unservable_path``. README must set
   ``OP_SOURCE_ROOT_HOST=./data/samples`` and pass a ``--root`` under
   ``/data/source/...``.
2. **F-31: the "Docker-only" pytest path doesn't exist.** The production
   ``yolo-api`` image installs only ``requirements.txt`` -- no pytest, no
   ``requirements-test.txt`` -- so ``docker compose exec yolo-api pytest
   ...`` fails with "executable file not found". README must install
   ``requirements-test.txt`` into the running container first.

Also: F-21's ``OP_INGEST_PRIMARY_CLASS_IDS`` gap -- an unset ingest
class allowlist means a stock detector's *entire* label space (all 80
COCO classes) becomes item proposals, which the walkthrough must warn
about before a reader runs the walker unnarrowed.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / 'README.md').read_text()


def _compose_source_root_mount() -> str:
    with (REPO_ROOT / 'docker-compose.yml').open() as fh:
        compose = yaml.safe_load(fh)
    volumes = compose['services']['yolo-api']['volumes']
    match = next(v for v in volumes if 'OP_SOURCE_ROOT_HOST' in str(v))
    return str(match)


def test_source_root_host_default_and_target_are_what_the_doc_assumes() -> None:
    """Sanity check on the assumption the README tests below rely on."""
    mount = _compose_source_root_mount()
    assert './data/source' in mount
    assert ':/data/source' in mount


def test_readme_sample_walker_sets_op_source_root_host() -> None:
    section = README[README.index('Try it with a public sample') :][:1500]
    assert 'OP_SOURCE_ROOT_HOST=./data/samples' in section, (
        "README's sample-walker instructions must point OP_SOURCE_ROOT_HOST "
        f'at data/samples (the compose mount target is fixed at /data/source, '
        f'which data/samples/coco_va_readme is NOT under by default):\n{section}'
    )


def test_readme_sample_walker_root_is_under_the_source_mount() -> None:
    section = README[README.index('Try it with a public sample') :][:2000]
    assert '--root /data/source/coco_va_readme/images' in section, (
        f"README's ingest_walker.py --root must be a container path under "
        f'/data/source (the real mount target), not a host-relative or '
        f'/app/data/samples path:\n{section}'
    )
    # The old, broken form must be gone.
    assert '--root data/samples/coco_va_readme/images' not in section


def test_readme_mentions_narrowing_ingest_class_ids() -> None:
    section = README[README.index('Try it with a public sample') :][:1500]
    assert 'OP_INGEST_PRIMARY_CLASS_IDS' in section, (
        "README's sample-walker instructions must mention "
        'OP_INGEST_PRIMARY_CLASS_IDS -- otherwise a stock detector proposes '
        f'items for its entire label space (all 80 COCO classes):\n{section}'
    )


def test_requirements_test_txt_exists() -> None:
    """Sanity check: the file README now tells an operator to pip install
    into the container really exists."""
    assert (REPO_ROOT / 'requirements-test.txt').exists()


def test_readme_docker_only_test_path_installs_test_deps_first() -> None:
    section = README[README.index('Docker-only path') :][:600]
    assert 'pip install -r requirements-test.txt' in section, (
        f"F-31: README's Docker-only test path must install test deps into "
        f'the running container first -- the production image ships '
        f'without pytest:\n{section}'
    )
    assert 'already has\nevery test dependency installed' not in README
    assert 'already has every test dependency installed' not in README
