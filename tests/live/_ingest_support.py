"""Shared pieces of the live ingest, search and OCR tests: a throwaway project and public-free pictures.

The harness's fake Triton (``docker/test/fake_triton.py``) answers the fused detector, the PE image
and text encoders and the OCR pipeline with deterministic outputs, so the pictures only need to
differ from each other: solid colour blocks on a plain ground, generated here (no photograph).
"""

from __future__ import annotations

import io
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw

from .conftest import API_PREFIX, VERIFY_DATA_DIR, create_project, request_when_free, wait_until


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# The detector the fake Triton serves (docker/test/fake_triton.py: one box per picture).
DETECTOR_MODEL = 'yolov11_small_trt_end2end'
# The path of the harness data mount inside the API container (docker/test/compose.yml).
CONTAINER_IMAGES_ROOT = '/verify-data/images'


@dataclass(frozen=True)
class IngestProject:
    slug: str
    prefix: str
    images_index: str
    items_index: str

    @property
    def host_dir(self) -> Path:
        return VERIFY_DATA_DIR / 'images' / self.slug

    @property
    def container_dir(self) -> str:
        return f'{CONTAINER_IMAGES_ROOT}/{self.slug}'


def picture(colour: tuple[int, int, int], *, size: tuple[int, int] = (160, 120)) -> bytes:
    """A JPEG: a ``colour`` block on a mid-grey ground. Distinct colours embed apart."""
    img = Image.new('RGB', size, (128, 128, 128))
    w, h = size
    ImageDraw.Draw(img).rectangle((w // 5, h // 5, 4 * w // 5, 4 * h // 5), fill=colour)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return buf.getvalue()


@contextmanager
def ingest_project(api: Any, opensearch: Any, stem: str) -> Iterator[IngestProject]:
    """A fresh active project (a deleted slug is retired for good, so each run takes a new one);
    deleted with its image folder on exit."""
    slug = f'{stem}{uuid.uuid4().hex[:6]}'
    create_project(api, slug, f'{stem} live')
    project = IngestProject(
        slug=slug,
        prefix=f'{API_PREFIX}/projects/{slug}',
        images_index=f'verify_prj_{slug}__images',
        items_index=f'verify_prj_{slug}__items',
    )
    assert wait_until(lambda: opensearch.head(f'/{project.images_index}').status_code == 200)
    # Ingest answers 503 until the project has a detector; the fake Triton serves this one.
    created = api.post(
        f'{project.prefix}/ingest_profiles',
        json={'name': 'harness_detector', 'body': {'primary': {'model': DETECTOR_MODEL}}},
    )
    assert created.status_code == 201, created.text
    activated = api.post(f'{project.prefix}/ingest_profiles/harness_detector/activate', json={})
    assert activated.status_code == 200, activated.text
    project.host_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield project
    finally:
        for path in sorted(project.host_dir.rglob('*'), reverse=True):
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                path.rmdir()
        project.host_dir.rmdir()
        # A force-removed detection worker stays a running job until its liveness lapses; the
        # served project_busy answer says to retry.
        deleted = request_when_free(
            lambda: api.delete(f'{API_PREFIX}/projects/{slug}', params={'confirm': slug}),
            busy='project_busy',
            timeout=90.0,
        )
        assert deleted.status_code == 202, deleted.text


def count(opensearch: Any, index: str) -> int:
    opensearch.post(f'/{index}/_refresh').raise_for_status()
    return int(opensearch.get(f'/{index}/_count').json()['count'])
