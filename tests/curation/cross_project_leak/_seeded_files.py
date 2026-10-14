"""Per-project documents, vectors and files the isolation environment is seeded with."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
from integration.ingest_fakes import jpeg_bytes

from curation.cross_project_leak._constants import CLASS_NAMES, EMBED_DIM


if TYPE_CHECKING:
    from pathlib import Path


def _source_root() -> str:
    """The shared source-image root (``OP_SOURCE_ROOT``, set by the fixture)."""
    return os.environ.get('OP_SOURCE_ROOT', '/images')


def _vec(i: int) -> list[float]:
    """A distinct unit embedding per seeded item (the same across projects,
    so both passes see the same data shapes)."""
    v = np.random.default_rng(i).normal(size=EMBED_DIM)
    return (v / np.linalg.norm(v)).tolist()


def _docs(slug: str) -> dict[str, dict[str, dict[str, Any]]]:
    """One project's seed data, keyed by IndexRole value -> doc id -> doc.

    Beyond a validated item and an unlabeled proposal, it seeds the state
    each write route acts on (so every one of them really writes): a
    human label to undo, a VLM suggestion to dismiss, region edits to
    undo, a region to verify, a high-purity candidate cluster to promote,
    a pending new-class proposal, a residual pool to refit UMAP on, and
    an ingested image whose label file can be imported."""
    from src.config.region_fields import get_region_fields
    from src.services.curation.class_sources import VLM_NEW_CLASS_PENDING_CLASS_SOURCE
    from src.services.curation.edit_history import EDIT_HISTORY_FIELD, EditKind, record_edit
    from src.services.curation.history import record_class_snapshot
    from src.services.curation.ingest_class_sources import classifier_class_sources
    from src.services.curation.region_boxes import RegionBox, boxes_write_fields

    F = get_region_fields()
    image_id = f'{slug}-img-0001'

    def crop(n: int, **fields: Any) -> dict[str, Any]:
        return {
            'crop_id': f'{slug}-item-{n:04d}',
            'image_id': image_id,
            'image_path': f'{slug}/{image_id}.jpg',
            'class_id': 1,
            'class_name': CLASS_NAMES[slug],
            'class_source': 'human',
            'class_validated': True,
            'bbox': [0.1, 0.1, 0.5, 0.5],
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'cluster_id': 1,
            'pe_embedding': _vec(n),
            'region_embedding': _vec(n),
            **fields,
        }

    unlabeled = {
        'class_id': None,
        'class_name': None,
        'class_source': 'proposal',
        'class_validated': False,
    }
    region_before: dict[str, Any] = {F.boxes: []}

    def boxes(state: str) -> dict[str, Any]:
        return boxes_write_fields(
            [RegionBox(box_id='b1', bbox_norm=(0.2, 0.2, 0.3, 0.3), state=state)],
            current_src={},
        )

    items = [
        crop(1),
        # An unlabeled proposal: the batch label/move/region routes refuse
        # to overwrite a human decision, so they act on this one.
        crop(2, **unlabeled),
        crop(
            3,
            class_id_history=record_class_snapshot(
                crop(3, **unlabeled), writer='human:label_crop', restorable=True
            ),
        ),
        crop(
            4,
            class_source='vlm',
            class_validated=False,
            vlm_confidence='high',
            # the detector's own answer: an eligible accuracy-audit candidate
            detector_class_name=f'{slug}_detector_class',
            **boxes('proposed'),
        ),
        *(
            crop(
                n,
                **boxes('accepted'),
                **{
                    EDIT_HISTORY_FIELD: record_edit(
                        region_before, kind=EditKind.REGION, writer='human:set_region'
                    ),
                },
            )
            for n in (5, 6)
        ),
        *(
            crop(
                n,
                class_source=next(iter(sorted(classifier_class_sources()))),
                class_validated=False,
                cluster_id=10001,
            )
            for n in range(7, 11)
        ),
        crop(
            11,
            **{
                **unlabeled,
                'class_source': VLM_NEW_CLASS_PENDING_CLASS_SOURCE,
                'vlm_proposed_class': f'{slug}-proposal',
            },
        ),
        # the residual pool must clear the clustering floor (32) for the UMAP refit to run
        *(crop(n, **unlabeled, cluster_id=None) for n in range(12, 48)),
    ]
    labeled_image = f'{slug}-img-0002'
    return {
        'items': {doc['crop_id']: doc for doc in items},
        'images': {
            image_id: {'image_id': image_id, 'image_path': f'{slug}/{image_id}.jpg'},
            labeled_image: {
                'image_id': labeled_image,
                'image_path': f'{_source_root()}/{slug}/{slug}-lab-0001.jpg',
            },
        },
        'labels_confirmed': {
            f'{slug}-label-0001': {
                'crop_id': f'{slug}-item-0001',
                'class_id': 1,
                'class_name': CLASS_NAMES[slug],
            }
        },
        'classes': {'1': {'class_id': 1, 'class_name': CLASS_NAMES[slug]}},
    }


def _job_dirs(slug: str, tmp_path: Path) -> list[Path]:
    """The per-project job dirs P1 routes through ``project_jobs_dir``."""
    roots = [
        tmp_path / 'jobs' / name for name in ('probe', 'select', 'scores', 'viz', 'region_drain')
    ]
    return [root / 'projects' / slug for root in roots]


def _seed_files(record: Any, tmp_path: Path) -> list[Path]:
    """Seed ``record``'s filesystem state; return the dirs that are its own
    (so a later pass can check they stay unchanged)."""
    from src.config.project_context import bind_project
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    slug = record.slug
    res = record.resources
    res.class_registry_path.parent.mkdir(parents=True, exist_ok=True)
    res.class_registry_path.write_text(
        json.dumps(
            {
                'version': 1,
                'classes': [
                    {'id': 1, 'name': CLASS_NAMES[slug]},
                    {'id': 2, 'name': f'{slug}_second'},
                ],
            }
        ),
        encoding='utf-8',
    )
    state = {
        'job_id': f'{slug}-job-state',
        'status': 'completed',
        'result': {'crop_ids': [f'{slug}-item-0001']},
    }
    # Source images every project may ingest from (the shared source root).
    images = tmp_path / 'images' / slug
    images.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(('img-0001', 'new-0001', 'new-0002', 'lab-0001')):
        (images / f'{slug}-{name}.jpg').write_bytes(jpeg_bytes(i + len(slug)))
    (images / f'{slug}-lab-0001.txt').write_text('1 0.5 0.5 0.4 0.4\n', encoding='utf-8')
    own_dirs = list(_job_dirs(slug, tmp_path))
    for job_dir in own_dirs:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / 'state.json').write_text(json.dumps(state), encoding='utf-8')
    # A finished training run with its checkpoint, in the project's own
    # jobs dir and (until P2 scopes the trainer, plan §5.3) in the shared
    # one the training service still reads.
    checkpoint = tmp_path / 'state' / 'training_runs' / slug / 'weights' / 'best.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'fake checkpoint')
    finished = {
        'job_id': f'{slug}-job-0001',
        'state': 'finished',
        'classes': [CLASS_NAMES[slug]],
        'checkpoint_path': str(checkpoint),
    }
    for jobs_dir in (res.train_jobs_dir, tmp_path / 'jobs' / 'train'):
        jobs_dir.mkdir(parents=True, exist_ok=True)
        (jobs_dir / f'{slug}-job-0001.status.json').write_text(
            json.dumps(finished), encoding='utf-8'
        )
    res.autolabel_dir.mkdir(parents=True, exist_ok=True)
    (res.autolabel_dir / 'state.json').write_text(
        json.dumps({'job_id': f'{slug}-job-autolabel', 'status': 'completed'}), encoding='utf-8'
    )
    with bind_project(record):
        # The same ``trained_at`` in every project: a cache keyed only by
        # it would serve one project's scored crops to another.
        FalsePositiveCentroidStore().save(
            np.eye(1, EMBED_DIM, dtype=np.float32),
            {'trained_at': '2026-01-01T00:00:00Z', 'subids': [f'{slug}-fp-0001']},
        )
    for sub in ('exports', 'uploads'):
        (res.export_root if sub == 'exports' else res.upload_root).mkdir(
            parents=True, exist_ok=True
        )
    return [
        *own_dirs,
        res.project_state_dir,
        res.train_jobs_dir,
        res.autolabel_dir,
        res.class_registry_path.parent,
    ]
