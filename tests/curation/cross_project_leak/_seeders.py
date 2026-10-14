"""Seeders that give each project its own items, files and stored documents."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
from integration.ingest_fakes import jpeg_bytes

from curation.cross_project_leak._constants import CLASS_NAMES, EMBED_DIM
from curation.cross_project_leak._transport import _FAKE_VLM_HOST, LeakEnv
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


if TYPE_CHECKING:
    from pathlib import Path


def _prompt_pack_body() -> dict[str, Any]:
    body = GENERIC_ITEM_PACK.to_dict()
    body.pop('name')
    return body


def _region_profile_body() -> dict[str, Any]:
    from dataclasses import asdict

    from src.config import DetectionProfile

    raw = asdict(DetectionProfile(name='p'))
    raw.pop('name')
    for key, value in raw.items():
        if isinstance(value, frozenset):
            raw[key] = sorted(value)
        elif isinstance(value, tuple):
            raw[key] = list(value)
    # Text-free, segmenter-only: the leak sweep's Triton is dead by design
    # (network disabled), so a detector leg or an OCR-needing text_reader
    # would always 422 detector_model_not_found/ocr_model_not_found here.
    raw['detector_model'] = ''
    raw['text_reader'] = 'none'
    raw['segmenter_text_prompt'] = 'test region'
    raw['display_name'] = 'Regions'
    raw['display_name_singular'] = 'Region'
    return raw


def _open_vocab_body() -> dict[str, Any]:
    return {'display_name': 'Sweep', 'targets': [{'prompt': 'test object', 'class_name': 'sweep'}]}


def _running_job(job_dir: str) -> Any:
    """A cancel acts on a running job: mark ``slug``'s one running (a
    fresh state, no heartbeat yet) right before its cancel is called."""

    def prepare(env: LeakEnv, slug: str) -> None:
        state = env.root / 'jobs' / job_dir / 'projects' / slug / 'state.json'
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(
            json.dumps({'job_id': f'{slug}-job-running', 'status': 'running'}), encoding='utf-8'
        )
        (state.parent / 'heartbeat').unlink(missing_ok=True)

    return prepare


def _promoted_model(env: LeakEnv, slug: str) -> None:
    """A model ``slug`` promoted into the shared Triton repo (seeded fresh
    before each route that acts on it: DELETE removes it)."""
    model_dir = env.root / 'models' / f'{slug}__model'
    (model_dir / '1').mkdir(parents=True, exist_ok=True)
    (model_dir / '1' / 'model.plan').write_bytes(b'fake plan')
    (model_dir / 'labels.txt').write_text(f'{CLASS_NAMES[slug]}\n', encoding='utf-8')
    (model_dir / 'promote.json').write_text(
        json.dumps(
            {
                'project': slug,
                'shared': False,
                'sharing_revision': 1,
                'classes': [{'model_id': 0, 'name': CLASS_NAMES[slug]}],
            }
        ),
        encoding='utf-8',
    )


def _region_box_seeded(env: LeakEnv, slug: str) -> None:
    """A box ``b1`` on the item, so the per-box PATCH/batch_box_state
    routes have a real box_id to act on. Writes directly into the
    fake transport's store (not through the app's own PUT
    /crops/{crop_id}/regions): another sweep call earlier in the same
    run may already have bumped this item's region_box_seq, which would
    make a fresh app-level write mint ``b2``/``b3``/... instead of the
    fixed ``b1`` this route's path param names -- writing the doc
    directly keeps the seeded id deterministic regardless of sweep
    order."""
    from src.config import get_region_fields
    from src.config.curation import IndexRole

    F = get_region_fields()
    index = env.records[slug].resources.indexes[IndexRole.ITEMS]
    doc_id = f'{slug}-item-0001'
    doc = env.transport.store.setdefault(index, {}).setdefault(doc_id, {})
    doc[F.boxes] = [
        {
            'box_id': 'b1',
            'bbox_norm': [0.1, 0.1, 0.4, 0.4],
            'state': 'accepted',
            'score': 1.0,
            'detector': 'human',
            'source': 'human',
        }
    ]
    doc[F.box_seq] = max(int(doc.get(F.box_seq) or 0), 1)
    doc[F.count] = 1
    doc[F.rejected_count] = 0


def _stored_prompt_pack(env: Any, slug: str) -> None:
    """(Re-)seed ``pack:<slug>-model`` at a known revision 1, with no
    activation recorded, directly in the fake transport's store --
    mirrors ``_region_box_seeded``. Run fresh immediately before EACH
    mutating ``/prompt_packs/{name}...`` route, so every one of them is
    independent of what an earlier route in the same pass left behind
    (revision-based OCC here is enforced entirely by comparing the
    stored ``revision`` field, never real OpenSearch ``if_seq_no``, so
    resetting that field is enough)."""
    from src.config.curation import IndexRole

    index = env.records[slug].resources.indexes[IndexRole.CONFIGS]
    name = f'{slug}-model'
    now = '2026-01-01T00:00:00+00:00'
    body = _prompt_pack_body()
    docs = env.transport.store.setdefault(index, {})
    doc = {
        'doc_type': 'config',
        'kind': 'prompt_pack',
        'name': name,
        'revision': 1,
        'body': body,
        'description': '',
        'created_at': now,
        'updated_at': now,
        'updated_by': None,
        'cloned_from': None,
    }
    docs[f'pack:{name}'] = doc
    docs[f'pack:{name}@1'] = {**doc, 'doc_type': 'revision'}
    meta = docs.setdefault('meta:config_revision', {'doc_type': 'meta', 'config_revision': 0})
    meta['config_revision'] = int(meta.get('config_revision', 0)) + 1
    # No leftover activation from an earlier route in this same pass --
    # 'expected_active: None' in route_bodies must match reality.
    docs.pop('activation:prompt_pack', None)
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()


def _stored_region_profile(env: Any, slug: str) -> None:
    """(Re-)seed ``profile:<slug>-model`` at a known revision 1, with no
    activation recorded -- mirrors ``_stored_prompt_pack``."""
    from src.config.curation import IndexRole

    index = env.records[slug].resources.indexes[IndexRole.CONFIGS]
    name = f'{slug}-model'
    now = '2026-01-01T00:00:00+00:00'
    body = _region_profile_body()
    docs = env.transport.store.setdefault(index, {})
    doc = {
        'doc_type': 'config',
        'kind': 'region_profile',
        'name': name,
        'revision': 1,
        'body': body,
        'description': '',
        'created_at': now,
        'updated_at': now,
        'updated_by': None,
        'cloned_from': None,
    }
    docs[f'profile:{name}'] = doc
    docs[f'profile:{name}@1'] = {**doc, 'doc_type': 'revision'}
    meta = docs.setdefault('meta:config_revision', {'doc_type': 'meta', 'config_revision': 0})
    meta['config_revision'] = int(meta.get('config_revision', 0)) + 1
    docs.pop('activation:detection_profile', None)
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()


def _region_failed_item(env: LeakEnv, slug: str) -> None:
    """``item-0002`` as an unlocked, machine-failed region (a rejected
    machine box, not validated): earlier sweep calls may have validated it,
    and a reprocess skips a validated set, so it must be reset right before
    each reprocess route that acts on it."""
    from src.config import get_region_fields
    from src.config.curation import IndexRole

    F = get_region_fields()
    index = env.records[slug].resources.indexes[IndexRole.ITEMS]
    doc = env.transport.store.setdefault(index, {}).setdefault(f'{slug}-item-0002', {})
    doc[F.status] = 'detection_failed'
    doc[F.validated] = False
    doc[F.boxes] = [
        {
            'box_id': 'b1',
            'bbox_norm': [0.1, 0.1, 0.4, 0.4],
            'state': 'rejected',
            'score': 0.3,
            'detector': 'leak_detector',
            'source': 'detector',
        }
    ]
    doc[F.count] = 0
    doc[F.rejected_count] = 1


# The VLM endpoint registry is deployment-wide (one endpoint both projects can
# use); what is per project is which one it activated. The seeded activation's
# ``previous`` is the per-project marker: an endpoint this project ran before
# and has since deleted, so it must only ever appear in that project's own
# ``/vlm/endpoints/active`` answer.
_SHARED_VLM = 'shared-vlm'
_VLM_ACTIVE = {'name': _SHARED_VLM, 'revision': 1}
_VLM_KEYS = frozenset(
    {
        ('POST', '/vlm/endpoints/{name}/activate'),
        ('POST', '/vlm/endpoints/active/rollback'),
        ('POST', '/vlm/endpoints/deactivate'),
    }
)
NAME_FOR: dict[tuple[str, str], str] = {
    ('POST', '/vlm/endpoints/{name}/activate'): _SHARED_VLM,
}


def _seed_vlm(env: Any, slug: str) -> None:
    """The shared endpoint in the global registry (revision 1, the private
    address ``_FAKE_VLM_HOST`` resolves to) and ``slug``'s activation of it,
    re-seeded before each VLM activation route so every one is independent
    of what an earlier route left behind."""
    from src.config.curation import IndexRole
    from src.services.config_store.global_store import global_configs_index
    from src.services.config_store.store import reset_config_stores
    from src.services.config_store.vlm_snapshot import reset_vlm_revision_cache
    from src.services.labeling.vlm_endpoint_body import VlmEndpointBody

    now = '2026-01-01T00:00:00+00:00'
    endpoint = {
        'doc_type': 'config',
        'kind': 'vlm_endpoint',
        'name': _SHARED_VLM,
        'revision': 1,
        'body': VlmEndpointBody(base_url=f'http://{_FAKE_VLM_HOST}/v1', model='leak-test-vlm')
        .normalized()
        .model_dump(),
        'description': '',
        'created_at': now,
        'updated_at': now,
        'updated_by': None,
        'cloned_from': None,
    }
    registry = env.transport.store.setdefault(global_configs_index(), {})
    registry[f'vlm:{_SHARED_VLM}'] = endpoint
    registry[f'vlm:{_SHARED_VLM}@1'] = {**endpoint, 'doc_type': 'revision'}
    meta = registry.setdefault('meta:config_revision', {'doc_type': 'meta', 'config_revision': 0})
    meta['config_revision'] = int(meta.get('config_revision', 0)) + 1
    index = env.records[slug].resources.indexes[IndexRole.CONFIGS]
    docs = env.transport.store.setdefault(index, {})
    docs['activation:vlm'] = {
        'doc_type': 'activation',
        'axis': 'vlm',
        **_VLM_ACTIVE,
        'activated_at': now,
        'previous': {'name': f'{slug}-model', 'revision': 1},
        'acked_refs': {},
        'external_ack_at': None,
    }
    project_meta = docs.setdefault(
        'meta:config_revision', {'doc_type': 'meta', 'config_revision': 0}
    )
    project_meta['config_revision'] = int(project_meta.get('config_revision', 0)) + 1
    reset_config_stores()
    reset_vlm_revision_cache()


def _prepare_vlm(env: Any, slug: str) -> None:
    _seed_vlm(env, slug)


def _stored_open_vocab(env: Any, slug: str) -> None:
    """(Re-)seed ``ovset:<slug>-model`` at a known revision 1, with no
    activation recorded -- mirrors ``_stored_region_profile``."""
    from src.config.curation import IndexRole

    index = env.records[slug].resources.indexes[IndexRole.CONFIGS]
    name = f'{slug}-model'
    now = '2026-01-01T00:00:00+00:00'
    docs = env.transport.store.setdefault(index, {})
    doc = {
        'doc_type': 'config',
        'kind': 'open_vocab_set',
        'name': name,
        'revision': 1,
        'body': _open_vocab_body(),
        'description': '',
        'created_at': now,
        'updated_at': now,
        'updated_by': None,
        'cloned_from': None,
    }
    docs[f'ovset:{name}'] = doc
    docs[f'ovset:{name}@1'] = {**doc, 'doc_type': 'revision'}
    docs.pop('activation:open_vocab', None)
    meta = docs.setdefault('meta:config_revision', {'doc_type': 'meta', 'config_revision': 0})
    meta['config_revision'] = int(meta.get('config_revision', 0)) + 1


def _export_with_val_split(env: Any, slug: str) -> None:
    """/train/start refuses an empty val split even with force; the seeded
    export has val=0, so give its manifest a val image."""
    import json

    export_root = env.records[slug].resources.export_root
    for manifest in export_root.glob('*/manifest.json'):
        data = json.loads(manifest.read_text())
        counts = data.get('split_counts')
        if isinstance(counts, dict):
            counts['val'] = max(int(counts.get('val') or 0), 1)
            manifest.write_text(json.dumps(data))


PREPARE: dict[tuple[str, str], Any] = {
    **dict.fromkeys(
        [('POST', '/train/start'), ('POST', '/train/start_campaign')], _export_with_val_split
    ),
    **dict.fromkeys(_VLM_KEYS, _prepare_vlm),
    ('POST', '/reprocess'): _region_failed_item,
    ('POST', '/crops/{crop_id}/reprocess'): _region_failed_item,
    ('PUT', '/models/{model_name}/sharing'): _promoted_model,
    ('DELETE', '/models/{model_name}'): _promoted_model,
    ('PATCH', '/crops/{crop_id}/regions/{box_id}'): _region_box_seeded,
    ('POST', '/regions/batch_box_state'): _region_box_seeded,
    ('POST', '/probe/cancel'): _running_job('probe'),
    ('POST', '/scores/cancel'): _running_job('scores'),
    ('POST', '/select/cancel'): _running_job('select'),
    ('POST', '/viz/projection/cancel'): _running_job('viz'),
    ('POST', '/prompt_packs/{name}/clone'): _stored_prompt_pack,
    ('PUT', '/prompt_packs/{name}'): _stored_prompt_pack,
    ('DELETE', '/prompt_packs/{name}'): _stored_prompt_pack,
    ('POST', '/prompt_packs/{name}/activate'): _stored_prompt_pack,
    ('POST', '/region_profiles/{name}/clone'): _stored_region_profile,
    ('PUT', '/region_profiles/{name}'): _stored_region_profile,
    ('DELETE', '/region_profiles/{name}'): _stored_region_profile,
    ('POST', '/region_profiles/{name}/activate'): _stored_region_profile,
    ('POST', '/open_vocab/{name}/clone'): _stored_open_vocab,
    ('PUT', '/open_vocab/{name}'): _stored_open_vocab,
    ('DELETE', '/open_vocab/{name}'): _stored_open_vocab,
    ('POST', '/open_vocab/{name}/activate'): _stored_open_vocab,
}


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
