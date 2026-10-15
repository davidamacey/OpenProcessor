"""Seeders that give each project its own stored configs, models and failed items."""

from __future__ import annotations

import json
from typing import Any

from curation.cross_project_leak._constants import CLASS_NAMES
from curation.cross_project_leak._transport import _FAKE_VLM_HOST, LeakEnv
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


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
    from src.config.curation_roles import IndexRole

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
    from src.config.curation_roles import IndexRole

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
    from src.config.curation_roles import IndexRole

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
    from src.config.curation_roles import IndexRole

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
    from src.config.curation_roles import IndexRole
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
    from src.config.curation_roles import IndexRole

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
