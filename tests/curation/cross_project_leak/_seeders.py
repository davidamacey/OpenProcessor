"""Seeders that give each project its own stored configs, models and failed items."""

from __future__ import annotations

import contextlib
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


def _ingest_profile_body() -> dict[str, Any]:
    return {'primary': {'model': 'sweep-model'}}


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


def _reseed_config(env: Any, slug: str, kind: str, axis: str, body: dict[str, Any]) -> None:
    import asyncio

    asyncio.run(_reseed_config_async(env.records[slug], kind, axis, body))


async def _reseed_config_async(record: Any, kind: str, axis: str, body: dict[str, Any]) -> None:
    """``<slug>-model`` back to a single revision 1 and ``axis`` never activated (the sweep
    bodies send ``expected_active: null``). No service un-activates to "never", so the
    activation row and the document (its revisions cascade) are removed first."""
    from sqlalchemy import delete

    from src.config.project_context import bind_project
    from src.db.models.config import ConfigActivation, ConfigDocument
    from src.services.config_store import documents
    from src.services.config_store.bound_project import bound_project_id
    from src.services.config_store.store import reset_config_stores

    name = f'{record.slug}-model'
    with bind_project(record):
        project_id = await bound_project_id()
        async with documents.tx('reseed the leak environment') as session:
            await session.execute(
                delete(ConfigActivation).where(
                    ConfigActivation.project_id == project_id, ConfigActivation.axis == axis
                )
            )
            await session.execute(
                delete(ConfigDocument).where(
                    ConfigDocument.project_id == project_id,
                    ConfigDocument.kind == kind,
                    ConfigDocument.name == name,
                )
            )
        await documents.save_config(kind=kind, name=name, body=body, expected_revision=None)
    reset_config_stores()


def _clear_activations(env: Any, slug: str, *axes: str) -> None:
    """``slug`` has never activated ``axes`` (no service un-activates to "never", so the
    rows are deleted, as :func:`_reseed_config_async` does)."""
    import asyncio

    asyncio.run(_clear_activations_async(env.records[slug], axes))


async def _clear_activations_async(record: Any, axes: tuple[str, ...]) -> None:
    from sqlalchemy import delete

    from src.config.project_context import bind_project
    from src.db.models.config import ConfigActivation, ConfigActivationEvent
    from src.services.config_store import documents
    from src.services.config_store.bound_project import bound_project_id
    from src.services.config_store.store import reset_config_stores

    with bind_project(record):
        project_id = await bound_project_id()
        async with documents.tx('clear activations in the leak environment') as session:
            # "Never activated" is no activation row and no activation event (list_axis_state).
            for table in (ConfigActivation, ConfigActivationEvent):
                await session.execute(
                    delete(table).where(table.project_id == project_id, table.axis.in_(axes))
                )
    reset_config_stores()


def _activate_detection_profile(env: Any, slug: str, *, detector: str, name: str = 'det') -> None:
    """``slug``'s active detection profile pins revision 1 of ``name``, which names
    ``detector``; saved and activated through the config store services the routes use."""
    import asyncio

    asyncio.run(_activate_detection_profile_async(env.records[slug], detector, name))


async def _activate_detection_profile_async(record: Any, detector: str, name: str) -> None:
    from src.config.project_context import bind_project
    from src.services.config_store import documents
    from src.services.config_store.store import reset_config_stores

    await _clear_activations_async(record, ('detection_profile',))
    with bind_project(record):
        await documents.save_config(
            kind='region_profile',
            name=name,
            body={'detector_model': detector},
            expected_revision=None,
        )
        await documents.activate(
            axis='detection_profile', name=name, revision=1, expected_active=None
        )
    reset_config_stores()


def _stored_prompt_pack(env: Any, slug: str) -> None:
    """(Re-)seed the ``{slug}-model`` prompt_pack at revision 1 with no activation, through
    the config store; run before each mutating route so it is independent of what an
    earlier route left behind."""
    _reseed_config(env, slug, 'prompt_pack', 'prompt_pack', _prompt_pack_body())


def _stored_region_profile(env: Any, slug: str) -> None:
    """(Re-)seed the ``{slug}-model`` region_profile at revision 1 with no activation, through
    the config store; run before each mutating route so it is independent of what an
    earlier route left behind."""
    _reseed_config(env, slug, 'region_profile', 'detection_profile', _region_profile_body())


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
    """The shared endpoint in the deployment-wide registry (the private address
    ``_FAKE_VLM_HOST`` resolves to) and ``slug``'s activation of it, written through
    the config store services the routes use and re-seeded before each VLM
    activation route so every one is independent of what an earlier route left
    behind. The activation's ``previous`` is the per-project ``<slug>-model``
    endpoint, activated first and deleted again."""
    import asyncio

    asyncio.run(_seed_vlm_async(env.records[slug]))


async def _seed_vlm_async(record: Any) -> None:
    from src.config.project_context import bind_project
    from src.db.repos.config_base import RevisionConflictError
    from src.services.config_store import documents
    from src.services.config_store.global_store import reset_global_config_store
    from src.services.config_store.store import get_config_store, reset_config_stores
    from src.services.config_store.vlm_activation import expected_active_for
    from src.services.labeling.vlm_endpoint_body import VlmEndpointBody

    reset_config_stores()
    reset_global_config_store()
    body = VlmEndpointBody(base_url=f'http://{_FAKE_VLM_HOST}/v1', model='leak-test-vlm')
    marker = f'{record.slug}-model'

    # Not ``save_endpoint``: that announces ``vlm.changed`` on the global stream, and a seed
    # must not look like an event of the route under test.
    async def save_config(name: str, endpoint: VlmEndpointBody) -> None:
        await documents.save_config(
            kind='vlm_endpoint',
            name=name,
            body=endpoint.normalized().model_dump(),
            expected_revision=None,
        )

    async def _activate(name: str) -> None:
        store = get_config_store()
        await store.refresh()
        await documents.activate(
            axis='vlm',
            name=name,
            revision=1,
            expected_active=expected_active_for(store.current.active_vlm),
        )

    with bind_project(record):
        store = get_config_store()
        await store.refresh()
        with contextlib.suppress(RevisionConflictError):  # already registered by an earlier seed
            await save_config(_SHARED_VLM, body)
        await save_config(marker, body)
        await _activate(marker)
        await _activate(_SHARED_VLM)
        await documents.delete_config(kind='vlm_endpoint', name=marker, expected_revision=1)
    reset_config_stores()
    reset_global_config_store()


def _prepare_vlm(env: Any, slug: str) -> None:
    _seed_vlm(env, slug)


def _stored_ingest_profile(env: Any, slug: str) -> None:
    """(Re-)seed the ``{slug}-model`` ingest_profile at revision 1 with no activation, through
    the config store; run before each mutating route so it is independent of what an
    earlier route left behind."""
    _reseed_config(env, slug, 'ingest_profile', 'ingest_profile', _ingest_profile_body())


def _stored_open_vocab(env: Any, slug: str) -> None:
    """(Re-)seed the ``{slug}-model`` open_vocab_set at revision 1 with no activation, through
    the config store; run before each mutating route so it is independent of what an
    earlier route left behind."""
    _reseed_config(env, slug, 'open_vocab_set', 'open_vocab', _open_vocab_body())


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
    ('POST', '/ingest_profiles/{name}/clone'): _stored_ingest_profile,
    ('PUT', '/ingest_profiles/{name}'): _stored_ingest_profile,
    ('DELETE', '/ingest_profiles/{name}'): _stored_ingest_profile,
    ('POST', '/ingest_profiles/{name}/activate'): _stored_ingest_profile,
    ('POST', '/open_vocab/{name}/clone'): _stored_open_vocab,
    ('PUT', '/open_vocab/{name}'): _stored_open_vocab,
    ('DELETE', '/open_vocab/{name}'): _stored_open_vocab,
    ('POST', '/open_vocab/{name}/activate'): _stored_open_vocab,
}
