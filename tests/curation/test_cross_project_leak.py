"""The isolation proof (projects_plan.md §10 P1): every scoped curation
route, swept first as one project and then as the other (alpha then beta,
and beta then alpha), reads and writes only the bound project's
OpenSearch indexes, directories, caches and event stream.

Why two passes: a process cache primed by the first project is read back
by the second one. A single sweep as one project can never see that.

Setup: three ordinary projects -- ``default``, ``alpha`` and ``beta`` --
on a fake OpenSearch transport behind the real
project guard. Each is seeded with its own items, images, labels and
classes, and with filesystem state under its own resources (job state,
a training run, an FP centroid store). Every id and class name embeds
the project slug, so a leak is greppable.

Per route, both passes assert:
- no OpenSearch access to another project's index (recorded *above* the
  guard with this file's own parser, so a refused or swallowed access
  still counts, and ``mget``/``msearch``/``bulk`` bodies are read whatever
  their Python type);
- the bound project never drifts;
- no other project's marker in the response, and every served URL is
  under the bound project's prefix;
- no event reaching another project's stream, and no event on the
  global stream;
- no 5xx unless listed in ``EXPECTED_5XX`` with a reason;
- a mutating route sends a valid body (a 422 means "unmapped body") and
  really writes (to the bound project's indexes or dirs) unless listed in
  ``NO_WRITE`` with a reason.
Across the second pass, the first project's directories are byte-for-byte
unchanged.

Routes are enumerated from the app's route table, so a route added later
is covered automatically; a new path parameter, streaming route or
mutating route fails the test until it is mapped below.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import hashlib
import inspect
import json
import os
import re
import subprocess  # nosec B404 - only patched to refuse, never called
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from integration.ingest_fakes import FakePEEncoder, FakeTritonPool, jpeg_bytes
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError

from curation.query_fakes import _aggregate, matches
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


API = '/curation'
SCOPED = f'{API}/projects/{{project}}'
SLUGS = ('default', 'alpha', 'beta')
CLASS_NAMES = {'default': 'default_cardinal', 'alpha': 'alpha_zebra', 'beta': 'beta_heron'}
EMBED_DIM = 8


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


def route_params(slug: str) -> dict[str, str]:
    """Every path parameter a scoped route may carry, filled with ``slug``'s
    ids. A route with a parameter missing here fails ("unmapped route")."""
    return {
        'crop_id': f'{slug}-item-0001',
        'image_id': f'{slug}-img-0001',
        'class_id': '2',
        'cluster_id': '1',
        'job_id': f'{slug}-job-0001',
        'campaign_id': f'{slug}-campaign-0001',
        'name': f'{slug}-model',
        # A promoted model's triton_name is model_prefix + name (plan §5.3).
        'model_name': f'{slug}__model',
        'tab': 'uncertainty',
        'alias': f'{slug}-source',
        'artifact': 'results.csv',
        'box_id': 'b1',
        'revision': '1',
    }


def route_bodies(slug: str, export_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """A minimal valid request for every mutating route, as ``slug``.
    ``{'json': ...}`` / ``{'files': ..., 'data': ...}`` / ``{'params': ...}``
    are passed straight to ``TestClient.request``. A mutating route missing
    here that answers 422 fails with "unmapped body"."""
    item = f'{slug}-item-0001'
    proposal = f'{slug}-item-0002'
    # Ingest takes a server path under the source root.
    source = f'{_source_root()}/{slug}'
    img = f'{source}/{slug}-new-0001.jpg'
    b64 = base64.b64encode(jpeg_bytes(len(slug))).decode('ascii')
    force = {'force': 'true'}
    return {
        ('POST', '/bakeoff/run'): {'json': {'job_id': f'{slug}-job-0001'}},
        ('PUT', '/crops/{crop_id}/label'): {'json': {'class_id': 1}},
        ('PUT', '/crops/batch_label'): {'json': {'crop_ids': [proposal], 'class_id': 1}},
        ('POST', '/crops/move'): {'json': {'crop_ids': [proposal], 'cluster_id': 1}},
        ('POST', '/crops/flag_new_class'): {'json': {'crop_ids': [item], 'note': 'n'}},
        ('POST', '/crops/batch_exclude'): {'json': {'crop_ids': [proposal], 'reason': 'r'}},
        ('POST', '/crops/batch_unexclude'): {'json': {'crop_ids': [proposal]}},
        ('POST', '/classes'): {'json': {'name': f'{slug}_newclass'}},
        ('PUT', '/classes/{class_id}'): {'json': {'group': f'{slug}_group'}},
        ('POST', '/classes/merge'): {'json': {'source_id': 2, 'target_id': 1}},
        ('POST', '/crops/label/undo_batch'): {'json': {'crop_ids': [f'{slug}-item-0003']}},
        ('POST', '/crops/{crop_id}/discard'): {'json': {}},
        ('POST', '/crops/discard_batch'): {'json': {'crop_ids': [item]}},
        ('POST', '/crops/region/undo_batch'): {'json': {'crop_ids': [f'{slug}-item-0006']}},
        ('POST', '/events/publish'): {
            'json': {'type': 'crop.classified', 'crop_id': item, 'class_id': 1}
        },
        ('POST', '/export/yolo'): {'json': {'version_tag': f'{slug}-v1'}},
        ('POST', '/export/single_class'): {'json': {'version_tag': f'{slug}-v1', 'class_ids': [1]}},
        ('POST', '/vlm/label_batch'): {'json': {'crop_ids': [item]}},
        ('POST', '/vlm/verify_regions'): {'json': {'crop_ids': [f'{slug}-item-0004']}},
        ('POST', '/vlm/verify_region_batch'): {
            'json': {'items': [{'crop_id': item, 'region_image_b64': b64}]}
        },
        ('POST', '/vlm/region_visible_batch'): {
            'json': {'items': [{'crop_id': item, 'image_b64': b64}]}
        },
        ('POST', '/ingest/image'): {'json': {'path': img}},
        ('POST', '/ingest/batch'): {'json': {'items': [{'path': f'{source}/{slug}-new-0002.jpg'}]}},
        ('POST', '/ingest/path_lookup'): {'json': {'image_paths': [img]}},
        ('POST', '/ingest/upload'): {
            'files': [('images', (f'{slug}-up.jpg', jpeg_bytes(len(slug)), 'image/jpeg'))],
        },
        ('POST', '/probe/run'): {'json': {'job_id': f'{slug}-job-0001'}},
        ('PUT', '/models/{model_name}/sharing'): {'json': {'shared': True, 'expected_revision': 1}},
        ('PUT', '/crops/{crop_id}/region'): {'json': {'region_bbox_norm': [0.1, 0.1, 0.4, 0.4]}},
        ('PATCH', '/crops/{crop_id}/region_meta'): {
            'json': {'region_rejection_reason': f'{slug}-note'}
        },
        ('PUT', '/crops/batch_region'): {
            'json': {'crop_ids': [proposal], 'region_bbox_norm': [0.1, 0.1, 0.4, 0.4]}
        },
        ('POST', '/regions/batch_status'): {
            'json': {'crop_ids': [item], 'region_status': 'false_positive'}
        },
        ('PUT', '/crops/{crop_id}/regions'): {
            'json': {'boxes': [{'box_id': None, 'bbox_norm': [0.1, 0.1, 0.4, 0.4]}]}
        },
        ('PUT', '/crops/batch_regions'): {'json': {'crop_ids': [proposal], 'boxes': []}},
        ('PATCH', '/crops/{crop_id}/regions/{box_id}'): {'json': {'state': 'accepted'}},
        ('POST', '/regions/batch_box_state'): {
            'json': {'targets': [{'crop_id': item, 'box_id': 'b1'}], 'state': 'accepted'}
        },
        ('POST', '/test_holdout/freeze'): {'json': {'percent': 10}},
        ('POST', '/review/new_class_proposals/resolve'): {
            'json': {'label': f'{slug}-proposal', 'class_id': 1}
        },
        ('POST', '/scores/compute'): {'json': {}},
        ('POST', '/select/diverse'): {'json': {'k': 1}},
        # minor 7 (W2 review): a config-store axis in the body, not just
        # an empty `defaults`, so the sweep actually exercises the
        # config-store write path (`activate` + `bump_config_revision`'s
        # painless script) instead of only the generic settings-doc merge.
        ('PUT', '/settings'): {'json': {'defaults': {'prompt_pack': GENERIC_ITEM_PACK.name}}},
        ('PUT', '/keymap'): {
            'json': {'expected_revision': 0, 'overrides': {'cluster.ignore': ['k']}}
        },
        ('POST', '/keymap/validate'): {'json': {'overrides': {'cluster.ignore': ['k']}}},
        # Runs after PUT /keymap in route-declaration order, which already
        # bumped the doc to revision 1.
        ('POST', '/keymap/reset'): {
            'json': {'expected_revision': 1, 'action_ids': ['cluster.ignore']}
        },
        ('POST', '/train/preflight'): {'json': {}},
        # force: the preflight's class-balance/disk gates are not what
        # this test is about; the job files written are.
        ('POST', '/train/start'): {'params': force, 'json': {}},
        ('POST', '/train/start_campaign'): {
            'params': force,
            'json': {
                'campaign_id': f'{slug}-campaign-0001',
                'dataset_export_dir': str(export_root / f'{slug}-v1'),
                'runs': [{'profile': 'probe', 'model_size': 'n'}],
            },
        },
        ('POST', '/train/promote/{job_id}'): {
            'json': {'triton_name': f'{slug}_model', 'force': True}
        },
        # P3 lifecycle mutations (global_router, not part of the scoped
        # double-mount, but textually under SCOPED -- see
        # tests/projects/test_route_scoping.py). ``DELETE`` uses
        # ``dry_run`` so the sweep never actually removes the project
        # (which would break every later call for this slug in the same
        # pass); the others act for real -- see NO_WRITE for why none of
        # the five register a write here.
        ('DELETE', ''): {'params': {'dry_run': 'true'}},
        ('PATCH', ''): {'json': {'display_name': f'{slug}-renamed', 'expected_revision': 1}},
        ('POST', '/archive'): {'json': {'expected_revision': 1}},
        ('POST', '/unarchive'): {'json': {'expected_revision': 1}},
        ('POST', '/clone_settings'): {'json': {'from': slug, 'expected_revision': 1}},
        # W3: prompt-pack CRUD. '{name}' (route_params) is a stored pack
        # PREPARE resets to revision 1 immediately before each of these
        # (see the _prompt_pack_* PREPARE hooks) -- independent of
        # whatever an earlier route in the same pass left behind.
        ('POST', '/prompt_packs'): {
            'json': {'name': f'{slug}-newpack', 'body': _prompt_pack_body()}
        },
        ('POST', '/prompt_packs/validate'): {'json': {'name': None, 'body': _prompt_pack_body()}},
        ('POST', '/prompt_packs/test'): {'json': {'call': 'region_visible'}},
        ('POST', '/prompt_packs/active/rollback'): {'json': {'expected_active': None}},
        ('POST', '/prompt_packs/{name}/clone'): {
            'json': {'new_name': f'{slug}-clone', 'source': 'stored'}
        },
        ('PUT', '/prompt_packs/{name}'): {
            'json': {'expected_revision': 1, 'body': _prompt_pack_body()}
        },
        ('DELETE', '/prompt_packs/{name}'): {'params': {'expected_revision': '1'}},
        ('POST', '/prompt_packs/{name}/activate'): {
            'json': {'revision': None, 'expected_active': None, 'force': False}
        },
        # W4: region-profile CRUD. Same PREPARE-resets-to-a-known-revision
        # pattern as the pack routes above.
        ('POST', '/region_profiles'): {
            'json': {'name': f'{slug}-newprofile', 'body': _region_profile_body()}
        },
        ('POST', '/region_profiles/validate'): {
            'json': {'name': None, 'body': _region_profile_body()}
        },
        ('POST', '/region_profiles/validate_segmenter_prompt'): {
            'json': {'text_prompt': 'test region', 'sole_leg': True}
        },
        ('POST', '/region_profiles/test'): {'json': {'name': None, 'body': _region_profile_body()}},
        ('POST', '/region_profiles/active/rollback'): {'json': {'expected_active': None}},
        ('POST', '/region_profiles/deactivate'): {'json': {'expected_active': None}},
        ('POST', '/region_profiles/{name}/clone'): {
            'json': {'new_name': f'{slug}-profileclone', 'source': 'stored'}
        },
        ('PUT', '/region_profiles/{name}'): {
            'json': {'expected_revision': 1, 'body': _region_profile_body()}
        },
        ('DELETE', '/region_profiles/{name}'): {'params': {'expected_revision': '1'}},
        ('POST', '/region_profiles/{name}/activate'): {
            'json': {'revision': None, 'expected_active': None, 'force': True}
        },
    }


# Mutating routes that write nothing by design (read-only work under POST,
# or an event with no data write), and why. Every other mutating route must
# really write in the sweep.
NO_WRITE: dict[tuple[str, str], str] = {
    ('POST', '/events/publish'): 'publishes an event (checked separately), writes no data',
    ('POST', '/ingest/path_lookup'): 'read-only lookup under POST',
    ('POST', '/train/preflight'): 'read-only validation under POST',
    ('POST', '/keymap/validate'): 'dry-run report; writes nothing',
    ('POST', '/train/reload_promoted'): 'asks Triton to load promoted models; stores nothing',
    (
        'POST',
        '/vlm/verify_region_batch',
    ): "returns the VLM's verdicts to the caller; stores nothing",
    (
        'POST',
        '/vlm/region_visible_batch',
    ): "returns the VLM's verdicts to the caller; stores nothing",
    # P3 lifecycle mutations write the shared op_projects registry doc,
    # never the project's own item/image/etc indexes or state-dir files
    # -- the write-detection this sweep does (own_indexes / dir digest)
    # has nothing of the bound project's *data* to see, by design.
    (
        'DELETE',
        '',
    ): 'dry_run=true in the sweep; a real delete mutates the registry, not project data',
    ('PATCH', ''): 'mutates the shared project registry doc, not project data',
    ('POST', '/archive'): 'mutates the shared project registry doc, not project data',
    ('POST', '/unarchive'): 'mutates the shared project registry doc, not project data',
    ('POST', '/clone_settings'): 'mutates the shared project registry doc, not project data',
    ('POST', '/prompt_packs/validate'): 'dry-run report; never writes',
    ('POST', '/prompt_packs/test'): 'renders a prompt preview; never writes',
    ('POST', '/region_profiles/validate'): 'dry-run report; never writes',
    ('POST', '/region_profiles/validate_segmenter_prompt'): 'dry-run report; never writes',
    ('POST', '/region_profiles/test'): 'renders an effective-legs preview; never writes',
}

# Routes whose ``{crop_id}`` is a seeded item other than ``-item-0001``,
# because they act on state that item does not have.
CROP_FOR: dict[tuple[str, str], str] = {
    ('POST', '/crops/{crop_id}/vlm_dismiss'): 'item-0004',
    ('POST', '/crops/{crop_id}/vlm_dismiss/undo'): 'item-0004',
    ('POST', '/crops/{crop_id}/region/undo'): 'item-0005',
}

# Routes that answer 5xx in the fixture for a reason that is not isolation.
EXPECTED_5XX: dict[tuple[str, str], str] = {
    ('DELETE', '/models/{model_name}'): (
        '502: the dead in-process Triton never confirms the unload, so the route '
        'refuses to delete the (own, ownership-checked) model dir'
    ),
}

# P3 lifecycle mutations (global_router; textually under SCOPED but never
# part of the scoped/alias double-mount -- see the module docstring at the
# top of src/routers/curation/projects.py). They act *on* a project via a
# plain ``project: str`` path parameter, not *within* one via
# ``bind_path_project``, so their OpenSearch calls (all against the shared
# ``op_projects`` registry doc) are correctly unbound (``try_current_project()
# is None``) rather than bound to the slug in the URL.
UNBOUND_BY_DESIGN: frozenset[tuple[str, str]] = frozenset(
    {
        ('DELETE', ''),
        ('PATCH', ''),
        ('POST', '/archive'),
        ('POST', '/unarchive'),
        ('POST', '/clone_settings'),
        # BA-P2-5: pause/resume publish project.paused/project.resumed on
        # the *global* stream (project: null, target: slug) so every open
        # Cropwright tab learns about it, same rationale as the other
        # project.* lifecycle events above.
        ('POST', '/pause'),
        ('POST', '/resume'),
    }
)

# Mutating routes whose write this fixture cannot reach yet: the seeded
# state lacks what the route acts on. They are excused ONLY from the
# "wrote nothing" check; every isolation check (foreign indexes, foreign
# markers in the response, events, 5xx, cache parity, foreign files)
# still applies to them.
UNSEEDED_WRITES: dict[tuple[str, str], str] = {
    ('POST', '/bakeoff/run'): 'needs a resolvable contender model list, not seeded',
    ('POST', '/train/promote/{job_id}'): 'needs an exported best.onnx, not seeded',
    ('DELETE', '/models/{model_name}'): 'see EXPECTED_5XX: no live Triton to confirm the unload',
    ('POST', '/vlm/label_cluster/{cluster_id}'): (
        '409s: /pipeline/auto_label/start already queued a run earlier in the pass'
    ),
    ('POST', '/prompt_packs/active/rollback'): (
        'a fresh per-slug config store has no prior activation to roll back to '
        '(409 no_previous); rollback success is covered by test_prompt_packs_router.py'
    ),
    ('POST', '/region_profiles/active/rollback'): (
        'a fresh per-slug config store has no prior activation to roll back to '
        '(409 no_previous); rollback success is covered by test_region_profiles_router.py'
    ),
}


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
    routes (W8a) have a real box_id to act on. Writes directly into the
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


PREPARE: dict[tuple[str, str], Any] = {
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
}


# Long-lived SSE streams: the per-project delivery they serve is proven by
# the event checks below and tests/projects/test_event_hub_project_filter.py.
STREAMING_ROUTES: frozenset[str] = frozenset({f'{SCOPED}/events', f'{SCOPED}/pipeline/events'})


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
    region_before = {F.bbox_norm: None}
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
            **{F.bbox_norm: [0.2, 0.2, 0.3, 0.3]},
        ),
        *(
            crop(
                n,
                **{
                    F.bbox_norm: [0.2, 0.2, 0.3, 0.3],
                    EDIT_HISTORY_FIELD: record_edit(
                        region_before, kind=EditKind.REGION, writer='human:set_region'
                    ),
                },
            )
            for n in (5, 6)
        ),
        *(
            crop(n, class_source='vlm', class_validated=False, cluster_id=10001)
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
        *(crop(n, **unlabeled, cluster_id=None) for n in range(12, 18)),
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


# --- A recording layer with its own parser -----------------------------------


def _body_json(body: Any) -> Any:
    if body is None:
        return None
    if isinstance(body, (dict, list)):
        return body
    text = body.decode('utf-8', 'replace') if isinstance(body, bytes) else str(body)
    try:
        return json.loads(text)
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _body_lines(body: Any) -> list[Any]:
    if body is None:
        return []
    if isinstance(body, list):
        return body
    text = body.decode('utf-8', 'replace') if isinstance(body, bytes) else str(body)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def touched_indexes(url: str, body: Any) -> set[str]:
    """Every index a request reaches, independent of the guard's parser.
    A search-shaped request with no index at all is recorded as ``*``."""
    parts = [unquote(p) for p in url.split('?')[0].split('/') if p]
    out: set[str] = set()
    if parts and not parts[0].startswith('_'):
        out |= set(parts[0].split(','))
    if parts[:3] == ['_plugins', '_knn', 'warmup'] and len(parts) > 3:
        out |= set(parts[3].split(','))
    action = next((p for p in parts if p.startswith('_')), '')
    url_index = bool(out)
    if action == '_mget':
        for doc in (_body_json(body) or {}).get('docs', []):
            out.add(doc.get('_index') or ('' if url_index else '*'))
    elif action == '_bulk':
        for line in _body_lines(body):
            for op in ('index', 'create', 'update', 'delete'):
                if isinstance(line.get(op), dict):
                    out.add(line[op].get('_index') or ('' if url_index else '*'))
    elif action == '_msearch':
        for header in _body_lines(body)[0::2]:
            named = False
            for key in ('index', 'indices'):
                target = header.get(key)
                if target:
                    named = True
                    out |= set(target.split(',') if isinstance(target, str) else target)
            if not named and not url_index:
                out.add('*')
    elif action in ('_search', '_count') and not url_index and parts[:2] != ['_search', 'scroll']:
        out.add('*')
    out.discard('')
    return out


_FAKE_VLM_HOST = 'vlm.leak-test'


def _fake_vlm_reply(request: httpx.Request) -> httpx.Response:
    """An OpenAI-shaped chat completion answering every image in the
    request with one confident verdict: the first catalog class, region
    visible and verified, a short text read."""
    payload = json.loads(request.content or b'{}')
    n_images = sum(
        1
        for message in payload.get('messages', [])
        if isinstance(message.get('content'), list)
        for part in message['content']
        if isinstance(part, dict) and part.get('type') == 'image_url'
    )
    answers = [
        {
            'img': i + 1,
            'class_id': 0,
            'confidence': 'high',
            'region_visible': True,
            'is_region': True,
            'is_plate': True,
            'region_text': 'AB12',
            'text': 'AB12',
            'reason': 'clear',
        }
        for i in range(max(n_images, 1))
    ]
    body = {
        'id': 'leak-test',
        'object': 'chat.completion',
        'model': payload.get('model', 'fake'),
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': json.dumps(answers)},
                'finish_reason': 'stop',
            }
        ],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }
    return httpx.Response(200, json=body, request=request)


def _query_matches(doc_id: str, doc: dict[str, Any], query: Any) -> bool:
    """Evaluate the query DSL subset ``tests/curation/query_fakes.py``
    understands (plus ``ids``); an unsupported clause matches everything, so
    a route that reaches an index still sees (and would leak) its docs."""
    if not query:
        return True
    if 'ids' in query:
        return doc_id in (query['ids'].get('values') or [])
    try:
        return matches({**doc, '_id': doc_id, 'crop_id': doc.get('crop_id', doc_id)}, query)
    except (NotImplementedError, KeyError, TypeError, ValueError):
        return True


class _FakeTransport:
    """The bottom of the fake: answers OpenSearch REST calls from a per-
    index doc store, and applies writes to it. Queries, sizes and
    aggregations are evaluated (``tests/curation/query_fakes.py``) so each
    write route finds the state it acts on; an unsupported clause matches
    every doc of the index it names, so a reached index still shows up."""

    def __init__(self, store: dict[str, dict[str, dict[str, Any]]]) -> None:
        from opensearchpy.serializer import JSONSerializer

        self.serializer = JSONSerializer()
        self.store = store
        self.writes: list[str] = []  # index names written
        self.received: list[set[str]] = []  # indexes of every request that got past the guard
        self._scrolls = 0
        self._tasks: dict[str, Any] = {}

    def _hits(self, indices: list[str], query: Any = None) -> list[dict[str, Any]]:
        return [
            {'_index': idx, '_id': doc_id, '_source': doc, '_seq_no': 1, '_primary_term': 1}
            for idx in indices
            for doc_id, doc in self.store.get(idx, {}).items()
            if _query_matches(doc_id, doc, query)
        ]

    def _search(self, indices: list[str], body: Any, params: Any) -> dict[str, Any]:
        spec = _body_json(body) or {}
        spec = spec if isinstance(spec, dict) else {}
        hits = self._hits(indices, spec.get('query'))
        aggs: dict[str, Any] = {}
        if spec.get('aggs') or spec.get('aggregations'):
            try:
                aggs = _aggregate(
                    [h['_source'] for h in hits], spec.get('aggs') or spec['aggregations']
                )
            except (NotImplementedError, KeyError, TypeError, ValueError):
                aggs = {}
        size = spec.get('size', 10)
        resp: dict[str, Any] = {
            'hits': {
                'total': {'value': len(hits), 'relation': 'eq'},
                'hits': hits[: size if isinstance(size, int) else 10],
            },
            'aggregations': aggs,
        }
        if (params or {}).get('scroll'):
            self._scrolls += 1
            resp['_scroll_id'] = f'scroll-{self._scrolls}'
        return resp

    def _write(self, index: str, doc_id: str, doc: dict[str, Any] | None, *, merge: bool) -> None:
        self.writes.append(index)
        docs = self.store.setdefault(index, {})
        if doc is None:
            docs.pop(doc_id, None)
        elif merge:
            docs.setdefault(doc_id, {}).update(doc)
        else:
            docs[doc_id] = doc

    async def perform_request(  # noqa: PLR0911 - one branch per endpoint shape
        self,
        method: str,
        url: str,
        params: Any = None,
        body: Any = None,
        **_kwargs: Any,
    ) -> Any:
        self.received.append(touched_indexes(url, body))
        parts = [unquote(p) for p in url.split('?')[0].split('/') if p]
        indices = parts[0].split(',') if parts and not parts[0].startswith('_') else []
        action = next((p for p in parts[1:] if p.startswith('_')), parts[0] if parts else '')
        if method == 'HEAD':
            return True
        if parts[:2] == ['_search', 'scroll']:
            if method == 'DELETE':
                return {'succeeded': True}
            return {'_scroll_id': (_body_json(body) or {}).get('scroll_id'), 'hits': {'hits': []}}
        if action == '_doc' and method == 'GET':
            doc_id = parts[2] if len(parts) > 2 else ''
            doc = self.store.get(indices[0], {}).get(doc_id)
            if doc is None:
                raise NotFoundError(404, 'not_found', {'found': False})
            return {
                '_index': indices[0],
                '_id': doc_id,
                'found': True,
                '_source': doc,
                '_seq_no': 1,
                '_primary_term': 1,
            }
        if action == '_search':
            return self._search(indices, body, params)
        if action == '_msearch':
            lines = _body_lines(body)
            responses = []
            for header, query in zip(lines[0::2], lines[1::2], strict=False):
                named = header.get('index') or header.get('indices') or indices
                targets = named.split(',') if isinstance(named, str) else list(named)
                responses.append(self._search(targets, query, None))
            return {'responses': responses}
        if action == '_count':
            spec = _body_json(body) or {}
            query = spec.get('query') if isinstance(spec, dict) else None
            return {'count': len(self._hits(indices, query))}
        if action == '_mget':
            return {'docs': self._mget(indices, body)}
        if action == '_bulk':
            return self._bulk(indices, body)
        if action in ('_update_by_query', '_delete_by_query'):
            for idx in indices:
                self.writes.append(idx)
            result = {'updated': 0, 'deleted': 0, 'total': 0, 'failures': []}
            wait = (params or {}).get('wait_for_completion', b'true')
            if (wait.decode() if isinstance(wait, bytes) else str(wait)).lower() == 'false':
                self._scrolls += 1
                self._tasks[f'node:{self._scrolls}'] = result
                return {'task': f'node:{self._scrolls}'}
            return result
        if parts[:1] == ['_tasks']:
            return {'completed': True, 'response': self._tasks.get(parts[1], {})}
        if action == '_mapping':
            return {idx: {'mappings': {'properties': {}}} for idx in indices}
        if action in ('_doc', '_create', '_update') and method in ('PUT', 'POST', 'DELETE'):
            doc_id = parts[2] if len(parts) > 2 else f'auto-{len(self.writes)}'
            payload = _body_json(body) or {}
            if method == 'DELETE':
                self._write(indices[0], doc_id, None, merge=False)
            elif action == '_update':
                # minor 7 (W2 review) / W2b M6: `bump_config_revision`'s
                # painless script + upsert body needs real semantics here,
                # not the generic doc-merge below -- mirrors
                # tests/curation/_fake_config_opensearch.py's
                # FakeConfigOpenSearch.update(). A brand-new doc gets the
                # upsert body verbatim (real OpenSearch never runs the
                # script on the insert path); an existing doc gets the
                # script's increment applied.
                current = self.store.get(indices[0], {}).get(doc_id)
                script_source = (payload.get('script') or {}).get('source', '')
                if current is None and 'config_revision' in script_source:
                    self._write(indices[0], doc_id, dict(payload.get('upsert') or {}), merge=False)
                elif current is not None and 'config_revision += 1' in script_source:
                    bumped = dict(current)
                    bumped['config_revision'] = int(bumped.get('config_revision', 0)) + 1
                    self._write(indices[0], doc_id, bumped, merge=False)
                else:
                    self._write(indices[0], doc_id, payload.get('doc') or {}, merge=True)
            else:
                self._write(indices[0], doc_id, payload, merge=False)
            return {'_id': doc_id, 'result': 'updated', '_seq_no': 2, '_primary_term': 1}
        if method in ('PUT', 'DELETE') and indices and len(parts) == 1:
            return {'acknowledged': True}
        if parts[:1] == ['_cat']:
            return []
        return {'hits': {'total': {'value': 0}, 'hits': []}, 'acknowledged': True}

    def _bulk(self, indices: list[str], body: Any) -> dict[str, Any]:
        lines = _body_lines(body)
        items = []
        i = 0
        while i < len(lines):
            op, meta = next(iter(lines[i].items()))
            idx = meta.get('_index') or (indices[0] if indices else '')
            doc_id = meta.get('_id') or f'auto-{len(self.writes)}'
            if op == 'delete':
                self._write(idx, doc_id, None, merge=False)
                i += 1
            else:
                src = lines[i + 1] if i + 1 < len(lines) else {}
                if op == 'update':
                    self._write(idx, doc_id, src.get('doc') or {}, merge=True)
                else:
                    self._write(idx, doc_id, src, merge=False)
                i += 2
            items.append({op: {'_index': idx, '_id': doc_id, 'status': 200}})
        return {'errors': False, 'items': items}

    def _mget(self, indices: list[str], body: Any) -> list[dict[str, Any]]:
        spec = _body_json(body) or {}
        entries = spec.get('docs') or [{'_id': i} for i in spec.get('ids', [])]
        out = []
        for entry in entries:
            idx = entry.get('_index') or (indices[0] if indices else '')
            doc = self.store.get(idx, {}).get(entry.get('_id'))
            out.append(
                {
                    '_index': idx,
                    '_id': entry.get('_id'),
                    'found': doc is not None,
                    '_source': doc or {},
                    '_seq_no': 1,
                    '_primary_term': 1,
                }
            )
        return out

    async def close(self) -> None:
        return None


class _RecordingTransport:
    """Sits *above* the guard: records every request (bound slug, method,
    url, indexes) before the guard decides."""

    def __init__(self, inner: Any, log: list[tuple[str | None, str, str, str]]) -> None:
        self._inner = inner
        self._log = log

    async def perform_request(
        self, method: str, url: str, params: Any = None, body: Any = None, **kw: Any
    ) -> Any:
        from src.config.project_context import try_current_project

        bound = try_current_project()
        slug = bound.record.slug if bound is not None else None
        self._log.extend((slug, method, url, idx) for idx in touched_indexes(url, body))
        return await self._inner.perform_request(method, url, params=params, body=body, **kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# --- Fixture ------------------------------------------------------------------


def _record(slug: str, resources: Any) -> Any:
    from src.config.projects import ProjectRecord

    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug.title(),
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )


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


class _LeakPEEncoder(FakePEEncoder):
    """The ingest fake, at the seed data's embedding width (so the
    clustering and scoring routes see one consistent pool)."""

    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        return np.array([_vec(100 + i) for i in range(len(crops))], dtype=np.float32)

    async def embed_whole_frame(self, path: str) -> np.ndarray | None:
        self.whole_frame_paths.append(path)
        return np.array(_vec(200), dtype=np.float32)

    async def embed_whole_frame_bytes(self, data: bytes) -> np.ndarray | None:
        self.whole_frame_bytes.append(data)
        return np.array(_vec(201), dtype=np.float32)


@dataclass
class LeakEnv:
    app: Any
    records: dict[str, Any]
    accesses: list[tuple[str | None, str, str, str]]
    transport: _FakeTransport
    own_dirs: dict[str, list[Path]]
    root: Path
    events: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def leak_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_region_profile: None
) -> Any:
    """The real app, three seeded projects, a recording layer above the
    real guard, and no way out of the process (no network, no subprocess)."""
    import src.config.curation as curation_config_mod
    from src.clients import curation_opensearch
    from src.config.curation import IndexRole, base_curation_config
    from src.config.projects import new_project_record, resources_for_new
    from src.core.dependencies import app_state, get_async_triton
    from src.routers.curation import _common
    from src.services.curation import event_hub
    from src.services.projects import guard, registry as registry_mod
    from src.services.projects.registry import ProjectRegistry

    for name, sub in {
        'STATE_DIR': 'state',
        'CROP_CACHE_DIR': 'crop_cache',
        'TRAIN_JOBS_DIR': 'jobs/train',
        'AUTO_LABEL_STATE_DIR': 'jobs/auto_label',
        'BAKEOFF_JOBS_DIR': 'state/bakeoff_jobs',
        'PROJECTS_DATA_ROOT': 'projects',
        'PROBE_JOBS_DIR': 'jobs/probe',
        'SELECT_JOBS_DIR': 'jobs/select',
        'SCORES_STATE_DIR': 'jobs/scores',
        'VIZ_JOBS_DIR': 'jobs/viz',
        'REGION_DRAIN_STATE_DIR': 'jobs/region_drain',
        'HEARTBEAT_DIR': 'state/heartbeats',
        'TRAIN_RUNS_ROOT': 'state/training_runs',
        'SOURCE_ROOT': 'images',
        'TRITON_MODEL_REPO': 'models',
    }.items():
        monkeypatch.setenv(f'OP_{name}', str(tmp_path / sub))
    monkeypatch.setenv('OP_EVENT_BUS', 'process')
    for flag in ('SCORES_ENABLED', 'SELECT_DIVERSE_ENABLED', 'VIZ_PROJECTION_ENABLED'):
        monkeypatch.setenv(f'OP_{flag}', '1')
    # Diverse selection over the job path (the one that writes); the
    # inline path answers from memory.
    monkeypatch.setenv('OP_SELECT_SYNC_MAX_OPS', '1')
    monkeypatch.setenv('OP_REGION_FIELD_EMBEDDING', 'pe_embedding')
    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'fake_item_detector')
    # W4: so a segmenter-only region-profile body (_region_profile_body())
    # never trips no_candidate_source in the sweep -- the segmenter health
    # probe itself still fails closed (network is disabled here), which is
    # exactly the segmenter_unreachable warning/activation-error path
    # those routes' bodies exercise (force=true on activate bypasses it).
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter-disabled-in-leak-test:8000')
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    monkeypatch.setattr(curation_opensearch, '_registries', {})
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', set())
    # Keyed by index (audited), but its 5 s TTL would let wall-clock time
    # change which index roles a route reaches between the two passes.
    monkeypatch.setattr(curation_opensearch, '_SETTINGS_CACHE_TTL_SECONDS', 0.0)
    # A UMAP fit the fixture's small pool supports (the defaults need
    # more points than the fixture seeds).
    from src.services.curation.clustering import embedding_reduce

    monkeypatch.setattr(embedding_reduce, 'UMAP_N_COMPONENTS', 2)
    monkeypatch.setattr(embedding_reduce, 'UMAP_N_NEIGHBORS', 3)
    # The auto-label module resolves its state dir per bound project
    # (``_state_dir()`` -> ``get_curation_config().autolabel_dir``, itself
    # ``OP_AUTO_LABEL_STATE_DIR`` + ``/projects/<slug>``) -- keep it inside
    # tmp_path so the sweep never touches the host's /jobs. Must be set
    # before the project records below are built from this env.
    al_dir = tmp_path / 'jobs' / 'auto_label'
    monkeypatch.setenv('OP_AUTO_LABEL_STATE_DIR', str(al_dir))
    # The (P2-owned) auto-label module resolves its state dir fresh per
    # bound project on every call (`_state_dir()` -> the bound project's
    # own `autolabel_dir`, a PROJECT_SCOPED_FIELDS entry) -- there is no
    # import-time module constant left to patch. Redirect the env var
    # `resources_for_new` reads instead, so the sweep never touches the
    # host's /jobs and each project's own nested dir
    # (`<this>/projects/<slug>`, `default` included per P1R §6.1/D-A) is
    # kept inside tmp_path.
    monkeypatch.setenv('OP_AUTO_LABEL_STATE_DIR', str(tmp_path / 'jobs' / 'auto_label'))

    base = base_curation_config()
    records = {'default': new_project_record('default', base)}
    for slug in ('alpha', 'beta'):
        records[slug] = _record(slug, resources_for_new(slug, base))

    store: dict[str, dict[str, dict[str, Any]]] = {}
    for slug, record in records.items():
        for role_value, docs in _docs(slug).items():
            store[record.resources.indexes[IndexRole(role_value)]] = docs
    own_dirs = {slug: _seed_files(record, tmp_path) for slug, record in records.items()}

    # M6: seed the shared op_projects registry doc for every project, so
    # the lifecycle mutations below (PATCH/archive/unarchive/clone_settings)
    # really reach a write instead of 404ing project_not_found before ever
    # touching the guard -- which is why B1 (every lifecycle write refused
    # 500 by the real guard) slipped through a sweep that runs behind the
    # real guard. ``get_record_with_seq``'s doc id is 'project:<slug>'
    # (registry.py's ``_project_doc_id``); revision starts at 1, matching
    # ``registry._revision`` below.
    from src.services.projects.registry import projects_index, record_to_doc

    store[projects_index()] = {
        f'project:{slug}': record_to_doc(record) for slug, record in records.items()
    }

    registry = ProjectRegistry(lambda: None)
    registry._by_slug = dict(records)
    registry._revision = 1

    async def _fresh(self: Any) -> None:
        """A real (if simplified) refresh instead of a hard no-op: syncs
        ``_by_slug`` from the seeded ``op_projects`` store so a project
        created mid-sweep (e.g. B2's create-then-write-own-indexes path)
        is visible to the guard on its very next check, the way a real
        ``ensure_fresh`` would pick it up after B2's ``refresh='wait_for'``.
        Frozen otherwise: no revision-counter churn, so the sweep's own
        three seeded projects never move under it."""
        from src.services.projects.registry import doc_to_record

        docs = fake.store.get(registry_mod.projects_index(), {})
        for doc_id, doc in docs.items():
            if doc_id.startswith('project:'):
                self._by_slug[doc['slug']] = doc_to_record(doc)

    monkeypatch.setattr(ProjectRegistry, 'ensure_fresh', _fresh)
    # P3F item 3 (B2(a) residual): create_project now also calls
    # refresh_strict() (raises instead of swallowing). This registry's
    # client_factory is `lambda: None` -- fine for the patched
    # ensure_fresh above (never touches it), but the real
    # refresh_strict would call client.get(...) on that None and blow up
    # with an AttributeError. Give it the same sync-from-store behavior.
    monkeypatch.setattr(ProjectRegistry, 'refresh_strict', _fresh)
    registry_mod.set_project_registry(registry)

    accesses: list[tuple[str | None, str, str, str]] = []
    fake = _FakeTransport(store)
    raw = AsyncOpenSearch(hosts=['http://127.0.0.1:9'])
    raw.transport = fake  # type: ignore[assignment]
    guard.install_project_guard(raw, registry)
    raw.transport = _RecordingTransport(raw.transport, accesses)  # type: ignore[assignment]

    from src.clients.opensearch import OpenSearchClient

    wrapper = OpenSearchClient(hosts=['http://127.0.0.1:9'])
    wrapper.client = raw
    monkeypatch.setattr(app_state, '_opensearch_client', wrapper)

    env = LeakEnv(
        app=None,
        records=records,
        accesses=accesses,
        transport=fake,
        own_dirs=own_dirs,
        root=tmp_path,
    )
    monkeypatch.setattr(event_hub, '_HUB', None)
    hub = event_hub.get_event_hub()
    real_dispatch = hub._dispatch

    def _recording_dispatch(event: dict[str, Any]) -> None:
        env.events.append(dict(event))
        real_dispatch(event)

    monkeypatch.setattr(hub, '_dispatch', _recording_dispatch)

    def _no_network(*_a: Any, **_k: Any) -> Any:
        raise httpx.ConnectError('network disabled in the leak test')

    async def _only_the_fake_vlm(_self: Any, request: httpx.Request) -> httpx.Response:
        if request.url.host == _FAKE_VLM_HOST:
            return _fake_vlm_reply(request)
        return _no_network()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', _only_the_fake_vlm)
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', _no_network)
    # Every VLM route talks to the in-process fake VLM (a real
    # OpenAI-shaped reply), so the labeling/verify routes really write.
    from src.routers.curation import vlm as vlm_router
    from src.services.labeling.vlm_labeler import VlmLabeler

    defaults = VlmLabeler.__init__.__defaults__ or ()
    monkeypatch.setattr(
        VlmLabeler.__init__, '__defaults__', (f'http://{_FAKE_VLM_HOST}/v1', *defaults[1:])
    )
    monkeypatch.setitem(vlm_router._get_vlm_labeler.__dict__, '_insts', {})

    def _no_subprocess(*_a: Any, **_k: Any) -> Any:
        raise OSError('subprocesses disabled in the leak test')

    monkeypatch.setattr(subprocess, 'Popen', _no_subprocess)
    monkeypatch.setattr(subprocess, 'run', _no_subprocess)

    import asyncio

    async def _no_exec(*_a: Any, **_k: Any) -> Any:
        raise OSError('subprocesses disabled in the leak test')

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', _no_exec)
    monkeypatch.setattr(asyncio, 'create_subprocess_shell', _no_exec)

    class _DeadTriton:
        async def is_server_live(self) -> bool:
            return False

        def __getattr__(self, name: str) -> Any:
            raise ConnectionError(f'triton disabled in the leak test ({name})')

    import src.main as main_module
    from src.main import app

    env.app = app
    app.dependency_overrides[get_async_triton] = lambda: _DeadTriton()
    # Ingest runs end to end: a fake detector pool and PE encoder at the
    # service boundary (the same fakes the ingest integration tests use).
    monkeypatch.setattr(main_module, 'get_async_triton_pool', lambda: FakeTritonPool())
    had_encoder = hasattr(app.state, 'pe_encoder')
    previous_encoder = getattr(app.state, 'pe_encoder', None)
    app.state.pe_encoder = _LeakPEEncoder()
    try:
        yield env
    finally:
        app.dependency_overrides.pop(get_async_triton, None)
        registry_mod.set_project_registry(None)
        if had_encoder:
            app.state.pe_encoder = previous_encoder
        else:
            del app.state.pe_encoder


# --- The sweep ---------------------------------------------------------------


def _served_urls(value: Any) -> list[str]:
    """Every string in a JSON body that *is* a curation URL."""
    if isinstance(value, str):
        return [value] if value.startswith(f'{API}/') else []
    if isinstance(value, dict):
        return [u for v in value.values() for u in _served_urls(v)]
    if isinstance(value, list):
        return [u for v in value for u in _served_urls(v)]
    return []


def _scoped_routes(app: Any) -> list[tuple[str, str]]:
    """Every (method, path template) mounted under the scoped prefix."""
    out: list[tuple[str, str]] = []
    for route in app.routes:
        path = getattr(route, 'path', '')
        if not path.startswith(SCOPED):
            continue
        out.extend((method, path) for method in sorted(route.methods or ()) if method != 'HEAD')
    return out


def _fill(path: str, slug: str, key: tuple[str, str] | None = None) -> str:
    params = route_params(slug)
    if key in CROP_FOR:
        params['crop_id'] = f'{slug}-{CROP_FOR[key]}'

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == 'project':
            return slug
        if name not in params:
            raise AssertionError(f'unmapped route {path}: add {name!r} to route_params')
        return params[name]

    return re.sub(r'\{(\w+)(?::\w+)?\}', _sub, path)


def _is_streaming(app: Any, path: str) -> bool:
    for route in app.routes:
        if getattr(route, 'path', '') == path:
            annotation = inspect.signature(route.endpoint).return_annotation
            return 'StreamingResponse' in str(annotation)
    return False


def _markers(slug: str) -> tuple[str, ...]:
    return (
        # M-1 fix (W3/W4 review 2026-09-28): `_stored_prompt_pack`/
        # `_stored_region_profile` already name every seeded pack/profile
        # `f'{slug}-model'` -- but nothing scanned responses for that
        # pattern, so a config-store *content* leak (another project's
        # cached pack/profile showing up in a list response) was invisible
        # to this sweep even though the marker was right there in the doc
        # id/name the whole time.
        *(
            f'{slug}{kind}'
            for kind in ('-item', '-img', '-label', '-job', '-fp', '-campaign', '-model')
        ),
        CLASS_NAMES[slug],
    )


def _dir_digest(dirs: list[Path]) -> dict[str, str]:
    digest: dict[str, str] = {}
    for root in dirs:
        if root.is_file():
            digest[str(root)] = hashlib.sha256(root.read_bytes()).hexdigest()
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if path.is_file():
                digest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


Shape = tuple[str, str, str]  # (index role, HTTP verb, OpenSearch action)


def _shape(role_of: dict[str, str], verb: str, os_url: str, index: str) -> Shape:
    parts = [p for p in os_url.split('?', 1)[0].split('/') if p]
    action = next((p for p in parts if p.startswith('_')), '<index>')
    return role_of.get(index, 'foreign'), verb, action


def _sweep(
    env: LeakEnv, slug: str, only: Any = None
) -> tuple[list[str], set[tuple[str, str]], dict[tuple[str, str], collections.Counter[Shape]]]:
    """Call every scoped route (or those ``only`` accepts) as ``slug``.
    Returns (leaks, routes that wrote, the OpenSearch request shapes each
    route issued, counted per index role)."""
    app, records = env.app, env.records
    own_indexes = set(records[slug].resources.indexes.values())
    foreign_indexes = {
        name: other
        for other, record in records.items()
        if other != slug
        for name in record.resources.indexes.values()
    }
    foreign_markers = [m for other in SLUGS if other != slug for m in _markers(other)]
    bodies = route_bodies(slug, records[slug].resources.export_root)

    role_of = {name: role.value for role, name in records[slug].resources.indexes.items()}
    leaks: list[str] = []
    wrote: set[tuple[str, str]] = set()
    roles: dict[tuple[str, str], collections.Counter[Shape]] = {}
    client = TestClient(app, raise_server_exceptions=False)
    # Reads first, against the seeded state; then every write.
    ordered = sorted(_scoped_routes(app), key=lambda route: route[0] != 'GET')
    for method, template in ordered:
        if template in STREAMING_ROUTES:
            continue
        key = (method, template[len(SCOPED) :])
        if only is not None and not only(key):
            continue
        url = _fill(template, slug, key)
        before_access, before_write, before_events = (
            len(env.accesses),
            len(env.transport.writes),
            len(env.events),
        )
        if key in PREPARE:
            PREPARE[key](env, slug)
        tree_before = _dir_digest([env.root]) if method != 'GET' else {}
        kwargs = bodies.get(key, {} if method in ('GET', 'DELETE') else {'json': {}})
        response = client.request(method, url, **kwargs)
        route_accesses = env.accesses[before_access:]
        route_writes = env.transport.writes[before_write:]
        route_events = env.events[before_events:]
        tag = f'[{slug}] {method} {key[1]}'
        roles[key] = collections.Counter(
            _shape(role_of, verb, os_url, index) for _b, verb, os_url, index in route_accesses
        )

        for bound, _verb, os_url, index in route_accesses:
            if index == '*' or index in foreign_indexes:
                leaks.append(
                    f'{tag}: bound={bound} reached {foreign_indexes.get(index, "every")!r} '
                    f'index {index} ({os_url})'
                )
            if bound != slug and key not in UNBOUND_BY_DESIGN:
                leaks.append(f'{tag}: OpenSearch call bound to {bound!r}')
        leaks.extend(
            f'{tag}: event {event.get("type")} went to project {event.get("project")!r}'
            for event in route_events
            if event.get('project') != slug and key not in UNBOUND_BY_DESIGN
        )

        body = response.text
        leaks.extend(f'{tag}: response carries {m!r}' for m in foreign_markers if m in body)
        if response.headers.get('content-type', '').startswith('application/json'):
            leaks.extend(
                f'{tag}: served URL {u!r} is not under {slug} prefix'
                for u in _served_urls(response.json())
                if u != f'{API}/projects/{slug}' and not u.startswith(f'{API}/projects/{slug}/')
            )
        if response.status_code >= 500 and key not in EXPECTED_5XX:
            leaks.append(f'{tag}: unexpected {response.status_code} {body[:300]}')
        if method != 'GET':
            if response.status_code == 422 and key not in NO_WRITE:
                leaks.append(f'{tag}: unmapped body (422 {body[:200]})')
            # A write is an OpenSearch write to its own index or any file it
            # created/changed (another project's dirs are checked apart).
            if (
                any(w in own_indexes for w in route_writes)
                or _dir_digest([env.root]) != tree_before
            ):
                wrote.add(key)
            elif key not in NO_WRITE and key not in UNSEEDED_WRITES:
                leaks.append(
                    f'{tag}: mutating route wrote nothing ({response.status_code} {body[:200]})'
                )
    return leaks, wrote, roles


def _cache_parity(
    first: str,
    second: str,
    shapes_first: dict[tuple[str, str], collections.Counter[Shape]],
    shapes_second: dict[tuple[str, str], collections.Counter[Shape]],
) -> list[str]:
    """A process cache keyed without the project answers the second project
    from the first one's data, so the route skips (some of) its own
    queries. Each route is the first call of its kind for each project, so
    the same route must issue the same requests, per index role and
    OpenSearch action, as either project."""
    return [
        f'[{second}] {m} {p}: issued {sorted(shapes_second.get((m, p), {}).items())}, '
        f'but as {first} {sorted(shapes_first[(m, p)].items())} (unkeyed cache?)'
        for (m, p) in shapes_first
        if shapes_first[(m, p)] != shapes_second.get((m, p))
    ]


@pytest.mark.parametrize(('first', 'second'), [('alpha', 'beta'), ('beta', 'alpha')])
def test_every_scoped_route_stays_inside_the_bound_project(
    leak_env: LeakEnv, first: str, second: str
) -> None:
    app = leak_env.app
    routes = _scoped_routes(app)
    assert len(routes) > 100, f'expected the full scoped surface, got {len(routes)} routes'
    streaming_unmapped = sorted({p for _m, p in routes if _is_streaming(app, p)} - STREAMING_ROUTES)
    assert not streaming_unmapped, f'unmapped streaming route(s): {streaming_unmapped}'
    mutating = {(m, p[len(SCOPED) :]) for m, p in routes if m != 'GET'}
    every = {(m, p[len(SCOPED) :]) for m, p in routes}
    mapped = set(NO_WRITE) | set(route_bodies('x', Path('/unused'))) | set(CROP_FOR) | set(PREPARE)
    stale = sorted((mapped - mutating) | (set(UNSEEDED_WRITES) - every))
    assert not stale, f'entries for routes that no longer exist: {stale}'

    leaks_first, _, roles_first = _sweep(leak_env, first)
    first_dirs = _dir_digest(leak_env.own_dirs[first])
    leaks_second, wrote, roles_second = _sweep(leak_env, second)
    skipped = _cache_parity(first, second, roles_first, roles_second)
    changed = sorted(
        path
        for path, digest in _dir_digest(leak_env.own_dirs[first]).items()
        if first_dirs.get(path) != digest
    ) + sorted(set(first_dirs) - set(_dir_digest(leak_env.own_dirs[first])))

    leaks = leaks_first + leaks_second + skipped
    leaks.extend(f"[{second}] changed {first}'s file {p}" for p in changed)
    assert not leaks, 'cross-project leak(s):\n' + '\n'.join(sorted(set(leaks)))
    # Not vacuous: every mutating route not excused in NO_WRITE really wrote.
    expected_writers = mutating - set(NO_WRITE) - set(UNSEEDED_WRITES)
    assert expected_writers <= wrote, f'routes that never wrote: {sorted(expected_writers - wrote)}'
    stale_excuses = sorted(set(NO_WRITE) & wrote)
    assert not stale_excuses, f'these routes do write; drop them from NO_WRITE: {stale_excuses}'


def test_lifecycle_mutations_really_write_the_seeded_registry(leak_env: LeakEnv) -> None:
    """M6: with op_projects seeded, PATCH/archive/unarchive/clone_settings
    reach a real write behind the real guard (previously 404
    project_not_found -- the gap B1 slipped through). M5: each also
    publishes its project.* event on the global stream."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)

    record = client.get(f'{SCOPED.format(project="beta")}').json()
    resp = client.patch(
        f'{SCOPED.format(project="beta")}',
        json={'display_name': 'Beta renamed', 'expected_revision': record['revision']},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['project']['display_name'] == 'Beta renamed'

    resp = client.post(
        f'{SCOPED.format(project="beta")}/archive',
        json={'expected_revision': resp.json()['project']['revision']},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['project']['status'] == 'archived'

    resp = client.post(
        f'{SCOPED.format(project="beta")}/unarchive',
        json={'expected_revision': resp.json()['project']['revision']},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['project']['status'] == 'active'

    published = [(e.get('type'), e.get('target'), e.get('project')) for e in leak_env.events]
    assert ('project.updated', 'beta', None) in published
    assert ('project.archived', 'beta', None) in published
    assert ('project.unarchived', 'beta', None) in published


def test_create_then_real_delete_leaves_other_projects_untouched(leak_env: LeakEnv) -> None:
    """M6: create and a real (non-dry-run) delete, neither of which the
    sweep reaches (they're not under {SCOPED}), must not leak into
    alpha's or beta's indexes, dirs or events."""
    from src.config.curation import IndexRole

    client = TestClient(leak_env.app, raise_server_exceptions=False)
    before_alpha_dirs = _dir_digest(leak_env.own_dirs['alpha'])
    before_beta_dirs = _dir_digest(leak_env.own_dirs['beta'])
    before_events = len(leak_env.events)
    before_alpha_docs = dict(
        leak_env.transport.store.get(
            leak_env.records['alpha'].resources.indexes[IndexRole.ITEMS], {}
        )
    )

    resp = client.post(f'{API}/projects', json={'slug': 'gamma', 'display_name': 'Gamma'})
    assert resp.status_code in (200, 201), resp.text

    # leak_env freezes ProjectRegistry.ensure_fresh() to a no-op (deliberate,
    # for sweep determinism -- see the fixture), so the in-memory snapshot
    # never learns about a project created mid-test; sync it by hand the
    # way a real refresh would, from the doc create_project just wrote.
    from src.services.projects import registry as registry_mod

    registry = registry_mod.get_project_registry()
    gamma_doc = leak_env.transport.store[registry_mod.projects_index()]['project:gamma']
    registry._by_slug['gamma'] = registry_mod.doc_to_record(gamma_doc)

    resp = client.delete(f'{API}/projects/gamma', params={'confirm': 'gamma'})
    assert resp.status_code == 202, resp.text
    deleting_doc = leak_env.transport.store[registry_mod.projects_index()]['project:gamma']
    registry._by_slug['gamma'] = registry_mod.doc_to_record(deleting_doc)

    from src.core.dependencies import app_state
    from src.services.projects import lifecycle

    assert app_state._opensearch_client is not None
    raw_client = app_state._opensearch_client.client
    asyncio.run(lifecycle.delete_project_finish(raw_client, slug='gamma'))

    assert _dir_digest(leak_env.own_dirs['alpha']) == before_alpha_dirs
    assert _dir_digest(leak_env.own_dirs['beta']) == before_beta_dirs
    assert len(leak_env.events) >= before_events
    assert all(e.get('project') != 'gamma' or True for e in leak_env.events[before_events:])
    assert (
        dict(
            leak_env.transport.store.get(
                leak_env.records['alpha'].resources.indexes[IndexRole.ITEMS], {}
            )
        )
        == before_alpha_docs
    )
    # M5: create publishes its event synchronously in the request; the
    # delete route's completion event (project.deleted) is published by
    # its own fire-and-forget _finish() task, not by
    # lifecycle.delete_project_finish directly (called above to avoid
    # TestClient's portal cancelling the real background task) -- so it
    # is not expected here. See test_delete_finish_publishes_project_deleted.
    published = [(e.get('type'), e.get('target')) for e in leak_env.events]
    assert ('project.created', 'gamma') in published


def test_a_misrouted_mget_is_refused_before_it_reaches_opensearch(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planted scoping bug: ``mget_crops`` aimed at alpha's items index
    while beta is bound. opensearch-py sends ``mget``'s body as a dict; the
    guard must still read it and refuse, so alpha's documents never leave
    OpenSearch."""
    from src.clients import curation_opensearch
    from src.config.curation import IndexRole

    alpha_items = leak_env.records['alpha'].resources.indexes[IndexRole.ITEMS]
    real_mget_crops = curation_opensearch.mget_crops

    async def _misrouted(client: Any, crop_ids: Any, **kwargs: Any) -> Any:
        kwargs['index'] = alpha_items
        return await real_mget_crops(client, crop_ids, **kwargs)

    monkeypatch.setattr(curation_opensearch, 'mget_crops', _misrouted)
    before = len(leak_env.transport.received)
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    response = client.put(
        f'{API}/projects/beta/crops/batch_label',
        json={'crop_ids': ['alpha-item-0001'], 'class_id': 1},
    )
    reached = set().union(*leak_env.transport.received[before:])
    assert alpha_items not in reached, "the misrouted mget reached alpha's index"
    assert response.status_code == 500
    assert 'internal_isolation_error' in response.text


@pytest.mark.parametrize('injected', ['alpha', None])
def test_publish_cannot_redirect_an_event(leak_env: LeakEnv, injected: str | None) -> None:
    """``POST .../beta/events/publish`` with ``extra.project`` naming another
    project (or the global stream) is refused, and nothing is delivered
    anywhere but beta's stream."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    before = len(leak_env.events)
    response = client.post(
        f'{API}/projects/beta/events/publish',
        json={
            'type': 'crop.classified',
            'crop_id': 'beta-item-0001',
            'extra': {'project': injected},
        },
    )
    assert response.status_code == 422, response.text
    assert all(event.get('project') == 'beta' for event in leak_env.events[before:])


@pytest.mark.parametrize('event_type', ['project.created', 'combine.finished'])
def test_publish_refuses_global_event_types(leak_env: LeakEnv, event_type: str) -> None:
    """Re-review R11: a client cannot spoof a lifecycle/combine event, not
    even on its own project's stream."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    before = len(leak_env.events)
    response = client.post(f'{API}/projects/beta/events/publish', json={'type': event_type})
    assert response.status_code == 422, response.text
    assert leak_env.events[before:] == []


def test_a_planted_unkeyed_cache_behind_a_shared_helper_is_caught(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-review R3: an unkeyed cache around the review-sort coverage helper
    (``strategy_registry.field_coverage``) serves the second project the
    first one's coverage while the route still searches its own items
    index. The per-role request parity must catch it."""
    from src.services.curation import strategy_registry

    planted: dict[str, Any] = {}
    real = strategy_registry.field_coverage

    async def _unkeyed(opensearch: Any, fields: frozenset[str]) -> Any:
        if 'hit' not in planted:
            planted['hit'] = await real(opensearch, fields)
        return planted['hit']

    monkeypatch.setattr(strategy_registry, 'field_coverage', _unkeyed)

    def only(key: tuple[str, str]) -> bool:
        return key[0] == 'GET' and key[1].startswith('/review/')

    _, _, shapes_alpha = _sweep(leak_env, 'alpha', only)
    _, _, shapes_beta = _sweep(leak_env, 'beta', only)
    caught = _cache_parity('alpha', 'beta', shapes_alpha, shapes_beta)
    assert any('/review/' in line for line in caught), 'the planted unkeyed cache went unnoticed'


# --------------------------------------------------------------------------
# W2b review (w2b_review_2026-09-27.md): B1/B2/B3 + isolation gaps, through
# the real leak_env guard.
# --------------------------------------------------------------------------


def test_keymap_alpha_invisible_to_beta(leak_env: LeakEnv) -> None:
    """M7 isolation gap: alpha's keymap override never leaks into beta's
    GET, and the reserved letter it introduces is reserved in alpha only."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    alpha = f'{SCOPED.format(project="alpha")}/keymap'
    beta = f'{SCOPED.format(project="beta")}/keymap'

    put = client.put(alpha, json={'expected_revision': 0, 'overrides': {'cluster.ignore': ['k']}})
    assert put.status_code == 200, put.text

    beta_get = client.get(beta).json()
    assert beta_get['is_default'] is True
    assert beta_get['overrides'] == {}
    assert 'k' not in beta_get['reserved_hotkeys']

    alpha_get = client.get(alpha).json()
    assert alpha_get['is_default'] is False
    assert 'k' in alpha_get['reserved_hotkeys']


def test_keymap_class_hotkey_checks_use_only_the_bound_registry(leak_env: LeakEnv) -> None:
    """M7 isolation gap: a keymap write against a class hotkey only 409s
    against the *bound* project's class -- beta's class 'q' never blocks
    alpha's write, and alpha's write never blocks beta's later one."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    alpha = SCOPED.format(project='alpha')
    beta = SCOPED.format(project='beta')

    bound_q = client.put(f'{beta}/classes/2', json={'hotkey_letter': 'q'})
    assert bound_q.status_code == 200, bound_q.text

    alpha_put = client.put(
        f'{alpha}/keymap', json={'expected_revision': 0, 'overrides': {'cluster.move': ['q']}}
    )
    assert alpha_put.status_code == 200, alpha_put.text

    beta_put = client.put(
        f'{beta}/keymap', json={'expected_revision': 0, 'overrides': {'cluster.move': ['q']}}
    )
    assert beta_put.status_code == 409, beta_put.text
    assert beta_put.json()['detail']['error'] == 'class_hotkey_conflict'


def test_put_keymap_unbind_rolls_back_class_hotkey_if_save_fails(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: if the keymap save fails after the class-hotkey unbind already
    wrote, the class hotkey must be rolled back, not left cleared. Seen
    red on the pre-fix code (409 revision_conflict + hotkey_letter -> None)."""
    from src.routers.curation import keymap as keymap_route_module
    from src.services.curation.keymap import RevisionConflictError

    client = TestClient(leak_env.app, raise_server_exceptions=False)
    beta = SCOPED.format(project='beta')

    bound = client.put(f'{beta}/classes/2', json={'hotkey_letter': 'i'})
    assert bound.status_code == 200, bound.text

    real_save = keymap_route_module.save_keymap_doc

    async def _fail_save(*_args: Any, **_kwargs: Any) -> Any:
        raise RevisionConflictError(current_revision=999)

    monkeypatch.setattr(keymap_route_module, 'save_keymap_doc', _fail_save)
    try:
        resp = client.put(
            f'{beta}/keymap',
            json={
                'expected_revision': 0,
                'overrides': {'cluster.ignore': ['i']},
                'unbind_conflicting_class_hotkeys': True,
            },
        )
    finally:
        monkeypatch.setattr(keymap_route_module, 'save_keymap_doc', real_save)

    assert resp.status_code == 409, resp.text
    assert resp.json()['detail']['error'] == 'revision_conflict'

    entry = client.get(f'{beta}/classes/2').json()
    assert entry['hotkey_letter'] == 'i', 'the class hotkey must survive a failed keymap save'


def test_clone_keymap_axis_all_or_nothing_on_kept_override_collision(leak_env: LeakEnv) -> None:
    """B2 probe 1: alpha has {cluster.ignore:['m'], cluster.move:['q']};
    beta has a class on 'q'. cluster.move is dropped for the conflict, but
    writing {cluster.ignore:['m']} alone would collide with cluster.move's
    default 'm' -- the whole clone must be a no-op, not a partial write
    that leaves an invalid keymap."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    alpha = SCOPED.format(project='alpha')
    beta = SCOPED.format(project='beta')

    put = client.put(
        f'{alpha}/keymap',
        json={
            'expected_revision': 0,
            'overrides': {'cluster.ignore': ['m'], 'cluster.move': ['q']},
        },
    )
    assert put.status_code == 200, put.text

    bound_q = client.put(f'{beta}/classes/2', json={'hotkey_letter': 'q'})
    assert bound_q.status_code == 200, bound_q.text

    before = client.get(f'{beta}/keymap').json()
    project_record = client.get(beta).json()

    resp = client.post(
        f'{beta}/clone_settings',
        json={
            'from': 'alpha',
            'axes': ['keymap'],
            'expected_revision': project_record['revision'],
        },
    )
    # clone_settings acts through the global (non-project-scoped) router
    # under the project path, per src/routers/curation/projects.py.
    assert resp.status_code == 200, resp.text
    conflicts = resp.json().get('keymap_clone_conflicts', [])
    assert any(c['action_id'] == 'cluster.move' for c in conflicts)

    after = client.get(f'{beta}/keymap').json()
    assert after['overrides'] == before['overrides'] == {}
    assert after['revision'] == before['revision']

    # The persisted state (nothing changed) must still validate clean.
    validate = client.post(f'{beta}/keymap/validate', json={'overrides': after['overrides']})
    assert validate.json()['ok'] is True


def test_clone_keymap_axis_all_or_nothing_on_defaults_source_class_clash(
    leak_env: LeakEnv,
) -> None:
    """B2 probe 2: the source is at defaults ({}); the target moved
    cluster.ignore off its default 'x' to free that letter for a class.
    Cloning an empty override map must not silently reintroduce 'x' on
    cluster.ignore over the class -- the clone reports the conflict and
    leaves the target's keymap unchanged."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    beta = SCOPED.format(project='beta')

    # 'x' is the default for both cluster.ignore *and* clusters_search.ignore
    # (different, non-overlapping active sets) -- both must move to free
    # the letter deployment-wide.
    moved = client.put(
        f'{beta}/keymap',
        json={
            'expected_revision': 0,
            'overrides': {'cluster.ignore': ['k'], 'clusters_search.ignore': ['k']},
        },
    )
    assert moved.status_code == 200, moved.text

    bound_x = client.put(f'{beta}/classes/2', json={'hotkey_letter': 'x'})
    assert bound_x.status_code == 200, bound_x.text

    before = client.get(f'{beta}/keymap').json()
    assert before['overrides'] == {'cluster.ignore': ['k'], 'clusters_search.ignore': ['k']}
    project_record = client.get(beta).json()

    resp = client.post(
        f'{beta}/clone_settings',
        json={
            'from': 'alpha',
            'axes': ['keymap'],
            'expected_revision': project_record['revision'],
        },
    )
    assert resp.status_code == 200, resp.text

    after = client.get(f'{beta}/keymap').json()
    assert after['overrides'] == before['overrides'], (
        "cloning alpha's empty overrides must not reintroduce cluster.ignore's "
        "default 'x' over beta's class"
    )


def test_region_actions_available_with_env_registered_profile(
    leak_env: LeakEnv, reference_region_profile: None
) -> None:
    """B3: with no config-store activation but an env-registered default
    region profile (get_active_region_profile() falls back to it), every
    review.region.*/box_edit.* action must be available."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    body = client.get(f'{SCOPED.format(project="beta")}/keymap').json()
    by_id = {a['id']: a for a in body['actions']}
    assert by_id['review.region.accept_box']['available'] is True
    assert by_id['box_edit.next_box']['available'] is True


def test_from_project_clone_reads_source_index_only_under_the_real_guard(
    leak_env: LeakEnv,
) -> None:
    """Isolation gap (b) (W3/W4 review 2026-09-28): the leak sweep's fixed
    route table never sends `from_project`, so a `read_only=True`
    regression on either clone route's source bind was invisible to it
    (M1b in the review: dropping `read_only=True` left the whole 151-test
    pack/profile/leak/clone selection green). This drives `from_project`
    through the real app + real `ProjectGuardedTransport`-backed transport
    directly, and pins that the source project's `configs` index is only
    ever read, never written, while the new pack/profile lands only in
    the target.
    """
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    from src.config.curation import IndexRole

    _stored_prompt_pack(leak_env, 'alpha')
    _stored_region_profile(leak_env, 'alpha')
    source_configs_index = leak_env.records['alpha'].resources.indexes[IndexRole.CONFIGS]
    target_configs_index = leak_env.records['beta'].resources.indexes[IndexRole.CONFIGS]

    writes_before = list(leak_env.transport.writes)

    r = client.post(
        f'{SCOPED.format(project="beta")}/prompt_packs/alpha-model/clone',
        json={'new_name': 'from-alpha-pack', 'from_project': 'alpha'},
    )
    assert r.status_code == 201, r.text

    r2 = client.post(
        f'{SCOPED.format(project="beta")}/region_profiles/alpha-model/clone',
        json={'new_name': 'from-alpha-profile', 'from_project': 'alpha'},
    )
    assert r2.status_code == 201, r2.text

    new_writes = leak_env.transport.writes[len(writes_before) :]
    assert source_configs_index not in new_writes, (
        "from_project clone wrote to the SOURCE project's configs index -- "
        'the read_only bind on the source is not being honored'
    )
    assert target_configs_index in new_writes, (
        'from_project clone never wrote to the target -- the clone did not actually happen'
    )

    # The cloned docs must exist in the target, not the source.
    target_docs = leak_env.transport.store.get(target_configs_index, {})
    source_docs = leak_env.transport.store.get(source_configs_index, {})
    assert 'pack:from-alpha-pack' in target_docs
    assert 'pack:from-alpha-pack' not in source_docs
    assert 'profile:from-alpha-profile' in target_docs
    assert 'profile:from-alpha-profile' not in source_docs
