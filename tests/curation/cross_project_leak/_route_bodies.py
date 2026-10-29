"""Route parameter values and request bodies for the isolation sweep."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from integration.ingest_fakes import jpeg_bytes

from curation.cross_project_leak._seeded_files import _source_root
from curation.cross_project_leak._seeders import (
    _VLM_ACTIVE,
    _open_vocab_body,
    _prompt_pack_body,
    _region_profile_body,
)
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


if TYPE_CHECKING:
    from pathlib import Path


def _dataset_zip(slug: str) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr(f'{slug}/data.yaml', f'train: images/train\nnames: [{slug}]\n')
    return buf.getvalue()


def route_params(slug: str) -> dict[str, str]:
    """Every path parameter a scoped route may carry, filled with ``slug``'s
    ids. A route with a parameter missing here fails ("unmapped route")."""
    return {
        'crop_id': f'{slug}-item-0001',
        'image_id': f'{slug}-img-0001',
        'class_id': '2',
        'cluster_id': '1',
        'job_id': f'{slug}-job-0001',
        'promote_id': f'{slug}-no-such-promote',
        'campaign_id': f'{slug}-campaign-0001',
        'name': f'{slug}-model',
        # A promoted model's triton_name is model_prefix + name (plan §5.3).
        'model_name': f'{slug}__model',
        'tab': 'uncertainty',
        'alias': f'{slug}-source',
        'artifact': 'results.csv',
        'box_id': 'b1',
        'revision': '1',
        'import_id': 'imp_20260101T000000_0123abcd',
        'run_id': 'bulk_20260101T000000_0123abcd',
        'loop_id': 'alp_20260101T000000000000_0123ab',
        'node_path': 'r10000',
        'source_id': 'src_unknown00',
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
        ('POST', '/classes/seed_from_detector'): {
            'json': {'names': ['seed probe'], 'dry_run': False}
        },
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
        ('POST', '/vlm/endpoints/{name}/activate'): {
            'json': {'expected_active': _VLM_ACTIVE, 'force': True}
        },
        ('POST', '/vlm/endpoints/active/rollback'): {'json': {'expected_active': _VLM_ACTIVE}},
        ('POST', '/vlm/endpoints/deactivate'): {'json': {'expected_active': _VLM_ACTIVE}},
        ('POST', '/vlm/label_batch'): {'json': {'crop_ids': [item]}},
        ('POST', '/vlm/verify_region_batch'): {
            'json': {'items': [{'crop_id': item, 'region_image_b64': b64}]}
        },
        ('POST', '/vlm/region_visible_batch'): {
            'json': {'items': [{'crop_id': item, 'image_b64': b64}]}
        },
        ('POST', '/ingest/image'): {'json': {'path': img}},
        ('POST', '/ingest/batch'): {'json': {'items': [{'path': f'{source}/{slug}-new-0002.jpg'}]}},
        ('POST', '/ingest/path_lookup'): {'json': {'image_paths': [img]}},
        ('PUT', '/ingest/policy'): {
            'json': {
                'expected_revision': 0,
                'embedding': {'mode': 'selected', 'classes': [f'{slug}_policy_class']},
            }
        },
        ('POST', '/ingest/policy/preview'): {'json': {'embedding': {'mode': 'lazy'}}},
        # The sweep checks where the write lands, not the audit gate (its own tests do).
        ('POST', '/clusters/auto_promote'): {'params': {'force': 'true'}},
        ('PUT', '/vlm/policy'): {
            'json': {'expected_revision': 0, 'scope': 'uncertain', 'max_crops_per_day': 7}
        },
        ('POST', '/ingest/upload'): {
            'files': [('images', (f'{slug}-up.jpg', jpeg_bytes(len(slug)), 'image/jpeg'))],
        },
        ('POST', '/probe/run'): {'json': {'job_id': f'{slug}-job-0001'}},
        ('POST', '/datasets/preview'): {'json': {'source': {'path': source}}},
        ('POST', '/datasets/imports'): {'json': {'source': {'path': source}}},
        ('POST', '/datasets/imports/{import_id}/cancel'): {},
        ('POST', '/ingest/bulk'): {'json': {'folder': f'{source}/no-such-folder'}},
        ('POST', '/ingest/bulk/{run_id}/cancel'): {},
        ('POST', '/ingest/sources'): {
            'json': {'name': 'sweep', 'backend': 'folder', 'folder': f'{source}/no-such-folder'}
        },
        ('PATCH', '/ingest/sources/{source_id}'): {'json': {'name': 'sweep'}},
        ('DELETE', '/ingest/sources/{source_id}'): {},
        ('POST', '/ingest/sources/{source_id}/run'): {},
        ('POST', '/ingest/sources/{source_id}/retry-dead'): {},
        ('POST', '/ingest/sources/{source_id}/events'): {'json': {'Records': []}},
        ('POST', '/ingest/objects'): {'json': {'items': [{'bucket': 'sweep', 'key': 'k.jpg'}]}},
        ('POST', '/ingest/objects/presign'): {
            'json': {'source_identifier': 's3://sweep/k.jpg#etag=e'}
        },
        ('POST', '/datasets/imports/{import_id}/resume'): {},
        ('POST', '/datasets/imports/{import_id}/undo'): {'json': {'dry_run': True}},
        ('POST', '/datasets/uploads'): {
            'files': [('file', ('d.zip', _dataset_zip(slug), 'application/zip'))]
        },
        ('PUT', '/models/{model_name}/sharing'): {'json': {'shared': True, 'expected_revision': 1}},
        ('PATCH', '/crops/{crop_id}/region_meta'): {
            'json': {'region_rejection_reason': f'{slug}-note'}
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
        # unified reprocess. The batch form names one item; the
        # single forms take their target from the path.
        ('POST', '/reprocess'): {
            'json': {'targets': {'crop_ids': [proposal]}, 'scopes': ['region'], 'dry_run': False}
        },
        ('POST', '/images/{image_id}/reprocess'): {'json': {'scopes': ['region']}},
        ('POST', '/crops/{crop_id}/reprocess'): {'json': {'scopes': ['region']}},
        ('POST', '/reprocess/jobs/{job_id}/cancel'): {'json': {}},
        # assist runs: a dry run on the seeded item (the sweep binds
        # no VLM, so the plan stops at the VLM gate); cancel names no live run.
        ('POST', '/assist/runs'): {
            'json': {
                'target': {'kind': 'crops', 'crop_ids': [item]},
                'actions': ['classify'],
                'dry_run': True,
            }
        },
        # auto-label loops: a dry-run plan (the sweep binds no VLM, so it stops at the VLM
        # gate); the signals name no live loop.
        ('POST', '/assist/loops'): {'json': {'scope': {'min_root_size': 1}, 'dry_run': True}},
        ('POST', '/assist/loops/{loop_id}/cancel'): {'json': {}},
        ('POST', '/assist/loops/{loop_id}/resume'): {'json': {}},
        ('POST', '/assist/loops/{loop_id}/rollback'): {'json': {'dry_run': False}},
        ('POST', '/assist/runs/{run_id}/cancel'): {'json': {}},
        ('POST', '/assist/runs/{run_id}/apply'): {'json': {'class_name': 'x', 'dry_run': False}},
        ('POST', '/test_holdout/freeze'): {'json': {'percent': 10}},
        ('POST', '/review/new_class_proposals/resolve'): {
            'json': {'label': f'{slug}-proposal', 'class_id': 1}
        },
        ('POST', '/scores/compute'): {'json': {}},
        ('POST', '/select/diverse'): {'json': {'k': 1}},
        # minor 7: a config-store axis in the body, not just
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
        # prompt-pack CRUD. '{name}' (route_params) is a stored pack
        # PREPARE resets to revision 1 immediately before each of these
        # (see the _prompt_pack_* PREPARE hooks) -- independent of
        # whatever an earlier route in the same pass left behind.
        ('POST', '/prompt_packs'): {
            'json': {'name': f'{slug}-newpack', 'body': _prompt_pack_body()}
        },
        ('POST', '/prompt_packs/validate'): {'json': {'name': None, 'body': _prompt_pack_body()}},
        ('POST', '/prompt_packs/test'): {'json': {'call': 'region_visible', 'crop_ids': [item]}},
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
        # region-profile CRUD. Same PREPARE-resets-to-a-known-revision
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
        ('POST', '/region_profiles/test'): {
            'json': {'crop_id': item, 'draft': _region_profile_body()}
        },
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
        # Open-vocabulary prompt sets: same shape as the region-profile routes.
        ('POST', '/open_vocab'): {'json': {'name': f'{slug}-newset', 'body': _open_vocab_body()}},
        ('POST', '/open_vocab/validate'): {'json': {'name': None, 'body': _open_vocab_body()}},
        ('POST', '/open_vocab/test'): {
            'json': {'image_id': f'{slug}-img-0001', 'target': _open_vocab_body()['targets'][0]}
        },
        ('POST', '/open_vocab/active/rollback'): {'json': {'expected_active': None}},
        ('POST', '/open_vocab/deactivate'): {'json': {'expected_active': None}},
        ('POST', '/open_vocab/{name}/clone'): {
            'json': {'new_name': f'{slug}-setclone', 'source': 'stored'}
        },
        ('PUT', '/open_vocab/{name}'): {
            'json': {'expected_revision': 1, 'body': _open_vocab_body()}
        },
        ('DELETE', '/open_vocab/{name}'): {'params': {'expected_revision': '1'}},
        ('POST', '/open_vocab/{name}/activate'): {
            'json': {'revision': None, 'expected_active': None, 'force': True}
        },
    }
