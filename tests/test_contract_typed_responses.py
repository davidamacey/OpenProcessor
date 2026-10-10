"""Every curation route that returns JSON declares its body in the OpenAPI contract.

A response whose schema is empty or a bare ``object`` carries no contract: a
field added to it reaches clients unannounced. The allowlist below is the
complete list of routes still in that state, each under a justification; a
route that gets a model must leave the list (a stale entry fails), and a new
untyped route fails until it is typed or listed with a reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONTRACT = Path(__file__).resolve().parents[1] / 'contracts' / 'openapi' / 'curation.json'
PROJECT_PREFIX = '/curation/projects/{project}'

STREAM_OR_FILE = frozenset(
    {
        'GET /curation/events',
        'GET /crops/{crop_id}/image',
        'GET /crops/{crop_id}/region_thumbnail',
        'GET /crops/{crop_id}/thumbnail',
        'GET /events',
        'GET /export/registry/{artifact}',
        'GET /images/root/{alias}',
        'GET /images/serve',
        'GET /pipeline/events',
        'GET /train/artifacts/{job_id}/{name}',
    }
)

LEGACY_DICT = frozenset(
    {
        'GET /train/manifest/{job_id}',
        'GET /class_sources',
        'POST /classes',
        'POST /classes/merge',
        'POST /classes/sync_to_opensearch',
        'PUT /classes/{class_id}',
        'POST /cluster/umap/rebuild',
        'GET /clusters',
        'POST /clusters/auto_promote',
        'POST /clusters/refine/{cluster_id}',
        'GET /clusters/representatives',
        'POST /crops/discard_batch',
        'POST /crops/flag_new_class',
        'POST /crops/label/undo_batch',
        'POST /crops/region/undo_batch',
        'POST /crops/{crop_id}/discard',
        'GET /crops/{crop_id}/history',
        'DELETE /crops/{crop_id}/label',
        'PUT /crops/{crop_id}/label',
        'POST /crops/{crop_id}/label/undo',
        'POST /crops/{crop_id}/region/undo',
        'POST /crops/{crop_id}/review_dismiss',
        'POST /crops/{crop_id}/review_undismiss',
        'POST /crops/{crop_id}/vlm_dismiss',
        'POST /crops/{crop_id}/vlm_dismiss/undo',
        'POST /events/publish',
        'GET /events/stats',
        'GET /export/datasets',
        'POST /export/single_class',
        'GET /export/single_class/status',
        'POST /export/yolo',
        'GET /images/cache/stats',
        'GET /models/status',
        'POST /pipeline/auto_label',
        'POST /probe/cancel',
        'POST /regions/cluster',
        'GET /regions/cluster/status',
        'GET /regions/clusters',
        'POST /regions/clusters/refine/{cluster_id}',
        'POST /regions/fp_centroids/build',
        'GET /regions/fp_centroids/status',
        'GET /regions/statuses',
        'GET /review/new_class_proposals/summary',
        'GET /review/raw_label_clusters',
        'GET /review/unmatched_terms',
        'POST /scores/cancel',
        'POST /scores/compute',
        'GET /scores/coverage',
        'GET /scores/status',
        'POST /select/cancel',
        'GET /select/status',
        'GET /stats/classes',
        'GET /test_holdout/stats',
        'GET /training_cohorts',
        'POST /viz/projection/cancel',
        'GET /viz/projection/status',
        'POST /vlm/label_batch',
        'POST /vlm/verify_regions',
    }
)


JUSTIFICATIONS = {
    'STREAM_OR_FILE': 'a server-sent event stream or a file/image download: not a JSON body',
    'LEGACY_DICT': (
        'a free-form dict payload from before 0.4.0 that gained no field in 0.4.0; '
        'typed route by route (this list only shrinks)'
    ),
}
ALLOWED = STREAM_OR_FILE | LEGACY_DICT


def _is_untyped(schema: dict[str, Any] | None) -> bool:
    if not schema:
        return True
    return (
        schema.get('type') == 'object'
        and not schema.get('properties')
        and 'additionalProperties' in schema
    )


def _untyped_routes() -> set[str]:
    paths = json.loads(CONTRACT.read_text())['paths']
    found = set()
    for path, operations in paths.items():
        for method, operation in operations.items():
            for status, response in operation['responses'].items():
                if not status.startswith('2') or status == '204':
                    continue
                schema = response.get('content', {}).get('application/json', {}).get('schema')
                if _is_untyped(schema):
                    found.add(f'{method.upper()} {path.removeprefix(PROJECT_PREFIX)}')
    return found


def test_every_json_response_is_typed_or_justified() -> None:
    untyped = _untyped_routes() - ALLOWED
    assert not untyped, (
        'these routes return an untyped body; give them a response model '
        f'(response_model= or responses={{200: {{"model": ...}}}}): {sorted(untyped)}'
    )


def test_allowlist_has_no_stale_entries() -> None:
    stale = ALLOWED - _untyped_routes()
    assert not stale, f'typed (or gone) since: remove from the allowlist: {sorted(stale)}'


def test_allowlist_groups_do_not_overlap() -> None:
    assert not STREAM_OR_FILE & LEGACY_DICT
    assert set(JUSTIFICATIONS) == {'STREAM_OR_FILE', 'LEGACY_DICT'}


def test_the_routes_that_gained_fields_are_typed() -> None:
    # The 0.4.0 additions the frontend reads: they must never fall back to the allowlist.
    must_be_typed = {
        'GET /search/text',
        'GET /stats/dataset',
        'GET /review/{tab}',
        'GET /review/{tab}/locate',
        'PUT /crops/batch_label',
        'POST /crops/move',
        'POST /crops/batch_exclude',
        'POST /crops/batch_unexclude',
        'PATCH /crops/{crop_id}/region_meta',
        'PUT /crops/{crop_id}/regions',
        'PUT /crops/batch_regions',
        'PATCH /crops/{crop_id}/regions/{box_id}',
        'POST /regions/batch_status',
        'POST /regions/batch_box_state',
        'GET /pipeline/auto_label/status',
        'GET /pipeline/auto_label/status/{job_id}',
        'POST /pipeline/auto_label/start',
        'POST /open_vocab/validate',
    }
    assert not must_be_typed & ALLOWED
