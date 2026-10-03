"""The curation wire is fully typed and consistent: every served filter has a spec,
class identity is by name, and the responses that gained fields declare them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args
from unittest.mock import AsyncMock

import pytest

from src.routers.curation._stats_models import DatasetStatsResponse
from src.routers.curation.stats import stats_dataset
from src.services.curation import review_queries
from src.services.curation.embedding_state import EmbeddingState
from src.services.curation.item_filter import Origin, ReviewStatus
from src.services.curation.review_filter_specs import EMBEDDING_STATE_LABELS, FILTER_SPECS
from src.services.curation.stats_embedding import (
    UNKNOWN_STATE_BUCKET,
    EmbeddingByState,
    embedding_summary,
)


CONTRACT = Path(__file__).resolve().parents[2] / 'contracts' / 'openapi' / 'curation.json'
PROJECT = '/curation/projects/{project}'


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def _query_params(path: str) -> set[str]:
    operation = _contract()['paths'][f'{PROJECT}{path}']['get']
    return {p['name'] for p in operation['parameters'] if p['in'] == 'query'}


# --- review tabs: one spec per served filter ---------------------------------


@pytest.mark.parametrize('tab', review_queries.KNOWN_TABS)
def test_every_filter_a_tab_serves_has_a_spec(tab: str) -> None:
    filters = set(review_queries.tab_filters(tab))
    assert filters <= set(FILTER_SPECS)
    served = {t['id']: t for t in review_queries.review_tab_catalog()}
    if tab in served:  # the regions tab is absent without a region profile
        assert {s['param'] for s in served[tab]['filter_specs']} == filters


def test_no_spec_is_orphaned() -> None:
    served = set(review_queries.COMMON_FILTERS) | {
        f for extra in review_queries.TAB_EXTRA_FILTERS.values() for f in extra
    }
    assert set(FILTER_SPECS) == served


def test_multi_valued_enums_publish_exactly_the_wire_values_with_labels() -> None:
    for param, literal in (
        ('origin', Origin),
        ('embedding_state', EmbeddingState),
        ('review_status', ReviewStatus),
    ):
        spec = FILTER_SPECS[param]
        assert spec['kind'] == 'multi_enum'
        assert [o['value'] for o in spec['options']] == list(get_args(literal))
        assert all(o['label'] for o in spec['options'])
    assert set(EMBEDDING_STATE_LABELS) == set(get_args(EmbeddingState))


def test_numeric_filters_carry_their_range() -> None:
    for param in ('conf_min', 'conf_max', 'min_area', 'max_area'):
        assert (FILTER_SPECS[param]['min'], FILTER_SPECS[param]['max']) == (0.0, 1.0)
    assert FILTER_SPECS['max_rank']['kind'] == 'integer'


def test_class_name_filters_say_where_the_names_come_from() -> None:
    for param in ('class_name', 'exclude_class_name'):
        spec = FILTER_SPECS[param]
        assert spec['kind'] == 'class_names'
        assert 'GET /classes' in spec['description']
        assert 'detector' in spec['description']


# --- class identity is by name -----------------------------------------------


@pytest.mark.parametrize(
    'path',
    ['/review/{tab}', '/review/{tab}/locate', '/search/text', '/crops', '/regions', '/clusters'],
)
def test_item_filter_routes_take_class_name_not_class_id(path: str) -> None:
    params = _query_params(path)
    assert 'class_name' in params
    assert 'class_id' not in params


def test_the_review_tab_catalog_no_longer_lists_class_id() -> None:
    assert 'class_id' not in review_queries.COMMON_FILTERS


# --- dataset stats ------------------------------------------------------------


def test_the_legacy_unknown_state_is_a_typed_documented_key() -> None:
    assert 'unknown' in EmbeddingByState.model_fields
    assert EmbeddingByState.model_fields['unknown'].description
    schema = _contract()['components']['schemas']['EmbeddingByState']
    assert set(schema['properties']) == {
        'embedded',
        'not_selected',
        'deferred',
        'failed',
        'unknown',
    }
    assert 'embedding_state' in schema['properties']['unknown']['description']


def test_by_state_always_carries_every_key() -> None:
    summary = embedding_summary({'embedded_items': {'doc_count': 1}}, 1)
    assert summary['by_state'] == {
        'embedded': 0,
        'not_selected': 0,
        'deferred': 0,
        'failed': 0,
        'unknown': 0,
    }


def test_a_state_outside_the_vocabulary_counts_as_unknown_instead_of_vanishing() -> None:
    aggs = {
        'embedding_states': {
            'buckets': [
                {'key': UNKNOWN_STATE_BUCKET, 'doc_count': 2},
                {'key': 'some_future_state', 'doc_count': 3},
            ]
        }
    }
    assert embedding_summary(aggs, 5)['by_state']['unknown'] == 5


@pytest.mark.asyncio
async def test_the_stats_model_names_every_key_the_route_emits() -> None:
    """A key the model lacks would be a field clients cannot see in the contract."""
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value={'hits': {'total': {'value': 0}}, 'aggregations': {}})
    payload = await stats_dataset(os_client)
    assert DatasetStatsResponse.model_validate(payload).model_dump(mode='json') == payload


# --- selection writes return their ids ---------------------------------------


def test_the_selection_writes_publish_updated_ids() -> None:
    schemas = _contract()['components']['schemas']
    for name in ('BatchRelabelResponse', 'BatchExcludeResponse', 'BatchUnexcludeResponse'):
        assert 'updated_ids' in schemas[name]['properties']


# --- reprocess detail ---------------------------------------------------------


def test_the_scope_detail_accepts_every_value_type_it_carries() -> None:
    from src.services.curation.reprocess_models import ReprocessScopeResult

    result = ReprocessScopeResult(
        scope='open_vocab',
        detail={
            'merged': 3,
            'estimated_minutes': 1.5,
            'segmenter_reachable': True,
            'note': 'x',
        },
    )
    assert result.detail['estimated_minutes'] == 1.5
    assert result.detail['segmenter_reachable'] is True
    assert result.model_dump()['detail']['segmenter_reachable'] is True


# --- error bodies are published -----------------------------------------------


@pytest.mark.parametrize(
    ('method', 'path', 'status', 'code'),
    [
        ('get', '/region_stage', '409', 'no_active_profile'),
        ('post', '/region_stage/pause', '409', 'no_active_profile'),
        ('post', '/classes/seed_from_detector', '503', 'detector_unavailable'),
        ('post', '/classes/seed_from_detector', '422', 'unknown_detector_names'),
        ('put', '/ingest/policy', '422', 'detector_not_servable'),
        ('put', '/ingest/policy', '503', 'detector_unavailable'),
    ],
)
def test_the_error_bodies_are_typed_in_the_contract(
    method: str, path: str, status: str, code: str
) -> None:
    contract = _contract()
    response = contract['paths'][f'{PROJECT}{path}'][method]['responses'][status]
    ref = response['content']['application/json']['schema']['$ref']
    detail_ref = contract['components']['schemas'][ref.rsplit('/', 1)[1]]['properties']['detail']
    detail = contract['components']['schemas'][detail_ref['$ref'].rsplit('/', 1)[1]]
    assert detail['properties']['error']['const'] == code or detail['properties']['error'].get(
        'enum'
    ) == [code]
    assert 'message' in detail['properties']


def test_the_policy_422_publishes_reasons_as_strings() -> None:
    schemas = _contract()['components']['schemas']
    assert schemas['DetectorNotServableDetail']['properties']['reasons'] == {
        'items': {'type': 'string'},
        'title': 'Reasons',
        'type': 'array',
    }


# --- ordered views hand back the embed request --------------------------------


def test_an_ordering_with_nothing_unranked_suggests_nothing() -> None:
    from src.services.curation.crop_orders import _embed_suggestion
    from src.services.curation.item_filter import ItemFilter

    assert _embed_suggestion(ItemFilter(), 0) is None
    request = _embed_suggestion(ItemFilter(cluster_id=7), 2)
    assert request is not None
    assert request['targets']['filter']['cluster_id'] == 7


def test_the_crops_page_contract_publishes_the_suggestion() -> None:
    page = _contract()['components']['schemas']['CropsPageResponse']['properties']
    assert 'suggested_reprocess' in page
    tabs = _contract()['components']['schemas']['ReviewEmptyState']['properties']
    assert {'has_unembedded_items', 'suggested_reprocess'} <= set(tabs)


def test_counters_accumulate_beside_non_counter_detail() -> None:
    from src.services.curation.reprocess_models import ReprocessScopeResult

    result = ReprocessScopeResult(scope='embed', detail={'estimated_vector_kb': 2.5})
    result.add_count('items', 2)
    result.add_count('items', 3)
    assert result.detail == {'estimated_vector_kb': 2.5, 'items': 5}
