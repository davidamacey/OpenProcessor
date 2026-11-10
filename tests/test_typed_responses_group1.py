"""Routes typed in lane typed-responses-1 name their response model in the OpenAPI contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


CONTRACT = Path(__file__).resolve().parents[1] / 'contracts' / 'openapi' / 'curation.json'
PREFIX = '/curation/projects/{project}'

TYPED = [
    ('get', '/models/status', 'ModelsStatusResponse'),
    ('get', '/clusters/representatives', 'ClusterRepresentativesPage'),
    ('post', '/probe/cancel', 'ProbeCancelResponse'),
    ('get', '/test_holdout/stats', 'HoldoutStats'),
    ('get', '/training_cohorts', 'TrainingCohortList'),
    ('get', '/class_sources', 'ClassSourceList'),
    ('post', '/classes', 'ClassCreated'),
    ('put', '/classes/{class_id}', 'RegistryClassEntry'),
]


@pytest.mark.parametrize(('method', 'path', 'model'), TYPED)
def test_response_schema_names_the_model(method: str, path: str, model: str) -> None:
    doc = json.loads(CONTRACT.read_text())
    operation = doc['paths'][PREFIX + path][method]
    status = next(s for s in operation['responses'] if s.startswith('2'))
    schema = operation['responses'][status]['content']['application/json']['schema']
    ref = schema['$ref'].rsplit('/', 1)[1]
    assert ref == model
    component = doc['components']['schemas'][ref]
    assert component.get('properties')


def test_holdout_stats_keeps_every_bucket_key() -> None:
    from src.schemas.curation.holdout import HoldoutStats

    body = {
        'total': 3,
        'by_class': [{'key': 2, 'doc_count': 3, 'deficient': True, 'extra_agg': 1}],
        'min_test_per_class': 20,
    }
    assert HoldoutStats(**body).model_dump() == body


def test_models_status_rows_keep_exactly_their_keys() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.schemas.curation.models_status import ModelsStatusResponse

    rows = [
        {
            'name': 'seg',
            'friendly_name': 'Segmenter',
            'role': 'r',
            'kind': 'external',
            'model_type': 't',
            'status': 'ready',
            'version': None,
            'last_error': None,
            'unloadable': False,
            'future_key': {'a': 1},
        },
        {
            'name': 'm',
            'friendly_name': 'M',
            'role': 'r',
            'kind': 'triton',
            'model_type': 't',
            'status': 'ready',
            'version': '1',
            'class_mapping': {'mapped_count': 1, 'unmapped': ['x']},
            'sharing_revision': None,
        },
    ]
    app = FastAPI()

    @app.get('/s', response_model=ModelsStatusResponse, response_model_exclude_unset=True)
    async def s() -> dict[str, object]:
        return {'models': rows}

    assert TestClient(app).get('/s').json() == {'models': rows}


def test_registry_class_entry_has_one_unprefixed_name() -> None:
    # Two models sharing a class name make FastAPI emit module-path names, which leak into TS types.
    names = json.loads(CONTRACT.read_text())['components']['schemas']
    assert 'RegistryClassEntry' in names
    assert not [n for n in names if n.endswith('__RegistryClassEntry')]


def test_class_merge_documents_dry_run_and_job() -> None:
    operation = json.loads(CONTRACT.read_text())['paths'][PREFIX + '/classes/merge']['post']
    refs = {
        code: operation['responses'][code]['content']['application/json']['schema']['$ref']
        for code in ('200', '202')
    }
    assert refs['200'].endswith('/ClassMergeDryRun')
    assert refs['202'].endswith('/JobView')
