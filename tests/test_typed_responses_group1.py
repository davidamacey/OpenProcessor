"""Routes typed in lane typed-responses-1 name their response model in the OpenAPI contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


CONTRACT = Path(__file__).resolve().parents[1] / 'contracts' / 'openapi' / 'curation.json'
PREFIX = '/curation/projects/{project}'

TYPED = [
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
    # pydantic prefixes the module path when two models share a name
    assert ref == model or ref.endswith(f'__{model}')
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
