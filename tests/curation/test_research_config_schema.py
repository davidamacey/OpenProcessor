"""``GET .../<kind>/schema`` serves the JSON Schema of a preset body, generated from its model."""

from __future__ import annotations

import pytest
from _curation_app import SCOPED, mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import router as curation_router
from src.routers.curation.projects import global_router
from src.schemas.curation.research_config import RESEARCH_BODY_MODELS


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.include_router(global_router, prefix='/curation')
    return TestClient(app)


PATHS = {
    'serving_profile': '/curation/serving_profiles/schema',
    'cluster_profile': f'{SCOPED}/cluster_profiles/schema',
    'assist_loop_preset': f'{SCOPED}/assist_loop_presets/schema',
    'near_dup_profile': f'{SCOPED}/near_dup_profiles/schema',
}


@pytest.mark.parametrize('kind', sorted(PATHS))
def test_each_kind_serves_the_schema_of_its_body_model(client: TestClient, kind: str) -> None:
    r = client.get(PATHS[kind])
    assert r.status_code == 200, r.text
    served = r.json()
    assert served['kind'] == kind
    assert served['json_schema'] == RESEARCH_BODY_MODELS[kind].model_json_schema()


def test_the_cluster_schema_carries_enums_required_fields_and_strictness(
    client: TestClient,
) -> None:
    schema = client.get(PATHS['cluster_profile']).json()['json_schema']
    assert schema['properties']['target']['enum'] == ['item', 'region_box']
    assert schema['properties']['backend']['default'] == 'cpu'
    assert {'target', 'method', 'refresh'} <= set(schema['required'])
    assert schema['additionalProperties'] is False


def _bare_fields(schema: dict, node: dict, path: str = '') -> list[str]:
    """Paths of the properties with no description, on the field or on the model it points at."""

    def described(prop: dict) -> bool:
        refs = [prop, *prop.get('anyOf', []), *prop.get('allOf', [])]
        return any(
            'description' in n or 'description' in schema['$defs'].get(n.get('$ref', '')[8:], {})
            for n in refs
        )

    def models_of(prop: dict) -> list[dict]:
        found = []
        for n in [prop, *prop.get('anyOf', []), prop.get('items', {})]:
            if '$ref' in n:
                found.append(schema['$defs'][n['$ref'][8:]])
            elif 'items' in n and '$ref' in n['items']:
                found.append(schema['$defs'][n['items']['$ref'][8:]])
        return found

    bare = []
    for name, prop in node.get('properties', {}).items():
        here = f'{path}.{name}'
        if not described(prop):
            bare.append(here)
        for sub in models_of(prop):
            bare += _bare_fields(schema, sub, here)
    return bare


@pytest.mark.parametrize('kind', sorted(PATHS))
def test_every_field_at_every_depth_has_help_text(kind: str) -> None:
    schema = RESEARCH_BODY_MODELS[kind].model_json_schema()
    assert _bare_fields(schema, schema) == []
