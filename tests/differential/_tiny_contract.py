"""A three-operation contract and its families file, for runner tests that must stay fast."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

THING = {
    'type': 'object',
    'required': ['id'],
    'properties': {'id': {'type': 'integer'}, 'name': {'type': 'string'}},
}


def operation(operation_id: str, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        'operationId': operation_id,
        'responses': {
            '200': {
                'description': 'ok',
                'content': {'application/json': {'schema': THING}},
            },
            '422': {'description': 'invalid'},
        },
        **extra,
    }


def contract(*, impossible_query: bool = False) -> dict[str, Any]:
    limit = (
        {'type': 'integer', 'minimum': 5, 'maximum': 1}
        if impossible_query
        else {'type': 'integer', 'minimum': 1}
    )
    return {
        'openapi': '3.1.0',
        'info': {'title': 'tiny', 'version': '1'},
        'paths': {
            '/things': {
                'get': operation(
                    'list_things',
                    {
                        'parameters': [
                            {'name': 'limit', 'in': 'query', 'required': True, 'schema': limit}
                        ]
                    },
                ),
                'post': operation(
                    'create_thing',
                    {
                        'requestBody': {
                            'required': True,
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'object',
                                        'required': ['name'],
                                        'properties': {'name': {'type': 'string', 'minLength': 1}},
                                    }
                                }
                            },
                        }
                    },
                ),
            },
            '/things/{thing_id}': {
                'get': operation(
                    'get_thing',
                    {
                        'parameters': [
                            {
                                'name': 'thing_id',
                                'in': 'path',
                                'required': True,
                                'schema': {'type': 'string', 'minLength': 1, 'maxLength': 8},
                            }
                        ]
                    },
                ),
            },
        },
    }


FAMILIES = """
[[family]]
name = "things"
contract = "tiny"
slice = "R0"
paths = ["/things/*"]
"""
WITH_FIXTURES = FAMILIES.replace('slice = "R0"\n', 'slice = "R0"\nfixtures = ["things"]\n')


def write(
    directory: Path, *, impossible_query: bool = False, families: str = FAMILIES
) -> tuple[Path, Path]:
    """Returns the contract directory and the families file."""
    contracts = directory / 'contracts'
    contracts.mkdir()
    (contracts / 'tiny.json').write_text(json.dumps(contract(impossible_query=impossible_query)))
    families_file = directory / 'families.toml'
    families_file.write_text(families)
    return contracts, families_file
