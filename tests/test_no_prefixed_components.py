"""No OpenAPI component carries a module-path prefix.

FastAPI prefixes a component with its module path (``src__routers__...__Name``) when two
models share a class name; the generated TypeScript and every client then see an unreadable
name. Give each model a distinct name instead.
"""

from __future__ import annotations

import json
from pathlib import Path


CONTRACTS = Path(__file__).resolve().parents[1] / 'contracts' / 'openapi'


def test_no_component_name_is_module_prefixed() -> None:
    prefixed = {
        f'{path.name}:{name}'
        for path in sorted(CONTRACTS.glob('*.json'))
        for name in json.loads(path.read_text()).get('components', {}).get('schemas', {})
        if name.startswith('src__')
    }
    assert not prefixed, sorted(prefixed)
