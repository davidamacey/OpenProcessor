"""Response builders generated from the vendored OpenAPI contract.

``fill(schema_name, **over)`` returns every property of a component schema in
``contracts/openapi/curation.json`` with a type-correct default, then applies
the overrides. An override key the schema does not have raises, and so does a
schema name the contract no longer defines, so a backend rename fails the stub
instead of leaving it speaking the old wire. Named helpers below cover the
shapes the label-operation, job and cluster specs share.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


_PATH = Path(__file__).resolve().parents[3] / 'contracts' / 'openapi' / 'curation.json'
_SCHEMAS: dict[str, Any] = json.loads(_PATH.read_text())['components']['schemas']

_ISO = '2026-01-01T00:00:00Z'

# Factories, so each filled body gets its own list/dict.
_SCALAR_DEFAULTS: dict[str, Callable[[], Any]] = {
    'integer': int,
    'number': float,
    'boolean': bool,
    'array': list,
    'object': dict,
}


def _default(spec: dict[str, Any]) -> Any:
    if 'default' in spec:
        return spec['default']
    if 'const' in spec:
        return spec['const']
    if '$ref' in spec:
        target = _SCHEMAS[spec['$ref'].rsplit('/', 1)[-1]]
        return fill(spec['$ref'].rsplit('/', 1)[-1]) if 'properties' in target else _default(target)
    if 'enum' in spec:
        return spec['enum'][0]
    for branch in spec.get('anyOf', []):
        if branch.get('type') == 'null':
            return None
    kind = spec.get('type')
    if kind == 'string':
        return _ISO if spec.get('format') == 'date-time' else ''
    factory = _SCALAR_DEFAULTS.get(kind)
    return factory() if factory else None


def fill(schema_name: str, **over: Any) -> dict[str, Any]:
    props: dict[str, Any] = _SCHEMAS[schema_name]['properties']
    unknown = set(over) - set(props)
    if unknown:
        raise KeyError(f'{schema_name} has no field(s) {sorted(unknown)} in the contract')
    out = {name: _default(spec) for name, spec in props.items()}
    out.update(over)
    return out


def label_op_accepted(op_id: str = 'op-1', job_id: str = 'job-1') -> dict[str, Any]:
    return fill('LabelOpAccepted', op_id=op_id, job_id=job_id)


def job(job_id: str = 'job-1', status: str = 'completed', **over: Any) -> dict[str, Any]:
    """One ``GET /jobs/{id}`` body. ``result`` is the job's own result document."""
    return fill(
        'JobView',
        id=job_id,
        kind='label_write',
        project='default',
        status=status,
        queue='default',
        scope='project',
        requested_by='e2e',
        **over,
    )


def undo_response(op_id: str = 'op-1', **over: Any) -> dict[str, Any]:
    """The ``LabelOpUndoResponse`` a finished ``label_undo`` job carries as its result."""
    return fill('LabelOpUndoResponse', op_id=op_id, status='undone', **over)


def cluster_card(uid: str = 'cl-fixture-0001', **over: Any) -> dict[str, Any]:
    base = {
        'size': 2,
        'labelled_count': 2,
        'dominant_class_name': 'ducati',
        'dominant_count': 2,
        'label_agreement': 1.0,
        'labelled_share': 1.0,
        'validation_state': 'none',
        'review_counts': {'all': 0},
        'representatives': [fill('ClusterRepresentative', crop_id='crop-0', class_name='ducati')],
    }
    base.update(over)
    return fill('ClusterCard', cluster_uid=uid, **base)


def cluster_list(*cards: dict[str, Any], **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        'items': list(cards),
        'total': len(cards),
        'representatives_limit': 50,
        'sort': 'size_desc',
    }
    base.update(over)
    return fill('ClusterListResponse', **base)


def cluster_facets(uid: str, **over: Any) -> dict[str, Any]:
    return fill('ClusterFacets', cluster_uid=uid, **over)


def item_cluster(uid: str, **over: Any) -> dict[str, Any]:
    """An item's nested ``cluster`` (``ClusterRef``): the serving partition's assignment."""
    base = {
        'partition_id': 'part-fixture-1',
        'state': 'assigned',
        'score': 0.9,
        'margin': 0.1,
        'assigned_by': 'run',
    }
    base.update(over)
    return fill('ClusterRef', uid=uid, **base)


def auto_label_job(status: str = 'idle', **over: Any) -> dict[str, Any]:
    """``GET /pipeline/auto_label/status`` and the 202 of ``/start`` (``AutoLabelJobState``)."""
    base = {'job_id': 'job-0', 'pipeline': 'auto_label', 'queue_max_age_seconds': 3600.0}
    base.update(over)
    return fill('AutoLabelJobState', status=status, **base)


def embedding_space(space_key: str = 'pe_core', **over: Any) -> dict[str, Any]:
    base = {'role': 'primary', 'status': 'ready', 'model_key': 'pe-core', 'model_version': '1'}
    base.update(over)
    return fill('EmbeddingSpaceView', space_key=space_key, **base)


def embedding_spaces(*spaces: dict[str, Any]) -> dict[str, Any]:
    return fill('EmbeddingSpaceListResponse', spaces=list(spaces) or [embedding_space()])
