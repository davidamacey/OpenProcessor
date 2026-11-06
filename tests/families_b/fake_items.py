"""A small in-memory items index: ``bool``/``term``/``terms`` queries, count, update_by_query."""

from __future__ import annotations

import re
from typing import Any


def _match(doc: dict[str, Any], query: dict[str, Any]) -> bool:
    if 'term' in query:
        ((field, value),) = query['term'].items()
        return doc.get(field) == value
    if 'terms' in query:
        ((field, values),) = query['terms'].items()
        return doc.get(field) in values
    if 'bool' in query:
        spec = query['bool']
        return all(_match(doc, q) for q in spec.get('filter', [])) and not any(
            _match(doc, q) for q in spec.get('must_not', [])
        )
    raise AssertionError(f'the fake does not evaluate {query!r}')


def _apply(doc: dict[str, Any], script: dict[str, Any]) -> None:
    """Executes the ``ctx._source.x = params.y`` / ``= literal`` statements of a painless script."""
    for statement in script['source'].split(';'):
        found = re.fullmatch(r'\s*ctx\._source\.(\w+) = (.+?)\s*', statement)
        if not found:
            continue
        field, rhs = found.groups()
        if rhs.startswith('params.'):
            doc[field] = script['params'][rhs.removeprefix('params.')]
        elif rhs == 'null':
            doc[field] = None
        else:
            doc[field] = rhs.strip("'")


class _Tasks:
    def __init__(self, owner: FakeItems) -> None:
        self._owner = owner

    async def get(self, task_id: str) -> dict[str, Any]:
        return {'completed': True, 'response': self._owner.results[task_id]}


class FakeItems:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.results: dict[str, dict[str, Any]] = {}
        self.update_calls = 0
        self.tasks = _Tasks(self)

    async def count(self, index: str, body: dict[str, Any]) -> dict[str, int]:  # noqa: ARG002
        return {'count': sum(_match(d, body['query']) for d in self.docs)}

    async def update_by_query(self, index: str, body: dict[str, Any], **_: Any) -> dict[str, str]:  # noqa: ARG002
        self.update_calls += 1
        hits = [d for d in self.docs if _match(d, body['query'])]
        for doc in hits:
            _apply(doc, body['script'])
        task = f'task-{self.update_calls}'
        self.results[task] = {'updated': len(hits)}
        return {'task': task}
