"""The worked example: ``{project}`` values from ``GET /curation/projects``.

Pages through the project list of each stack (``next_cursor`` is passed back as ``cursor``), archived
projects included, and keeps the slugs both hold.

What another family's provider copies from this one: ask the stack's own list route, read the field the
wire contract names, page to the end with a cap, refuse an answer of the wrong shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from .base import Fixtures, ProviderError, present_in_both


if TYPE_CHECKING:
    from collections.abc import Iterator

LIST_PATH = '/curation/projects'
MAX_PAGES = 100


class ProjectsProvider:
    name = 'projects'

    def collect(self, reference: httpx.Client, candidate: httpx.Client) -> Fixtures:
        return present_in_both(
            'project', slugs(reference, 'reference'), slugs(candidate, 'candidate')
        )


def slugs(client: httpx.Client, stack: str) -> list[str]:
    return [slug for page in pages(client, stack) for slug in page]


def pages(client: httpx.Client, stack: str) -> Iterator[list[str]]:
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        params: dict[str, Any] = {'include_archived': 'true'}
        if cursor:
            params['cursor'] = cursor
        body = fetch(client, stack, params)
        yield [project_slug(entry, stack) for entry in body['projects']]
        cursor = body.get('next_cursor')
        if not cursor:
            return
    raise ProviderError(f'{stack}: the project list has more than {MAX_PAGES} pages')


def fetch(client: httpx.Client, stack: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = client.get(LIST_PATH, params=params)
    except httpx.HTTPError as error:
        raise ProviderError(f'{stack}: GET {LIST_PATH} failed: {error}') from error
    if response.status_code != httpx.codes.OK:
        raise ProviderError(f'{stack}: GET {LIST_PATH} answered {response.status_code}')
    try:
        body = response.json()
    except ValueError as error:
        raise ProviderError(f'{stack}: GET {LIST_PATH} did not answer JSON') from error
    if not isinstance(body, dict) or not isinstance(body.get('projects'), list):
        raise ProviderError(f'{stack}: GET {LIST_PATH} has no "projects" list')
    return body


def project_slug(entry: Any, stack: str) -> str:
    if not isinstance(entry, dict) or not isinstance(entry.get('slug'), str):
        raise ProviderError(f'{stack}: a project in the list has no string "slug"')
    return str(entry['slug'])
