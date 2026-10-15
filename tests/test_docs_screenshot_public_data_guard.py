"""The docs screenshot capture refuses to run against private data (issue #284).

Pages that render a list of every project (the project list, combine, copy-from, sharing) are
captured only when each project the backend lists is on the operator's public allowlist; the
project being captured must be on it too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'frontend' / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from docs_screenshots import public_data  # noqa: E402  # path set above


def test_allowlist_is_required_and_parsed() -> None:
    assert public_data.parse_allowlist('a, b ,,c') == {'a', 'b', 'c'}
    with pytest.raises(ValueError, match='allowlist'):
        public_data.parse_allowlist('')
    with pytest.raises(ValueError, match='allowlist'):
        public_data.parse_allowlist(None)


@pytest.mark.parametrize(
    'name',
    [
        'projects',
        'combine-wizard',
        'combine-job',
        'projects-delete-dry-run',
        'projects-copy-settings',
        'models-sharing',
        'models-unshare-force',
        'clone-from-project',
    ],
)
def test_pages_that_list_every_project_are_recognised(name: str) -> None:
    assert public_data.lists_all_projects(name)


@pytest.mark.parametrize('name', ['dashboard', 'review-regions-multibox', 'settings'])
def test_per_project_pages_are_not_flagged(name: str) -> None:
    assert not public_data.lists_all_projects(name)


def test_project_to_capture_must_be_allowlisted() -> None:
    with pytest.raises(ValueError, match='private-one'):
        public_data.require_allowlisted_project('private-one', {'sample-coco'})
    public_data.require_allowlisted_project('sample-coco', {'sample-coco'})
    with pytest.raises(ValueError, match='--project'):
        public_data.require_allowlisted_project(None, {'sample-coco'})


def test_foreign_projects_are_the_listed_slugs_outside_the_allowlist() -> None:
    payload = {'projects': [{'slug': 'default'}, {'slug': 'sample-coco'}, {'slug': 'client-x'}]}
    assert public_data.foreign_projects(payload, {'sample-coco', 'default'}) == ['client-x']
    assert public_data.foreign_projects({'projects': []}, {'a'}) == []


def test_names_to_skip_drops_listing_pages_when_a_foreign_project_exists() -> None:
    payload = {'projects': [{'slug': 'sample-coco'}, {'slug': 'client-x'}]}
    names = ['projects', 'dashboard', 'combine-wizard', 'models-sharing']
    skipped = public_data.listing_pages_to_refuse(names, payload, {'sample-coco'})
    assert skipped == ['projects', 'combine-wizard', 'models-sharing']
    assert public_data.listing_pages_to_refuse(names, payload, {'sample-coco', 'client-x'}) == []
