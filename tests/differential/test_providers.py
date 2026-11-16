"""Fixtures providers: values present in both stacks, drawn over HTTP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from ._fake_stack import FakeStack, json_answer
from .providers import PROVIDERS, ProviderError, present_in_both
from .providers.projects import MAX_PAGES, ProjectsProvider


if TYPE_CHECKING:
    from collections.abc import Callable


def project_list(*pages: list[str]) -> Callable[[httpx.Request], httpx.Response]:
    """A /curation/projects that serves ``pages``, each reached by the cursor the one before gave."""

    def answer(request: httpx.Request) -> httpx.Response:
        assert request.url.path == '/curation/projects'
        assert request.url.params['include_archived'] == 'true'
        index = int(request.url.params.get('cursor', '0'))
        more = index + 1 < len(pages)
        return json_answer(
            {
                'projects': [{'slug': slug, 'status': 'active'} for slug in pages[index]],
                'next_cursor': str(index + 1) if more else None,
            }
        )

    return answer


def collect(reference: FakeStack, candidate: FakeStack):
    return ProjectsProvider().collect(reference.client(), candidate.client())


def test_the_projects_provider_is_registered_under_the_name_a_family_lists() -> None:
    assert isinstance(PROVIDERS['projects'], ProjectsProvider)


def test_values_in_both_stacks_are_kept_sorted_and_the_rest_become_notes() -> None:
    fixtures = collect(
        FakeStack('reference', project_list(['default', 'beta', 'only-ref'])),
        FakeStack('candidate', project_list(['beta', 'default', 'only-cand'])),
    )
    assert fixtures.values == {'project': ['beta', 'default']}
    assert any("'only-ref' is only in the reference" in n for n in fixtures.notes)
    assert any("'only-cand' is only in the candidate" in n for n in fixtures.notes)


def test_every_page_is_read_by_following_the_cursor() -> None:
    reference = FakeStack('reference', project_list(['a'], ['b'], ['c']))
    candidate = FakeStack('candidate', project_list(['c', 'a', 'b']))
    assert collect(reference, candidate).values == {'project': ['a', 'b', 'c']}
    assert len(reference.seen) == 3


def test_two_stacks_with_nothing_in_common_give_an_empty_list_not_an_error() -> None:
    fixtures = collect(
        FakeStack('reference', project_list(['x'])), FakeStack('candidate', project_list(['y']))
    )
    assert fixtures.values == {'project': []}


@pytest.mark.parametrize(
    ('answer', 'message'),
    [
        (lambda _r: json_answer({}, 500), 'answered 500'),
        (lambda _r: json_answer(None, raw='not json'), 'did not answer JSON'),
        (lambda _r: json_answer({'items': []}), 'no "projects" list'),
        (lambda _r: json_answer({'projects': [{'name': 'x'}]}), 'no string "slug"'),
    ],
)
def test_a_stack_that_answers_wrongly_is_an_error_not_an_empty_list(
    answer: Callable[[httpx.Request], httpx.Response], message: str
) -> None:
    with pytest.raises(ProviderError, match=message):
        collect(FakeStack('reference', answer), FakeStack('candidate', project_list(['a'])))


def test_an_unreachable_stack_is_an_error() -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refused')

    with pytest.raises(ProviderError, match='refused'):
        collect(FakeStack('reference', project_list(['a'])), FakeStack('candidate', refuse))


def test_a_list_that_never_ends_stops_at_the_page_cap() -> None:
    def endless(_request: httpx.Request) -> httpx.Response:
        return json_answer({'projects': [], 'next_cursor': 'again'})

    with pytest.raises(ProviderError, match=f'more than {MAX_PAGES} pages'):
        collect(FakeStack('reference', endless), FakeStack('candidate', project_list(['a'])))


def test_present_in_both_is_the_whole_contract_of_a_provider() -> None:
    fixtures = present_in_both('thing_id', ['b', 'a', 'z'], ['a', 'b', 'q'])
    assert fixtures.values == {'thing_id': ['a', 'b']}
    assert len(fixtures.notes) == 2
