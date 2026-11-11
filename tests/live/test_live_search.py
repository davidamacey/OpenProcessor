"""Live: semantic search over items the test itself ingests, with the harness's fake Triton.

The fake PE image encoder turns a picture's coarse colour layout into its vector, so a picture is
its own nearest neighbour and a differently coloured one is not. The fake text encoder is a hash
of the token ids: it cannot read a query, so the text route is checked for its contract (scope,
order, paging, refusals) and not for meaning. Routes: ``GET /search/text``, ``POST /search/image``
(upload and stored ``image_id``), ``GET /search/similar``, and the ``space`` and ``target``
parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ._ingest_support import IngestProject, count, ingest_project, picture


if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = pytest.mark.live

COLOURS = {'red': (200, 30, 30), 'green': (30, 200, 30), 'blue': (30, 30, 200)}


@pytest.fixture(scope='module')
def corpus(api: Any, opensearch: Any) -> Iterator[dict[str, Any]]:
    """Three ingested pictures; ``crops`` maps a colour to the item ids made from it."""
    with ingest_project(api, opensearch, 'search') as project:
        resp = api.post(
            f'{project.prefix}/ingest/upload',
            files=[
                ('images', (f'{name}.jpg', picture(rgb), 'image/jpeg'))
                for name, rgb in COLOURS.items()
            ],
        )
        assert resp.status_code == 200, resp.text
        results = resp.json()['results']
        assert [r['status'] for r in results] == ['success'] * 3, resp.text
        image_ids = {name: r['image_id'] for name, r in zip(COLOURS, results, strict=True)}
        assert count(opensearch, project.items_index) >= 3
        hits = opensearch.post(
            f'/{project.items_index}/_search',
            json={'size': 50, '_source': ['image_id'], 'query': {'match_all': {}}},
        ).json()['hits']['hits']
        crops: dict[str, list[str]] = {name: [] for name in COLOURS}
        for hit in hits:
            for name, image_id in image_ids.items():
                if hit['_source']['image_id'] == image_id:
                    crops[name].append(hit['_id'])
        assert all(crops.values()), crops
        yield {'project': project, 'image_ids': image_ids, 'crops': crops}


def _project(corpus: dict[str, Any]) -> IngestProject:
    project: IngestProject = corpus['project']
    return project


def _image_ids(page: dict[str, Any]) -> list[str]:
    return [item['image_id'] for item in page['items']]


def test_text_search_is_scoped_ordered_and_paged(api: Any, corpus: dict[str, Any]) -> None:
    project = _project(corpus)
    resp = api.get(f'{project.prefix}/search/text', params={'q': 'a red box', 'page_size': 50})
    assert resp.status_code == 200, resp.text
    page = resp.json()
    assert page['total'] >= 3, page
    assert set(_image_ids(page)) == set(corpus['image_ids'].values())
    scores = [item['semantic_score'] for item in page['items']]
    assert scores == sorted(scores, reverse=True)
    assert page['space'], page

    first = api.get(
        f'{project.prefix}/search/text', params={'q': 'a red box', 'page_size': 1, 'page': 1}
    ).json()
    second = api.get(
        f'{project.prefix}/search/text', params={'q': 'a red box', 'page_size': 1, 'page': 2}
    ).json()
    assert first['items'][0]['crop_id'] == page['items'][0]['crop_id']
    assert second['items'][0]['crop_id'] == page['items'][1]['crop_id']


def test_text_search_refuses_an_empty_query_and_an_unknown_space(
    api: Any, corpus: dict[str, Any]
) -> None:
    project = _project(corpus)
    assert api.get(f'{project.prefix}/search/text', params={'q': ''}).status_code == 422
    unknown = api.get(f'{project.prefix}/search/text', params={'q': 'x', 'space': 'no_such_space'})
    assert unknown.status_code in (400, 404, 409, 422), unknown.text
    assert unknown.status_code != 500


def test_image_search_finds_the_same_picture_first(api: Any, corpus: dict[str, Any]) -> None:
    project = _project(corpus)
    for name, rgb in COLOURS.items():
        resp = api.post(
            f'{project.prefix}/search/image',
            files={'image': (f'{name}.jpg', picture(rgb), 'image/jpeg')},
            params={'page_size': 10},
        )
        assert resp.status_code == 200, resp.text
        page = resp.json()
        assert _image_ids(page)[0] == corpus['image_ids'][name], (name, page['items'][:2])
        # The query is the whole picture and the stored vector is the item's crop of it, so the
        # best score is high but not 1.0; the other pictures score clearly below it.
        scores = [item['semantic_score'] for item in page['items']]
        assert scores[0] > 0.9, scores
        assert scores[0] > scores[1], scores


def test_image_search_by_a_stored_image_id_and_its_refusals(
    api: Any, corpus: dict[str, Any]
) -> None:
    project = _project(corpus)
    stored = api.post(
        f'{project.prefix}/search/image', data={'image_id': corpus['image_ids']['green']}
    )
    assert stored.status_code == 200, stored.text
    assert _image_ids(stored.json())[0] == corpus['image_ids']['green']

    missing = api.post(f'{project.prefix}/search/image', data={'image_id': 'nope'})
    assert missing.status_code == 404, missing.text
    assert missing.json()['detail']['error'] == 'image_not_found', missing.text
    unreadable = api.post(
        f'{project.prefix}/search/image',
        files={'image': ('bad.jpg', b'not a picture', 'image/jpeg')},
    )
    assert unreadable.status_code == 400, unreadable.text
    assert unreadable.json()['detail']['error'] == 'image_unreadable', unreadable.text


def test_similar_ranks_the_source_first_unless_excluded(api: Any, corpus: dict[str, Any]) -> None:
    project = _project(corpus)
    source = corpus['crops']['blue'][0]
    kept = api.get(
        f'{project.prefix}/search/similar', params={'crop_id': source, 'exclude_self': False}
    )
    assert kept.status_code == 200, kept.text
    assert kept.json()['items'][0]['crop_id'] == source
    excluded = api.get(f'{project.prefix}/search/similar', params={'crop_id': source})
    assert excluded.status_code == 200, excluded.text
    assert source not in [i['crop_id'] for i in excluded.json()['items']]

    unknown = api.get(f'{project.prefix}/search/similar', params={'crop_id': 'no_such_crop'})
    assert unknown.status_code == 404, unknown.text


def test_a_named_space_is_echoed_and_searched(api: Any, corpus: dict[str, Any]) -> None:
    project = _project(corpus)
    spaces = api.get(f'{project.prefix}/embedding/spaces')
    assert spaces.status_code == 200, spaces.text
    ready = [
        s
        for s in spaces.json()['spaces']
        if s['target'] == 'item' and s['role'] == 'primary' and s['status'] == 'ready'
    ]
    assert ready, spaces.text
    key = ready[0]['space_key']
    resp = api.get(f'{project.prefix}/search/text', params={'q': 'a red box', 'space': key})
    assert resp.status_code == 200, resp.text
    assert resp.json()['space'] == key
    similar = api.get(
        f'{project.prefix}/search/similar',
        params={'crop_id': corpus['crops']['red'][0], 'space': key},
    )
    assert similar.status_code == 200, similar.text
    assert similar.json()['space'] == key


def test_box_target_says_when_no_box_vectors_exist(api: Any, corpus: dict[str, Any]) -> None:
    """A project with no region boxes has no box vectors: the page is empty and says why."""
    project = _project(corpus)
    resp = api.get(f'{project.prefix}/search/text', params={'q': 'a red box', 'target': 'box'})
    assert resp.status_code == 200, resp.text
    page = resp.json()
    assert page['items'] == [], page
    assert page['unembedded_in_scope'] >= 3, page
