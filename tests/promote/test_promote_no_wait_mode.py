"""D-22b: `wait=true` is removed from the promote route; a promote is always a job."""

from __future__ import annotations


def test_the_promote_route_has_no_wait_parameter() -> None:
    from src.app.factory import create_app

    schema = create_app().openapi()
    route = schema['paths']['/curation/projects/{project}/train/promote/{job_id}']['post']
    names = {p['name'] for p in route.get('parameters', [])}
    assert 'wait' not in names
    assert 'force' in names  # control: the other query parameter is still there
    assert set(route['responses']) >= {'202', '409', '422'}
    assert '200' not in route['responses']


def test_the_status_poll_answers_a_job_id() -> None:
    from src.app.factory import create_app

    schema = create_app().openapi()
    route = schema['paths']['/curation/projects/{project}/train/promote/{job_id}/jobs/{promote_id}']
    [promote_id] = [p for p in route['get']['parameters'] if p['name'] == 'promote_id']
    assert promote_id['schema']['format'] == 'uuid'
