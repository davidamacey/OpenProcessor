"""Research presets: serving profiles (deployment-wide), cluster profiles, assist-loop presets and
near-duplicate profiles. Payloads are built from the vendored contract (``fill``), so a renamed
field fails the stub. The fail-closed stub means any request a page makes that a test did not
register fails the test.

  1. `test_list_and_open_every_kind` — all four kinds: list rows, the served active chip, Open.
  2. `test_edit_save_stale_409_then_keep_my_edits` — Save sends `expected_revision`; a served 409
     offers "Keep my edits" and the next Save claims the served revision.
  3. `test_a_refused_body_lists_the_served_reasons`.
  4. `test_cluster_profile_activation_is_per_axis_and_a_stale_ref_is_a_served_409`.
  5. `test_serving_profile_activation_sends_the_setting_revision`.
  6. `test_new_clone_and_delete`.
  7. `test_unknown_kind`.
  8. `test_rows_fit_430` — no page-wide horizontal scroll at 430 px, every action reachable.
  9. `test_the_editor_lists_the_served_fields_beside_the_body` — the field help is the served JSON
     Schema (type, required, choices, limits, default, words); a schema that cannot be read is said
     as served and the body stays editable.
 10. `test_a_cluster_profile_on_the_axis_of_another_target_is_a_served_422`.

The axes of a kind are served with its schema (`.../schema`), so every page of a kind reads that
first. Saving with a stale revision is the existing `test_edit_save_stale_409_then_keep_my_edits`.

Set OP_E2E_SHOT_DIR to also save full-page screenshots (1600 and 430 px).
"""

from __future__ import annotations

import copy
import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from fixtures.contract import fill
from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import shot
from harness.waits import expect_handled
from playwright.sync_api import expect


SHOT_WIDTHS = (1600, 430)

# slug, collection, kind, scope
KINDS = [
    ('serving-profiles', 'serving_profiles', 'serving_profile', 'deployment'),
    ('cluster-profiles', 'cluster_profiles', 'cluster_profile', 'project'),
    ('assist-loop-presets', 'assist_loop_presets', 'assist_loop_preset', 'project'),
    ('near-dup-profiles', 'near_dup_profiles', 'near_dup_profile', 'project'),
]
AXES = {
    'serving_profile': ['serving'],
    'cluster_profile': ['clustering_item', 'clustering_region_box'],
    'assist_loop_preset': ['assist_loop_preset'],
    'near_dup_profile': ['near_dup'],
}
AXIS_WIRE = {
    'clustering_item': ('Items', 'item'),
    'clustering_region_box': ('Region boxes', 'region_box'),
    'assist_loop_preset': ('Assist loop', None),
    'near_dup': ('Near-duplicates', None),
}
# Compact JSON Schemas in the shape pydantic writes (a $ref into $defs, enums, limits, nullable).
JSON_SCHEMAS = {
    'serving_profile': {
        'type': 'object',
        'required': ['servers'],
        'properties': {
            'servers': {
                'type': 'array',
                'description': 'The Triton servers, one per card.',
                'items': {'$ref': '#/$defs/Server'},
                'minItems': 1,
            }
        },
        '$defs': {
            'Server': {
                'type': 'object',
                'required': ['name'],
                'properties': {
                    'name': {'type': 'string', 'description': 'The name load_list refers to.'},
                    'numa_node': {
                        'anyOf': [{'type': 'integer', 'minimum': 0}, {'type': 'null'}],
                        'default': None,
                        'description': 'The NUMA node it is pinned to.',
                    },
                },
            }
        },
    },
    'cluster_profile': {
        'type': 'object',
        'required': ['target', 'method', 'refresh'],
        'properties': {
            'target': {
                'enum': ['item', 'region_box'],
                'type': 'string',
                'description': 'What is clustered; it must match the axis.',
            },
            'method': {
                'enum': ['ivf', 'minibatch_kmeans'],
                'type': 'string',
                'description': 'The clustering method.',
            },
            'backend': {
                'enum': ['cpu', 'gpu_if_validated'],
                'type': 'string',
                'default': 'cpu',
                'description': 'gpu_if_validated uses the GPU only where it was validated.',
            },
            'refresh': {'$ref': '#/$defs/Refresh', 'description': 'When a new run starts.'},
        },
        '$defs': {
            'Refresh': {
                'type': 'object',
                'required': ['max_deferral_s'],
                'properties': {
                    'max_deferral_s': {
                        'type': 'integer',
                        'minimum': 0,
                        'description': 'Longest a run waits for ingest to go quiet.',
                    }
                },
            }
        },
    },
    'assist_loop_preset': {
        'type': 'object',
        'required': ['budget'],
        'properties': {
            'on_budget': {
                'enum': ['complete', 'pause'],
                'type': 'string',
                'default': 'pause',
                'description': 'What a reached limit does.',
            },
            'budget': {'type': 'object', 'description': 'Hard caps.', 'properties': {}},
        },
    },
    'near_dup_profile': {
        'type': 'object',
        'properties': {
            'version': {'const': 1, 'type': 'integer', 'default': 1, 'description': 'Format.'},
            'item': {
                'type': 'object',
                'description': 'Cuts for items.',
                'properties': {
                    'dedup_cosine': {
                        'type': 'number',
                        'maximum': 1.0,
                        'default': 0.98,
                        'description': 'Items at or above it are duplicates.',
                    }
                },
            },
        },
    },
}
BODIES = {
    'serving_profile': {'servers': [{'name': 'gpu0'}], 'instances': {'yolo': 2}},
    'cluster_profile': {'target': 'item', 'method': 'ivf', 'level': 'class'},
    'assist_loop_preset': {'strategy': 'cluster_sampled', 'on_budget': 'pause'},
    'near_dup_profile': {'version': 1, 'item': {'dedup_cosine': 0.97, 'leak_cosine': 0.93}},
}


def doc(kind: str, name: str, revision: int = 1, **over: Any) -> dict[str, Any]:
    base = {
        'kind': kind,
        'name': name,
        'revision': revision,
        'description': f'{name} words',
        'body': copy.deepcopy(BODIES[kind]),
        'created_at': '2026-11-01T10:00:00Z',
        'updated_at': '2026-11-02T10:00:00Z',
        'updated_by': 'api',
    }
    base.update(over)
    return fill('ResearchConfigDoc', **base)


def active(axis: str, name: str | None = None, revision: int | None = None, **over: Any):
    state = 'active' if name else 'none'
    return fill(
        'ResearchActive',
        axis=axis,
        state=state,
        name=name,
        revision=revision,
        activated_at='2026-11-03T10:00:00Z' if name else None,
        **over,
    )


def serve_kind(stub: Any, slug: str, coll: str, kind: str, scope: str, state: dict[str, Any]):
    """The reads every page of one kind makes. `state` is what the write handlers update."""
    state.setdefault('docs', {'strict': doc(kind, 'strict', 2), 'loose': doc(kind, 'loose', 1)})
    state.setdefault('active', {})
    axes = [
        fill('ResearchAxis', axis=a, label=AXIS_WIRE[a][0], target=AXIS_WIRE[a][1])
        for a in AXES[kind]
        if a in AXIS_WIRE
    ]
    schema_reply = state.get('schema') or (
        200,
        fill('ResearchConfigSchema', kind=kind, axes=axes, json_schema=JSON_SCHEMAS[kind]),
    )
    stub.on('GET', rf'/{coll}/schema$', schema_reply)
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on(
        'GET',
        rf'/{coll}$',
        lambda _r, _m: (200, fill('ResearchConfigList', items=list(state['docs'].values()))),
    )
    for name in ('strict', 'loose'):
        stub.on('GET', rf'/{coll}/{name}$', lambda _r, _m, n=name: (200, state['docs'][n]))
    if scope == 'deployment':

        def serving_active(_r: Any, _m: Any):
            a = state['active'].get('serving')
            if a is None:
                return (
                    404,
                    {'detail': {'error': 'not_found', 'message': 'no serving profile is active'}},
                )
            return (200, a)

        stub.on('GET', rf'/{coll}/active$', serving_active)
    else:

        def project_active(request: Any, _m: Any):
            axis = parse_qs(urlparse(request.url).query)['axis'][0]
            return (200, state['active'].get(axis) or active(axis))

        stub.on('GET', rf'/{coll}/active$', project_active)


def open_list(page: Any, app_url: str, slug: str) -> None:
    page.goto(f'{app_url}/p/default/settings/presets/{slug}')
    page.get_by_test_id('preset-table').wait_for(timeout=ACTION_TIMEOUT_MS)


def open_editor(page: Any, app_url: str, slug: str, name: str = 'strict') -> None:
    page.goto(f'{app_url}/p/default/settings/presets/{slug}/{name}')
    page.get_by_test_id('preset-body').wait_for(timeout=ACTION_TIMEOUT_MS)


def serve_presets(stub: Any) -> dict[str, Any]:
    """The state for the a11y page list: cluster profiles with one active axis."""
    slug, coll, kind, scope = KINDS[1]
    state: dict[str, Any] = {'active': {'clustering_item': active('clustering_item', 'strict', 2)}}
    serve_kind(stub, slug, coll, kind, scope, state)
    return state


@pytest.mark.parametrize(('slug', 'coll', 'kind', 'scope'), KINDS, ids=[k[0] for k in KINDS])
def test_list_and_open_every_kind(stub, page, app_url, slug, coll, kind, scope):
    axis = AXES[kind][0]
    state: dict[str, Any] = {}
    if scope == 'deployment':
        state['active'] = {
            'serving': fill('ServingActive', name='strict', revision=1, setting_revision=4, body={})
        }
    else:
        state['active'] = {axis: active(axis, 'strict', 1)}
    serve_kind(stub, slug, coll, kind, scope, state)

    open_list(page, app_url, slug)
    rows = page.get_by_test_id('preset-row')
    expect(rows).to_have_count(2)
    strict = page.locator('[data-testid="preset-row"][data-name="strict"]')
    expect(strict.get_by_test_id('preset-active-chip')).to_have_count(1)
    expect(
        page.locator('[data-testid="preset-row"][data-name="loose"]').get_by_test_id(
            'preset-active-chip'
        )
    ).to_have_count(0)
    # saved r2 while r1 is the active one: said, not hidden
    expect(strict).to_contain_text('saved r2, active r1')
    expect(page.get_by_test_id('preset-scope')).to_have_text(
        'deployment-wide' if scope == 'deployment' else 'this project'
    )
    shot(page, f'list-{slug}', widths=SHOT_WIDTHS)

    strict.get_by_role('link', name='Open').click()
    body = page.get_by_test_id('preset-body')
    expect(body).to_have_value(json.dumps(BODIES[kind], indent=2))
    expect(page.get_by_test_id('config-meta')).to_contain_text('revision 2')
    expect(page.get_by_test_id('config-save')).to_be_disabled()
    shot(page, f'editor-{slug}', widths=SHOT_WIDTHS)


def test_edit_save_stale_409_then_keep_my_edits(stub, page, app_url):
    slug, coll, kind, scope = KINDS[3]
    state: dict[str, Any] = {}
    serve_kind(stub, slug, coll, kind, scope, state)
    puts: list[Any] = []

    def put(request: Any, _m: Any):
        body = request.post_data_json
        puts.append(body)
        if len(puts) == 1:
            return (
                409,
                {
                    'detail': {
                        'error': 'revision_conflict',
                        'message': "'strict' is at revision 4",
                        'current_revision': 4,
                    }
                },
            )
        return (200, doc(kind, 'strict', 5, body=body['body'], description=body['description']))

    stub.on('PUT', rf'/{coll}/strict$', put)
    open_editor(page, app_url, slug)

    # not JSON: nothing to save, and it says why (the only check the page makes itself)
    body = page.get_by_test_id('preset-body')
    body.fill('{"version": ')
    expect(page.get_by_test_id('preset-body-error')).to_be_visible()
    expect(page.get_by_test_id('config-save')).to_be_disabled()

    edited = {'version': 1, 'item': {'dedup_cosine': 0.99, 'leak_cosine': 0.9}}
    body.fill(json.dumps(edited))
    expect(page.get_by_test_id('preset-body-error')).to_have_count(0)
    with expect_handled(page, lambda r: r.method == 'PUT'):
        page.get_by_test_id('config-save').click()
    assert puts[0] == {
        'body': edited,
        'description': 'strict words',
        'expected_revision': 2,
    }
    conflict = page.get_by_test_id('save-conflict')
    expect(conflict).to_contain_text("'strict' is at revision 4")
    shot(page, 'editor-stale-409', widths=SHOT_WIDTHS)

    page.get_by_test_id('keep-mine').click()
    with expect_handled(page, lambda r: r.method == 'PUT'):
        page.get_by_test_id('config-save').click()
    assert puts[1]['expected_revision'] == 4
    expect(page.get_by_test_id('config-meta')).to_contain_text('revision 5')
    expect(page.get_by_test_id('save-conflict')).to_have_count(0)


def test_a_refused_body_lists_the_served_reasons(stub, page, app_url):
    slug, coll, kind, scope = KINDS[1]
    serve_kind(stub, slug, coll, kind, scope, {})
    stub.on(
        'PUT',
        rf'/{coll}/strict$',
        (
            422,
            {
                'detail': {
                    'error': 'validation_failed',
                    'message': 'invalid config body: 2 error(s)',
                    'reasons': [
                        'method: Input should be ivf or minibatch_kmeans',
                        'bogus: Extra inputs are not permitted',
                    ],
                }
            },
        ),
    )
    open_editor(page, app_url, slug)
    page.get_by_test_id('preset-body').fill('{"method": "x", "bogus": 1}')
    page.get_by_test_id('config-save').click()
    expect(page.get_by_test_id('save-error')).to_have_text('invalid config body: 2 error(s)')
    reasons = page.get_by_test_id('save-reasons').locator('li')
    expect(reasons).to_have_text(
        [
            'method: Input should be ivf or minibatch_kmeans',
            'bogus: Extra inputs are not permitted',
        ]
    )
    # the edit is kept
    expect(page.get_by_test_id('preset-body')).to_have_value('{"method": "x", "bogus": 1}')
    shot(page, 'editor-refused-body', widths=SHOT_WIDTHS)


def test_cluster_profile_activation_is_per_axis_and_a_stale_ref_is_a_served_409(
    stub, page, app_url
):
    slug, coll, kind, scope = KINDS[1]
    state: dict[str, Any] = {'active': {'clustering_item': active('clustering_item', 'loose', 1)}}
    serve_kind(stub, slug, coll, kind, scope, state)
    posts: list[Any] = []

    def activate(request: Any, _m: Any):
        body = request.post_data_json
        posts.append(body)
        if len(posts) == 1:
            return (
                409,
                {
                    'detail': {
                        'error': 'active_conflict',
                        'message': 'the active cluster_profile does not match expected_active',
                        'current': {'name': 'other', 'revision': 3},
                    }
                },
            )
        state['active']['clustering_region_box'] = active('clustering_region_box', 'strict', 2)
        return (200, state['active']['clustering_region_box'])

    stub.on('POST', rf'/{coll}/strict/activate$', activate)
    open_editor(page, app_url, slug)
    expect(page.get_by_test_id('preset-active-row')).to_have_count(2)

    page.get_by_test_id('config-activate').click()
    dialog = page.get_by_role('dialog')
    confirm = dialog.get_by_role('button', name='Activate', exact=True)
    expect(confirm).to_be_disabled()  # the target must be chosen
    dialog.get_by_test_id('activate-axis').select_option('clustering_item')
    expect(dialog.get_by_test_id('activate-from-to')).to_contain_text('loose r1')
    expect(dialog.get_by_test_id('activate-from-to')).to_contain_text('strict r2')
    with expect_handled(page, lambda r: r.method == 'POST'):
        confirm.click()
    assert posts[0] == {
        'axis': 'clustering_item',
        'revision': 2,
        'expected_active': {'name': 'loose', 'revision': 1},
    }
    expect(dialog.get_by_test_id('activate-error')).to_contain_text(
        'does not match expected_active'
    )
    shot(page, 'activate-stale-409', widths=SHOT_WIDTHS)

    dialog.get_by_test_id('activate-axis').select_option('clustering_region_box')
    expect(dialog.get_by_test_id('activate-from-to')).to_contain_text('none')
    with expect_handled(page, lambda r: r.method == 'POST'):
        confirm.click()
    assert posts[1] == {'axis': 'clustering_region_box', 'revision': 2, 'expected_active': None}
    expect(page.get_by_role('dialog')).to_have_count(0)
    expect(page.get_by_test_id('preset-active-chip')).to_contain_text('Region boxes')


def test_serving_profile_activation_sends_the_setting_revision(stub, page, app_url):
    slug, coll, kind, scope = KINDS[0]
    state: dict[str, Any] = {}
    serve_kind(stub, slug, coll, kind, scope, state)  # nothing active: the active read is a 404
    posts: list[Any] = []

    def activate(request: Any, _m: Any):
        posts.append(request.post_data_json)
        state['active']['serving'] = fill(
            'ServingActive', name='strict', revision=2, setting_revision=1, body={}
        )
        return (200, state['active']['serving'])

    stub.on('POST', rf'/{coll}/strict/activate$', activate)
    open_editor(page, app_url, slug)
    expect(page.get_by_test_id('preset-active')).to_contain_text('nothing active')
    page.get_by_test_id('config-activate').click()
    dialog = page.get_by_role('dialog')
    expect(dialog).to_contain_text('deployment-wide')
    with expect_handled(page, lambda r: r.method == 'POST'):
        dialog.get_by_role('button', name='Activate', exact=True).click()
    assert posts == [{'revision': 2, 'expected_revision': None}]
    expect(page.get_by_test_id('preset-active-chip')).to_be_visible()


def test_new_clone_and_delete(stub, page, app_url):
    slug, coll, kind, scope = KINDS[3]
    state: dict[str, Any] = {}
    serve_kind(stub, slug, coll, kind, scope, state)
    puts: list[Any] = []

    def put(request: Any, match: Any):
        puts.append((request.url.rsplit('/', 1)[-1], request.post_data_json))
        name = request.url.rsplit('/', 1)[-1]
        body = request.post_data_json
        created = doc(kind, name, 1, body=body['body'], description=body['description'])
        state['docs'][name] = created
        stub.on('GET', rf'/{coll}/{name}$', created)
        return (200, created)

    stub.on('PUT', rf'/{coll}/[^/]+$', put)
    deletes: list[str] = []

    def delete(request: Any, _m: Any):
        deletes.append(request.url.split('/curation', 1)[1])
        state['docs'].pop('loose')
        return (204, '', 'text/plain')

    stub.on('DELETE', rf'/{coll}/loose$', delete)
    open_list(page, app_url, slug)

    # New: a refusal stays in the dialog with every reason; then a body the server accepts
    page.get_by_test_id('preset-new').click()
    dialog = page.get_by_role('dialog')
    dialog.get_by_test_id('preset-new-name').fill('fresh')
    dialog.get_by_test_id('preset-new-body').fill('{"version": 1, "bogus": 2}')
    stub.on(
        'PUT',
        rf'/{coll}/fresh$',
        (
            422,
            {
                'detail': {
                    'error': 'validation_failed',
                    'message': 'invalid config body: 1 error(s)',
                    'reasons': ['bogus: Extra inputs are not permitted'],
                }
            },
        ),
    )
    dialog.get_by_role('button', name='Create').click()
    expect(dialog.get_by_test_id('preset-create-error')).to_have_text(
        'invalid config body: 1 error(s): bogus: Extra inputs are not permitted'
    )
    stub.on('PUT', rf'/{coll}/fresh$', put)
    dialog.get_by_test_id('preset-new-body').fill('{"version": 1}')
    with expect_handled(page, lambda r: r.method == 'PUT'):
        dialog.get_by_role('button', name='Create').click()
    assert puts[-1] == (
        'fresh',
        {'body': {'version': 1}, 'description': '', 'expected_revision': None},
    )
    expect(page.get_by_test_id('preset-body')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_role('heading', name='fresh')).to_be_visible()

    # Clone from the list copies the served document under a new name
    open_list(page, app_url, slug)
    page.locator('[data-testid="preset-row"][data-name="strict"]').get_by_role(
        'button', name='Clone'
    ).click()
    page.get_by_test_id('clone-name').fill('strict-copy')
    with expect_handled(page, lambda r: r.method == 'PUT'):
        page.get_by_role('dialog').get_by_role('button', name='Clone', exact=True).click()
    assert puts[-1][0] == 'strict-copy'
    assert puts[-1][1]['expected_revision'] is None
    assert puts[-1][1]['body'] == BODIES[kind]
    expect(page.get_by_test_id('preset-body')).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    # Delete names the revision the list served
    open_list(page, app_url, slug)
    page.locator('[data-testid="preset-row"][data-name="loose"]').get_by_role(
        'button', name='Delete'
    ).click()
    with expect_handled(page, lambda r: r.method == 'DELETE'):
        page.get_by_role('dialog').get_by_role('button', name='Delete', exact=True).click()
    assert deletes == [f'/projects/default/{coll}/loose?expected_revision=1']
    expect(page.locator('[data-testid="preset-row"][data-name="loose"]')).to_have_count(0)


def test_unknown_kind(stub, page, app_url):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    page.goto(f'{app_url}/p/default/settings/presets/nonsense')
    expect(page.get_by_test_id('preset-unknown-kind')).to_be_visible(timeout=ACTION_TIMEOUT_MS)


@pytest.mark.parametrize('where', ['list', 'editor'])
def test_rows_fit_430(stub, page, app_url, where):
    slug, coll, kind, scope = KINDS[1]
    state: dict[str, Any] = {'active': {'clustering_item': active('clustering_item', 'strict', 2)}}
    serve_kind(stub, slug, coll, kind, scope, state)
    page.set_viewport_size({'width': 430, 'height': 900})
    if where == 'list':
        open_list(page, app_url, slug)
    else:
        open_editor(page, app_url, slug)
    overflow = page.evaluate(
        'document.documentElement.scrollWidth - document.documentElement.clientWidth'
    )
    assert overflow <= 0, f'the page scrolls sideways by {overflow}px'
    if where == 'list':
        names = ('strict', 'loose')
        for name in names:
            link = (
                page.locator(f'[data-testid="preset-row"][data-name="{name}"]')
                .get_by_role('link')
                .first
            )
            box = link.bounding_box()
            assert box is not None and box['x'] >= 0 and box['x'] + box['width'] <= 430
        for name in names:
            # the actions sit past the table's own scroll and are reachable
            delete = page.locator(f'[data-testid="preset-row"][data-name="{name}"]').get_by_role(
                'button', name='Delete'
            )
            delete.scroll_into_view_if_needed()
            expect(delete).to_be_visible()
    else:
        box = page.get_by_test_id('config-save').bounding_box()
        assert box is not None and box['x'] >= 0 and box['x'] + box['width'] <= 430
    shot(page, f'{where}-430-rows', widths=(430,))


def test_the_editor_lists_the_served_fields_beside_the_body(stub, page, app_url):
    slug, coll, kind, scope = KINDS[1]
    serve_kind(stub, slug, coll, kind, scope, {})
    open_editor(page, app_url, slug)
    fields = page.get_by_test_id('preset-schema-fields')
    expect(fields).to_be_visible()

    def field(path: str) -> Any:
        return fields.locator(f'[data-testid="preset-schema-field"][data-path="{path}"]')

    target = field('target')
    expect(target).to_contain_text('required')
    expect(target).to_contain_text('One of item, region_box')
    expect(target).to_contain_text('What is clustered; it must match the axis.')
    expect(field('backend')).to_contain_text('Default "cpu"')
    # a nested model behind a $ref: the path is dotted, the limit is the schema's
    nested = field('refresh.max_deferral_s')
    expect(nested).to_contain_text('at least 0')
    expect(nested).to_contain_text('Longest a run waits for ingest to go quiet.')
    expect(field('refresh')).to_contain_text('When a new run starts.')
    shot(page, 'editor-field-help', widths=SHOT_WIDTHS, height=2400)


def test_a_schema_that_cannot_be_read_is_said_as_served_and_the_body_stays_editable(
    stub, page, app_url
):
    slug, coll, kind, scope = KINDS[0]
    serve_kind(
        stub,
        slug,
        coll,
        kind,
        scope,
        {
            'schema': (
                500,
                {'detail': {'error': 'internal', 'message': 'the schema could not be built'}},
            )
        },
    )
    open_editor(page, app_url, slug)
    expect(page.get_by_test_id('preset-schema-error')).to_contain_text(
        'the schema could not be built'
    )
    page.get_by_test_id('preset-body').fill('{"servers": []}')
    expect(page.get_by_test_id('config-save')).to_be_enabled()
    shot(page, 'editor-schema-error', widths=SHOT_WIDTHS, height=2400)


def test_a_cluster_profile_on_the_axis_of_another_target_is_a_served_422(stub, page, app_url):
    slug, coll, kind, scope = KINDS[1]
    serve_kind(stub, slug, coll, kind, scope, {})
    message = (
        "cluster profile 'strict' has target item, so it is activated on clustering_item, "
        'not clustering_region_box'
    )
    posts: list[Any] = []

    def activate(request: Any, _m: Any):
        posts.append(request.post_data_json)
        return (
            422,
            {
                'detail': {
                    'error': 'axis_target_mismatch',
                    'message': message,
                    'axis': 'clustering_region_box',
                }
            },
        )

    stub.on('POST', rf'/{coll}/strict/activate$', activate)
    open_editor(page, app_url, slug)
    page.get_by_test_id('config-activate').click()
    dialog = page.get_by_role('dialog')
    dialog.get_by_test_id('activate-axis').select_option('clustering_region_box')
    with expect_handled(page, lambda r: r.method == 'POST'):
        dialog.get_by_role('button', name='Activate', exact=True).click()
    assert posts[0]['axis'] == 'clustering_region_box'
    expect(dialog.get_by_test_id('activate-error')).to_have_text(message)
    # nothing was activated: the dialog stays, the chip is absent
    expect(page.get_by_test_id('preset-active-chip')).to_have_count(0)
    shot(page, 'activate-axis-mismatch', widths=SHOT_WIDTHS)
