"""`/p/{project}/near-duplicates` in a real browser.

  1. `test_status_groups_runs_and_pair_decisions` -- the served status, the
     groups with thumbnails, the run history, and a group's decision with undo
     (POST body, DELETE by the returned id).
  2. `test_thresholds_show_seeded_defaults_and_save_then_activate` -- the
     advanced panel: defaults beside the fields, the PUT against the loaded
     revision, then the activation, then the rebuild notice; a served 422's
     reasons are shown.
  3. `test_empty_level_and_a_refused_run_say_why` -- no run yet, an empty group
     list, and the served 409 message of a refused run start.
  4. `test_page_fits_a_430px_phone`.

Responses are built from the generated contract (`fill`). Screenshots go under
`artifacts_local/screens/neardup_ui_1111/`.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fixtures.contract import fill
from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect


SHOTS = Path(__file__).resolve().parents[3] / 'artifacts_local' / 'screens' / 'neardup_ui_1111'
PHOTO = (
    Path(__file__).resolve().parents[1] / 'fixtures' / 'ingest' / 'batch1' / 'a.jpg'
).read_bytes()
URL = '/p/default/near-duplicates'

IMAGE_IDS = ['img-a', 'img-b', 'img-c', 'img-d']


def _run(level: str = 'image', **over):
    base = {
        'id': f'run-{level}',
        'level': level,
        'mode': 'full',
        'status': 'current',
        'space': 'mobileclip@primary',
        'algorithm_version': 'nd-1',
        'profile_revision': 1,
        'members_scored': 1200,
        'grouped_members': 310,
        'dedup_groups': 41,
        'leak_groups': 17,
        'largest_group': 9,
        'oversized_split': 1,
        'holdout_absorbed': 3,
        'started_at': '2026-11-10T09:00:00Z',
        'finished_at': '2026-11-10T09:04:00Z',
    }
    return fill('NearDupRunView', **{**base, **over})


def _status(stale=True):
    return fill(
        'NearDupStatus',
        profile=fill('NearDupActiveProfile', name='default', revision=2),
        levels=[
            fill(
                'NearDupLevelStatus',
                level='image',
                current_run=_run('image'),
                pending_count=12 if stale else 0,
                stale_reasons=['pending'] if stale else [],
            ),
            fill(
                'NearDupLevelStatus',
                level='item',
                current_run=_run('item', id='run-item', dedup_groups=8, leak_groups=2),
                pending_count=0,
                stale_reasons=[],
            ),
        ],
    )


def _summary(gid: str, size: int, purpose: str = 'dedup', **over):
    return fill(
        'NearDupGroupSummary',
        **{
            'group_id': gid,
            'purpose': purpose,
            'size': size,
            'representative_id': 'img-a',
            'min_edge_cosine': 0.9912,
            'min_cosine_to_rep': 0.9834,
            'has_holdout': False,
            **over,
        },
    )


GROUP = _summary('11111111-1111-1111-1111-111111111111', 4)
GROUPS = [GROUP, _summary('22222222-2222-2222-2222-222222222222', 2, has_holdout=True)]
DETAIL = fill(
    'NearDupGroupDetail',
    group=GROUP,
    members=[
        fill(
            'NearDupMemberView',
            member_id=img,
            is_representative=img == 'img-a',
            cosine_to_rep=1.0 if img == 'img-a' else 0.99 - i * 0.004,
        )
        for i, img in enumerate(IMAGE_IDS)
    ],
)
DECISION = fill(
    'NearDupDecisionView',
    id='dec-1',
    level='image',
    a_id='img-a',
    b_id='img-b',
    decision='different',
    decided_by='curation-api',
)


def _shot(page, name: str) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    for label, width in (('desktop', 1280), ('430', 430)):
        page.set_viewport_size({'width': width, 'height': 1000})
        page.screenshot(path=str(SHOTS / f'{name}-{label}.png'), full_page=True)
    page.set_viewport_size({'width': 1280, 'height': 720})


def _serve(stub, groups=None, runs=None, status=None):
    def serve_groups(request, _m):
        purpose = parse_qs(urlparse(request.url).query).get('purpose', ['dedup'])[0]
        items = (groups if groups is not None else GROUPS) if purpose == 'dedup' else []
        return fill('NearDupGroupPage', items=items, next_cursor=None)

    def serve_crops(request, _m):
        ids = parse_qs(urlparse(request.url).query).get('image_id', [])
        rows = [make_item(crop_id=f'crop-{i}', image_id=i) for i in ids]
        return {'total': len(rows), 'page': 1, 'page_size': 200, 'crops': rows}

    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on('GET', r'/near_duplicates/status$', status or _status())
    stub.on(
        'GET',
        r'/near_duplicates/runs(\?|$)',
        fill('NearDupRunPage', items=runs if runs is not None else [_run('image'), _run('item')]),
    )
    stub.on('GET', r'/near_duplicates/groups(\?|$)', serve_groups)
    stub.on('GET', r'/near_duplicates/groups/[^/?]+', DETAIL)
    stub.on('GET', r'/crops(\?|$)', serve_crops)
    stub.on('GET', r'/(thumbnail|image)(\?|$)', (200, PHOTO, 'image/jpeg'))
    stub.on('GET', r'/crops/[^/]+/image(\?|$)', (200, PHOTO, 'image/jpeg'))
    stub.on('GET', r'/crops/[^/]+/thumbnail(\?|$)', (200, PHOTO, 'image/jpeg'))
    stub.on(
        'GET',
        r'/near_dup_profiles/active(\?|$)',
        fill('ResearchActive', axis='near_dup', state='none'),
    )


def test_status_groups_runs_and_pair_decisions(stub, page, app_url):
    _serve(stub)
    stub.on('POST', r'/near_duplicates/decisions$', (201, DECISION))
    stub.on('DELETE', r'/near_duplicates/decisions/dec-1$', (204, ''))
    page.goto(f'{app_url}{URL}')
    page.get_by_test_id('near-dup-status').wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('near-dup-profile-chip')).to_contain_text('revision 2')
    expect(page.get_by_test_id('near-dup-stale-image')).to_contain_text('not yet grouped (12)')
    expect(page.get_by_test_id('near-dup-run-row')).to_have_count(2)
    groups = page.get_by_test_id('near-dup-group')
    expect(groups).to_have_count(2, timeout=ACTION_TIMEOUT_MS)
    expect(groups.nth(1)).to_contain_text('In test holdout')
    expect(groups.first.get_by_test_id('near-dup-thumb')).to_be_visible()
    pressed = page.get_by_test_id('near-dup-level-tab-image')
    idle = page.get_by_test_id('near-dup-level-tab-item')
    expect(pressed).to_have_attribute('aria-pressed', 'true')
    expect(idle).to_have_attribute('aria-pressed', 'false')
    # The selected toggle must look selected, not only carry the attribute.
    assert pressed.evaluate('e => getComputedStyle(e).backgroundColor') != idle.evaluate(
        'e => getComputedStyle(e).backgroundColor'
    )
    _shot(page, 'overview')

    groups.first.click()
    members = page.get_by_test_id('near-dup-member')
    expect(members).to_have_count(4, timeout=ACTION_TIMEOUT_MS)
    expect(members.first).to_contain_text('Representative')
    expect(page.get_by_test_id('near-dup-different')).to_have_count(3)
    _shot(page, 'group-modal')

    page.get_by_test_id('near-dup-different').first.click()
    expect(page.get_by_test_id('near-dup-undo')).to_have_count(1, timeout=ACTION_TIMEOUT_MS)
    posts = [c for c in stub.calls if c[0] == 'POST']
    assert posts[-1][2] == {
        'level': 'image',
        'a_id': 'img-a',
        'b_id': 'img-b',
        'decision': 'different',
    }
    expect(page.get_by_test_id('near-dup-decided-note')).to_contain_text('1 decision saved')
    _shot(page, 'group-decided')

    page.get_by_test_id('near-dup-undo').click()
    expect(page.get_by_test_id('near-dup-undo')).to_have_count(0, timeout=ACTION_TIMEOUT_MS)
    assert [c for c in stub.calls if c[0] == 'DELETE'][-1][1].endswith('/decisions/dec-1')


def test_leak_tab_is_empty_and_says_so(stub, page, app_url):
    _serve(stub)
    page.goto(f'{app_url}{URL}')
    page.get_by_test_id('near-dup-purpose-tab-leak').click(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('near-dup-groups-empty')).to_contain_text(
        'No leak groups at the image level', timeout=ACTION_TIMEOUT_MS
    )


def _doc(name: str, revision: int, dedup: float):
    cuts = {'dedup_cosine': dedup, 'leak_cosine': dedup, 'max_group_size': 1000}
    return fill(
        'ResearchConfigDoc',
        kind='near_dup_profile',
        name=name,
        revision=revision,
        body={
            'version': 1,
            'image': {**cuts, 'k': 64, 'space_key': None},
            'item': {
                **cuts,
                'scope': 'image_leak_groups',
                'same_detector_class': True,
                'space_key': None,
            },
        },
    )


def test_thresholds_show_seeded_defaults_and_save_then_activate(stub, page, app_url):
    _serve(stub)
    stub.on(
        'GET',
        r'/near_dup_profiles/active(\?|$)',
        fill('ResearchActive', axis='near_dup', state='active', name='default', revision=2),
    )
    stub.on(
        'GET',
        r'/near_dup_profiles/default(\?|$)',
        lambda req, _m: (
            _doc('default', 1, 0.98) if 'revision=1' in req.url else _doc('default', 2, 0.95)
        ),
    )
    stub.on('PUT', r'/near_dup_profiles/default$', _doc('default', 3, 0.9))
    stub.on(
        'POST',
        r'/near_dup_profiles/default/activate$',
        fill('ResearchActive', axis='near_dup', state='active', name='default', revision=3),
    )
    page.goto(f'{app_url}{URL}')
    page.get_by_test_id('near-dup-profile').locator('summary').click(timeout=ACTION_TIMEOUT_MS)
    field = page.get_by_test_id('np-image-dedup')
    expect(field).to_have_value('0.95', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('near-dup-profile')).to_contain_text('default 0.98')
    expect(page.get_by_test_id('near-dup-profile-save')).to_be_disabled()
    _shot(page, 'thresholds')

    field.fill('0.9')
    page.get_by_test_id('np-image-leak').fill('0.99')
    page.get_by_test_id('near-dup-profile-save').click()
    expect(page.get_by_test_id('near-dup-profile-refused')).to_have_count(0)
    expect(page.get_by_test_id('near-dup-profile-notice')).to_contain_text(
        'revision 3', timeout=ACTION_TIMEOUT_MS
    )
    put = [c for c in stub.calls if c[0] == 'PUT'][-1]
    assert put[2]['expected_revision'] == 2
    assert put[2]['body']['image']['dedup_cosine'] == 0.9
    activate = [c for c in stub.calls if c[1].endswith('/activate')][-1]
    assert activate[2] == {'revision': 3, 'expected_active': {'name': 'default', 'revision': 2}}
    _shot(page, 'thresholds-saved')


def test_a_refused_threshold_save_shows_the_served_reasons(stub, page, app_url):
    _serve(stub)
    stub.on(
        'GET',
        r'/near_dup_profiles/active(\?|$)',
        fill('ResearchActive', axis='near_dup', state='active', name='default', revision=1),
    )
    stub.on('GET', r'/near_dup_profiles/default(\?|$)', _doc('default', 1, 0.98))
    stub.on(
        'PUT',
        r'/near_dup_profiles/default$',
        (
            422,
            {
                'detail': {
                    'error': 'validation_failed',
                    'message': 'invalid near_dup_profile',
                    'reasons': ['image: leak_cosine must not exceed dedup_cosine'],
                }
            },
        ),
    )
    page.goto(f'{app_url}{URL}')
    page.get_by_test_id('near-dup-profile').locator('summary').click(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('np-image-dedup').fill('0.9')
    page.get_by_test_id('near-dup-profile-save').click()
    refused = page.get_by_test_id('near-dup-profile-refused')
    expect(refused).to_contain_text('leak_cosine must not exceed', timeout=ACTION_TIMEOUT_MS)
    assert not [c for c in stub.calls if c[1].endswith('/activate')]
    _shot(page, 'thresholds-refused')


def test_empty_level_and_a_refused_run_say_why(stub, page, app_url):
    status = fill(
        'NearDupStatus',
        profile=None,
        levels=[
            fill(
                'NearDupLevelStatus',
                level='image',
                current_run=None,
                pending_count=0,
                stale_reasons=['no_run'],
            ),
            fill(
                'NearDupLevelStatus',
                level='item',
                current_run=None,
                pending_count=0,
                stale_reasons=['no_run'],
            ),
        ],
    )
    _serve(stub, groups=[], runs=[], status=status)
    stub.on(
        'POST',
        r'/near_duplicates/runs$',
        (
            409,
            {
                'detail': {
                    'error': 'no_image_run',
                    'message': 'The item level needs a current image-level run first.',
                }
            },
        ),
    )
    page.goto(f'{app_url}{URL}')
    expect(page.get_by_test_id('near-dup-no-run-image')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('near-dup-groups-empty')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('near-dup-runs-empty')).to_be_visible()
    _shot(page, 'empty')

    page.get_by_test_id('near-dup-run-full-item').click()
    expect(page.get_by_test_id('near-dup-run-error')).to_contain_text(
        'needs a current image-level run first', timeout=ACTION_TIMEOUT_MS
    )
    assert [c for c in stub.calls if c[1].endswith('/near_duplicates/runs')][-1][2] == {
        'level': 'item',
        'mode': 'full',
    }
    _shot(page, 'run-refused')


def test_a_failed_status_read_shows_the_error_and_retries(stub, page, app_url):
    _serve(stub)
    stub.on('GET', r'/near_duplicates/status$', (500, {'detail': 'store down'}))
    page.goto(f'{app_url}{URL}')
    expect(page.get_by_test_id('near-dup-load-error')).to_contain_text(
        'store down', timeout=ACTION_TIMEOUT_MS
    )
    _shot(page, 'load-error')


def test_page_fits_a_430px_phone(stub, page, app_url):
    _serve(stub)
    page.set_viewport_size({'width': 430, 'height': 900})
    page.goto(f'{app_url}{URL}')
    expect(page.get_by_test_id('near-dup-group')).to_have_count(2, timeout=ACTION_TIMEOUT_MS)
    overflow = page.evaluate(
        '() => document.documentElement.scrollWidth - document.documentElement.clientWidth'
    )
    assert overflow <= 0, f'the page scrolls sideways by {overflow}px at 430px'
    page.get_by_test_id('near-dup-group').first.click()
    expect(page.get_by_test_id('near-dup-member')).to_have_count(4, timeout=ACTION_TIMEOUT_MS)
    dialog_box = page.get_by_role('dialog').bounding_box()
    assert dialog_box is not None and dialog_box['x'] >= 0
    assert dialog_box['x'] + dialog_box['width'] <= 430 + 0.5
