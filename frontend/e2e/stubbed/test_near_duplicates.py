"""Settings › Near-duplicate profiles: the near-duplicate sections under the profile list.

  1. `test_the_explainer_runs_pinned_pairs_and_leak_groups` -- the plain-words explanation, the
     served status and run history, the pinned pairs, and the object-level leak groups, each row
     linking to Detections on its `near_dup` scope.
  2. `test_the_pinned_pairs_list_undoes_a_pin` -- the DELETE by the served id.
  3. `test_empty_level_and_a_refused_run_say_why` -- no run yet, an empty group list, and the
     served 409 message of a refused full rebuild.
  4. `test_a_failed_status_read_shows_the_error`, `test_page_fits_a_430px_phone`.
  5. `test_label_the_whole_group_uses_the_group_filter` -- a representative's details open its
     group in the Workspace, and "Label whole duplicate group" labels `{filter: {near_dup_group}}`
     after the server's dry run.

The thresholds are the profile documents above the sections (see test_presets.py); the group
browser and the same/different decisions are in the Workspace. Responses are built from the
generated contract (`fill`). Screenshots go under `artifacts_local/screens/uxr-d/`.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fixtures.contract import fill
from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import save_screens
from playwright.sync_api import expect
from test_gallery_view import serve as serve_gallery
from test_presets import serve_kind


PHOTO = (
    Path(__file__).resolve().parents[1] / 'fixtures' / 'ingest' / 'batch1' / 'a.jpg'
).read_bytes()
URL = '/p/default/settings/presets/near-dup-profiles'

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
            'representative_thumbnail_url': '/curation/projects/default/images/img-a/thumbnail',
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
            thumbnail_url=f'/curation/projects/default/images/{img}/thumbnail',
            image_id=img,
            crop_id=None,
            class_name='van' if img == 'img-c' else None,
            in_holdout=img == 'img-d',
            decision=None,
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
    decided_at='2026-11-11T09:00:00Z',
    a_thumbnail_url='/curation/projects/default/images/img-a/thumbnail',
    b_thumbnail_url='/curation/projects/default/images/img-b/thumbnail',
)


def _detail(pins: dict[str, dict]):
    """The group as the server serves it: each member carries its pin to the representative."""
    return {
        **DETAIL,
        'members': [{**m, 'decision': pins.get(m['member_id'])} for m in DETAIL['members']],
    }


def _serve(stub, groups=None, runs=None, status=None, pinned=False):
    pins: dict[str, dict] = {'img-b': {'id': DECISION['id'], 'kind': 'different'}} if pinned else {}

    def serve_groups(request, _m):
        purpose = parse_qs(urlparse(request.url).query).get('purpose', ['dedup'])[0]
        items = groups if groups is not None else GROUPS
        items = items if purpose == 'leak' else []
        return fill(
            'NearDupGroupPage',
            items=items,
            next_cursor=None,
            empty_reason=None if items else '0 leak groups at cosine 0.9 among 1200 items',
        )

    def serve_group(_request, _m):
        return _detail(pins)

    def serve_decisions(_request, _m):
        served = [DECISION] if pins else []
        return fill('NearDupDecisionPage', items=served, next_cursor=None)

    def decide(request, _m):
        pins['img-b'] = {'id': DECISION['id'], 'kind': 'different'}
        return 201, DECISION

    def undecide(_request, _m):
        pins.clear()
        return 204, ''

    # The profile list first; the stubs below are registered later, so they win.
    serve_kind(stub, 'near-dup-profiles', 'near_dup_profiles', 'near_dup_profile', 'project', {})
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on('GET', r'/near_duplicates/status$', status or _status())
    stub.on(
        'GET',
        r'/near_duplicates/runs(\?|$)',
        fill('NearDupRunPage', items=runs if runs is not None else [_run('image'), _run('item')]),
    )
    stub.on('GET', r'/near_duplicates/groups(\?|$)', serve_groups)
    stub.on('GET', r'/near_duplicates/groups/[^/?]+', serve_group)
    stub.on('GET', r'/near_duplicates/decisions(\?|$)', serve_decisions)
    stub.on('POST', r'/near_duplicates/decisions$', decide)
    stub.on('DELETE', r'/near_duplicates/decisions/dec-1$', undecide)
    stub.on('GET', r'/(images|crops)/[^/]+/thumbnail(\?|$)', (200, PHOTO, 'image/jpeg'))


def test_the_explainer_runs_pinned_pairs_and_leak_groups(stub, page, app_url):
    _serve(stub, pinned=True)
    page.set_viewport_size({'width': 1600, 'height': 1000})
    page.goto(f'{app_url}{URL}')
    explainer = page.get_by_test_id('near-dup-explainer')
    expect(explainer).to_contain_text('Nothing is ever deleted', timeout=ACTION_TIMEOUT_MS)
    expect(explainer).to_contain_text('off unless you turn it on')
    expect(page.get_by_test_id('near-dup-stale-image')).to_contain_text('not yet grouped (12)')
    expect(page.get_by_test_id('near-dup-run-row')).to_have_count(2)
    expect(page.get_by_test_id('near-dup-decision')).to_have_count(1, timeout=ACTION_TIMEOUT_MS)
    links = page.get_by_test_id('near-dup-group-workspace')
    expect(links).to_have_count(2, timeout=ACTION_TIMEOUT_MS)
    href = links.first.get_attribute('href') or ''
    assert 'view=detections' in href and 'scope=near_dup%3A' in href, href
    save_screens(page, 'uxr-d', 'settings-near-dup')


def test_the_pinned_pairs_list_undoes_a_pin(stub, page, app_url):
    _serve(stub, pinned=True)
    page.goto(f'{app_url}{URL}')
    expect(page.get_by_test_id('near-dup-decision')).to_have_count(1, timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('near-dup-decision-undo').click()
    expect(page.get_by_test_id('near-dup-decisions-empty')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert [c for c in stub.calls if c[0] == 'DELETE'][-1][1].endswith('/decisions/dec-1')


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
    save_screens(page, 'uxr-d', 'settings-near-dup-empty')

    page.get_by_test_id('near-dup-run-full-item').click()
    expect(page.get_by_test_id('near-dup-run-error')).to_contain_text(
        'needs a current image-level run first', timeout=ACTION_TIMEOUT_MS
    )
    assert [c for c in stub.calls if c[1].endswith('/near_duplicates/runs')][-1][2] == {
        'level': 'item',
        'mode': 'full',
    }


def test_a_failed_status_read_shows_the_error(stub, page, app_url):
    _serve(stub)
    stub.on('GET', r'/near_duplicates/status$', (500, {'detail': 'store down'}))
    page.goto(f'{app_url}{URL}')
    expect(page.get_by_test_id('near-dup-load-error')).to_contain_text(
        'store down', timeout=ACTION_TIMEOUT_MS
    )


def test_page_fits_a_430px_phone(stub, page, app_url):
    _serve(stub)
    page.set_viewport_size({'width': 430, 'height': 900})
    page.goto(f'{app_url}{URL}')
    expect(page.get_by_test_id('near-dup-group-workspace')).to_have_count(
        2, timeout=ACTION_TIMEOUT_MS
    )
    overflow = page.evaluate(
        '() => document.documentElement.scrollWidth - document.documentElement.clientWidth'
    )
    assert overflow <= 0, f'the page scrolls sideways by {overflow}px at 430px'


def test_label_the_whole_group_uses_the_group_filter(stub, page, app_url):
    from fixtures.contract import label_op_accepted
    from harness.label_ops import stub_label_op
    from harness.screens import shot

    serve_gallery(stub, [])
    group = fill('ImageNearDup', group_id='g1', purpose='dedup', size=3, is_representative=True)
    stub.on(
        'GET',
        r'/images/img-1$',
        fill(
            'ImageDetail',
            image=fill(
                'ImageInfo',
                image_id='img-1',
                file_name='img-1.jpg',
                in_holdout=False,
                near_dup=group,
            ),
            counts=fill('ImageCounts', items=2, labeled=1, unlabeled=1),
            items=[],
            boxes=[],
        ),
    )
    calls: list[dict] = []
    stub_label_op(stub, r'/crops/batch_label$', 'op-g', updated=3, method='PUT')

    def label(request, _match):
        body = request.post_data_json or {}
        calls.append(body)
        if body.get('dry_run'):
            return (200, {'dry_run': True, 'selected': 3})
        return (202, label_op_accepted('op-g', 'job-op-g'))

    stub.on('PUT', r'/crops/batch_label$', label)
    page.set_viewport_size({'width': 1600, 'height': 900})
    page.goto(f'{app_url}/p/default/workspace?view=gallery')
    expect(page.get_by_test_id('image-card').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('image-card-open').nth(1).click()
    page.get_by_test_id('scope-near-dup').click()
    expect(page.get_by_test_id('label-whole-group')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    shot(page, 'near-dup-group-scope', widths=(1600, 430))
    page.set_viewport_size({'width': 1600, 'height': 900})
    page.get_by_test_id('label-whole-group').click()
    page.get_by_role('combobox', name='Search classes').fill('widget')
    page.get_by_role('dialog').get_by_role('option', name=re.compile('widget')).first.click()
    expect(page.get_by_test_id('selection-count')).to_contain_text('3', timeout=ACTION_TIMEOUT_MS)
    assert calls[0]['dry_run'] is True
    assert calls[0]['selection']['filter']['near_dup_group'] == 'g1'
