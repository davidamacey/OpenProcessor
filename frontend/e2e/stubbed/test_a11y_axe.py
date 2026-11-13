"""axe-core on the main pages, against the stub: any violation fails.

axe is injected from `node_modules` (devDependency `axe-core`), so the check is
pinned and offline. Every page is rendered in a state that shows real content
(rows, cards, tables), because an empty page hides most violations. Later
waves add their new page states here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fixtures.image_editor import Editor
from fixtures.wire import project, projects_response
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect
from test_assist_loop_panel import (
    open_panel,
    serve_loop,
    serve_plan_form,
    show_finished_loop,
    show_plan_form,
)
from test_cluster_discard import CLASSES, CLUSTERS, crop
from test_cluster_facets import FACET_UID, serve_cluster
from test_cluster_list_filters import serve as serve_filtered_clusters
from test_curation_settings import METHODS_TODAY, SETTINGS_EMPTY
from test_embedding_spaces_page import serve_spaces
from test_embedding_view import serve as serve_embedding
from test_fresh_run_cosmetics import EMPTY_PROPOSALS
from test_gallery_view import serve as serve_gallery
from test_label_confirmation import _stub_audit
from test_label_ops import serve as serve_label_ops
from test_map_view import _cells as map_cells, _meta as map_meta
from test_near_duplicates import _serve as serve_near_dups
from test_next_steps import FRESH, serve_next_steps
from test_open_vocab import serve_open_vocab
from test_presets import serve_presets
from test_prompt_packs import serve_packs
from test_review_crop_viewport import review_item
from test_similar_crops import serve_similar
from test_source_missing import serve_cluster_grid, serve_review_pane
from test_table_view import serve as serve_table_view


AXE_JS = Path(__file__).resolve().parents[2] / 'node_modules' / 'axe-core' / 'axe.min.js'
API_PREFIX = '/curation'


def serve_clusters(stub) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/clusters(\?|$)', CLUSTERS)
    stub.on(
        'GET',
        r'/crops(\?|$)',
        {'total': 3, 'next_cursor': None, 'limit': 60, 'crops': [crop(i) for i in range(3)]},
    )


def serve_map(stub) -> None:
    stub.on('GET', r'/map$', (200, map_meta('')))
    stub.on('GET', r'/map/cells(\?|$)', (200, map_cells(), 'application/geo+json'))
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})


def serve_projects_page(stub) -> None:
    rows = [
        project(API_PREFIX, 'default', is_default=True, deletable=False),
        project(API_PREFIX, 'widgets-a', display_name='Widgets A'),
    ]
    stub.on('GET', rf'^{API_PREFIX}/projects$', (200, projects_response(API_PREFIX, rows)))


def serve_review(stub) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/methods(\?|$)', {'strategies': [], 'flags': {}})
    stub.on(
        'GET',
        r'/review/(?!tabs)',
        (
            200,
            {
                'items': [review_item(i) for i in range(3)],
                'total': 3,
                'next_cursor': None,
                'limit': 30,
            },
        ),
    )
    stub.on(
        'GET',
        r'/images/[^/]+/file$',
        (
            200,
            {'image': {'path': '/img.jpg', 'width': 640, 'height': 480}, 'items': [review_item(0)]},
        ),
    )


def serve_review_similar(stub) -> None:
    serve_review(stub)
    serve_similar(stub)


def serve_classes(stub) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/review/new_class_proposals/summary(\?|$)', EMPTY_PROPOSALS)
    stub.on('GET', r'/test_holdout/stats(\?|$)', {'total': 0, 'by_class': []})


def serve_settings(stub) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/scores/coverage(\?|$)', {'coverage': {}})
    stub.on('GET', r'/scores/status(\?|$)', {'status': 'idle'})
    stub.on('GET', r'/methods(\?|$)', METHODS_TODAY)
    stub.on('GET', r'/settings(\?|$)', SETTINGS_EMPTY)


def serve_prompt_packs(stub) -> None:
    serve_packs(stub)
    stub.on('GET', r'/scores/coverage(\?|$)', {'coverage': {}})
    stub.on('GET', r'/scores/status(\?|$)', {'status': 'idle'})


def serve_open_vocab_list(stub) -> None:
    serve_open_vocab(stub)
    stub.on('GET', r'/scores/coverage(\?|$)', {'coverage': {}})
    stub.on('GET', r'/scores/status(\?|$)', {'status': 'idle'})


def serve_table(stub) -> None:
    serve_clusters(stub)
    serve_table_view(stub)


def serve_clusters_with_filters(stub) -> None:
    serve_filtered_clusters(stub, [])


def serve_cluster_with_facets(stub) -> None:
    serve_cluster(stub, [])


def show_label_op_dry_run(page) -> None:
    page.get_by_role(
        'button', name='Details for Label to widget_a at 2026-10-30 10:00:00 UTC'
    ).click()
    page.get_by_test_id('label-op-undo').click()
    expect(page.get_by_test_id('label-op-confirm')).to_be_visible(timeout=ACTION_TIMEOUT_MS)


def open_class_picker(page) -> None:
    page.locator('img').first.click()
    page.get_by_test_id('assign-selected').click()


REVIEW_PATH = '/p/default/workspace?view=review&queue=to_label'
REVIEW_REGION_PATH = '/p/default/workspace?view=review&queue=regions'

# (id, path, stub setup, selector that proves the data loaded[, step that opens a panel])
PAGES = [
    ('projects', '/projects', serve_projects_page, '[data-testid="project-row-widgets-a"]'),
    (
        'workspace-clusters',
        '/p/default/workspace?view=clusters',
        serve_clusters,
        '[data-testid=workspace-view] li button',
    ),
    (
        'clusters-with-filters',
        '/p/default/workspace?view=clusters&flag=unvalidated',
        serve_clusters_with_filters,
        '[data-testid=workspace-view] li button',
    ),
    (
        'clusters-loop-plan',
        '/p/default/workspace?view=clusters',
        serve_plan_form,
        '[data-testid="vlm-menu-toggle"]',
        show_plan_form,
    ),
    (
        'workspace-vlm-menu',
        '/p/default/workspace?view=clusters',
        serve_plan_form,
        '[data-testid="vlm-menu-toggle"]',
        lambda page: (
            page.get_by_test_id('vlm-menu-toggle').click(),
            expect(page.get_by_test_id('vlm-menu-assist')).to_be_visible(timeout=ACTION_TIMEOUT_MS),
        ),
    ),
    (
        'clusters-loop-finished',
        '/p/default/workspace?view=clusters',
        lambda stub: serve_loop(stub, 'completed'),
        '[data-testid="vlm-menu-toggle"]',
        show_finished_loop,
    ),
    (
        'clusters-loop-live',
        '/p/default/workspace?view=clusters',
        lambda stub: serve_loop(stub, 'running'),
        '[data-testid="vlm-menu-toggle"]',
        lambda page: (
            open_panel(page),
            expect(page.get_by_test_id('loop-cancel')).to_be_visible(timeout=ACTION_TIMEOUT_MS),
        ),
    ),
    (
        'workspace-cluster-scope',
        '/p/default/workspace?view=clusters&scope=cluster:1',
        serve_clusters,
        '[data-testid="class-name"]',
    ),
    (
        'cluster-detail-facets',
        f'/p/default/workspace?view=clusters&scope=cluster:{FACET_UID}',
        serve_cluster_with_facets,
        '[data-testid="scope-refine"]',
        lambda page: (
            page.get_by_test_id('scope-refine').click(),
            expect(page.locator('[data-testid="cluster-facets"]').first).to_be_visible(
                timeout=ACTION_TIMEOUT_MS
            ),
        ),
    ),
    ('workspace-map', '/p/default/workspace?view=map', serve_map, '[data-testid=map-located]'),
    (
        'workspace-embedding',
        '/p/default/workspace?view=embedding&class_name=sedan',
        lambda stub: serve_embedding(stub, []),
        '[data-testid=embedding-legend-entry]',
    ),
    (
        'workspace-gallery',
        '/p/default/workspace?view=gallery',
        lambda stub: serve_gallery(stub, []),
        '[data-testid=image-card]',
    ),
    (
        'workspace-gallery-details',
        '/p/default/workspace?view=gallery',
        lambda stub: serve_gallery(stub, []),
        '[data-testid=image-card]',
        lambda page: (
            page.get_by_test_id('image-card-open').nth(1).click(),
            expect(page.get_by_test_id('image-details')).to_be_visible(timeout=ACTION_TIMEOUT_MS),
        ),
    ),
    (
        'workspace-detections',
        '/p/default/workspace?view=detections',
        serve_clusters,
        '[data-testid="detections-view"]',
    ),
    (
        'workspace-table',
        '/p/default/workspace?view=table',
        serve_table,
        '[data-testid="data-table-row"]',
    ),
    (
        'workspace-class-picker',
        '/p/default/workspace?view=detections',
        serve_clusters,
        '[data-testid="detections-view"] img',
        open_class_picker,
    ),
    ('workspace-review', REVIEW_PATH, serve_review, '[data-testid="queue-counter"]'),
    ('workspace-review-region', REVIEW_REGION_PATH, serve_review, '[data-testid="queue-counter"]'),
    (
        'workspace-review-similar',
        REVIEW_PATH,
        serve_review_similar,
        '[data-testid="review-similar"]',
    ),
    (
        'image-editor',
        '/p/default/images/img-1?view=gallery',
        Editor,
        '[data-testid="overlay-box"]',
    ),
    (
        'clusters-similar',
        '/p/default/workspace?view=clusters&similar_to=crop-0',
        serve_similar,
        '[data-testid="search-summary"]',
    ),
    (
        'cluster-detail-similar',
        '/p/default/workspace?view=clusters&scope=cluster:1',
        serve_similar,
        '[data-testid="crop-similar"]',
    ),
    (
        'cluster-detail-source-missing',
        '/p/default/workspace?view=clusters&scope=cluster:1',
        serve_cluster_grid,
        '[role="status"]',
    ),
    ('review-source-missing', REVIEW_PATH, serve_review_pane, '[role="status"]'),
    ('classes', '/p/default/classes', serve_classes, 'main'),
    ('settings', '/p/default/settings', serve_settings, 'main'),
    (
        'quality-audit',
        '/p/default/quality/audit',
        _stub_audit,
        '[data-testid="audit-start"]',
    ),
    (
        'quality-near-duplicates',
        '/p/default/quality/near-duplicates',
        serve_near_dups,
        '[data-testid="near-dup-status"]',
    ),
    (
        'dashboard-next-steps',
        '/p/default/dashboard',
        lambda stub: serve_next_steps(stub, FRESH),
        '[data-testid="next-steps"]',
    ),
    (
        'quality-history',
        '/p/default/quality/history',
        serve_label_ops,
        '[data-testid="label-op-row"]',
    ),
    (
        'quality-history-dry-run',
        '/p/default/quality/history',
        serve_label_ops,
        '[data-testid="label-op-row"]',
        show_label_op_dry_run,
    ),
    ('settings-prompt-packs', '/p/default/settings/prompt-packs', serve_prompt_packs, 'table'),
    ('settings-open-vocab', '/p/default/settings/open-vocab', serve_open_vocab_list, 'table'),
    (
        'settings-presets',
        '/p/default/settings/presets/cluster-profiles',
        serve_presets,
        '[data-testid="preset-table"]',
    ),
    (
        'settings-presets-editor',
        '/p/default/settings/presets/cluster-profiles/strict',
        serve_presets,
        '[data-testid="preset-body"]',
    ),
    (
        'settings-presets-activate-dialog',
        '/p/default/settings/presets/cluster-profiles/strict',
        serve_presets,
        '[data-testid="preset-body"]',
        lambda page: (
            page.get_by_test_id('config-activate').click(),
            expect(page.get_by_role('dialog')).to_be_visible(timeout=ACTION_TIMEOUT_MS),
        ),
    ),
    (
        'settings-embedding-register-dialog',
        '/p/default/settings/embedding-spaces',
        serve_spaces,
        '[data-testid="model-table"]',
        lambda page: (
            page.get_by_test_id('space-register').click(),
            expect(page.get_by_role('dialog')).to_be_visible(timeout=ACTION_TIMEOUT_MS),
        ),
    ),
    (
        'settings-embedding-spaces',
        '/p/default/settings/embedding-spaces',
        serve_spaces,
        '[data-testid="model-table"]',
    ),
]


@pytest.mark.parametrize(
    ('path', 'setup', 'ready', 'reveal'),
    [(*p[1:4], p[4] if len(p) > 4 else None) for p in PAGES],
    ids=[p[0] for p in PAGES],
)
def test_page_has_no_axe_violations(stub, page, app_url, path, setup, ready, reveal):
    setup(stub)
    page.goto(f'{app_url}{path}')
    expect(page.locator(ready).first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    if reveal is not None:
        reveal(page)
    page.wait_for_load_state('networkidle')
    page.add_script_tag(path=str(AXE_JS))
    result = page.evaluate("axe.run(document, { resultTypes: ['violations'] })")
    found = [
        {'rule': v['id'], 'impact': v['impact'], 'nodes': [n['target'] for n in v['nodes'][:5]]}
        for v in result['violations']
    ]
    assert found == [], json.dumps(found, indent=2)
