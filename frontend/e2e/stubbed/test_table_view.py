"""The Table view: served columns and cells, a served sort, column choice in the URL, row
selection into the bulk bar, and a table that scrolls inside its own container at 430 px."""

from __future__ import annotations

import re
import struct
import zlib
from urllib.parse import parse_qs, urlparse

from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import SCREENS
from harness.waits import wait_for_url_state
from playwright.sync_api import expect


COLUMNS = [
    {'id': 'thumbnail', 'label': '', 'sortable': False, 'align': 'start', 'kind': 'thumbnail'},
    {'id': 'class_name', 'label': 'Class', 'sortable': True, 'align': 'start', 'kind': 'text'},
    {
        'id': 'detector_confidence',
        'label': 'Confidence',
        'sortable': True,
        'align': 'end',
        'kind': 'number',
    },
    {'id': 'image_name', 'label': 'Image', 'sortable': True, 'align': 'start', 'kind': 'text'},
]
SORTS = [
    {'id': 'detector_confidence:desc', 'label': 'Confidence, high first'},
    {'id': 'detector_confidence:asc', 'label': 'Confidence, low first'},
    {'id': 'class_name:asc', 'label': 'Class, A to Z'},
    {'id': 'class_name:desc', 'label': 'Class, Z to A'},
]


def png(shade: int) -> bytes:
    """A 24x24 grey-green tile; stands in for a crop thumbnail."""
    rows = b''.join(
        b'\x00' + b''.join(bytes([shade + x * 3, 120 + y * 2, 110]) for x in range(24))
        for y in range(24)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body))

    return (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', 24, 24, 8, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(rows))
        + chunk(b'IEND', b'')
    )


def table_page(shown: list[str], sort: str, rows: str = 'detections') -> dict:
    cols = [c for c in COLUMNS if c['id'] in shown]
    items = [
        {
            'crop_id': f'crop-{i}',
            'image_id': f'img-{i}',
            'thumbnail_url': f'/curation/crops/crop-{i}/thumbnail',
            'cells': {
                'class_name': {'text': 'widget', 'value': 'widget', 'tone': 'machine'},
                'detector_confidence': {'text': f'0.{90 - i}', 'value': (90 - i) / 100},
                'image_name': {
                    'text': f'a_long_image_file_name_for_the_horizontal_scroll_check_{i}.jpg',
                    'value': f'f{i}',
                },
            },
        }
        for i in range(4)
    ]
    if rows == 'images':
        for item in items:
            item['crop_id'] = None
    return {
        'rows': rows,
        'columns': cols,
        'items': items,
        'next_cursor': None,
        'total': 4,
        'limit': 60,
        'sort_applied': sort,
        'sort_options': SORTS,
        'hidden': {'test_holdout': 0, 'excluded': 0},
        'join_truncated': False,
    }


def serve(stub) -> list[dict]:
    seen: list[dict] = []
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})

    def table(request, _match):
        q = {k: v[0] for k, v in parse_qs(urlparse(request.url).query).items()}
        seen.append(q)
        shown = (q.get('columns') or 'thumbnail,class_name,detector_confidence').split(',')
        return table_page(
            shown, q.get('sort', 'detector_confidence:desc'), q.get('rows', 'detections')
        )

    stub.on(
        'GET',
        r'/crops/crop-(\d)/thumbnail',
        lambda _r, m: (200, png(40 + int(m.group(1)) * 30), 'image/png'),
    )
    stub.on(
        'GET',
        r'/table/columns$',
        {
            'rows': 'detections',
            'columns': COLUMNS,
            'default': ['thumbnail', 'class_name', 'detector_confidence'],
        },
    )
    stub.on('GET', r'/table$', table)
    return seen


def test_header_click_asks_for_the_served_sort(stub, page, app_url):
    seen = serve(stub)
    page.goto(f'{app_url}/p/default/workspace?view=table')
    expect(page.get_by_test_id('data-table-row')).to_have_count(4, timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('data-table-sort-detector_confidence').click()
    expect(page.locator('th[aria-sort="ascending"], th[aria-sort="descending"]')).to_have_count(1)
    assert seen[-1]['sort'] == 'detector_confidence:asc'
    wait_for_url_state(page, has=('sort_table=detector_confidence:asc',))


def test_column_choice_is_in_the_url_and_the_request(stub, page, app_url):
    seen = serve(stub)
    page.goto(f'{app_url}/p/default/workspace?view=table')
    expect(page.get_by_test_id('data-table-row')).to_have_count(4, timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('column-picker').locator('summary').click()
    page.get_by_label('Image', exact=True).check()
    expect(page.get_by_role('columnheader', name='Image')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert seen[-1]['columns'] == 'thumbnail,class_name,detector_confidence,image_name'
    expect(page).to_have_url(
        re.compile(r'columns=thumbnail(,|%2C)class_name'), timeout=ACTION_TIMEOUT_MS
    )


def test_space_and_shift_arrows_select_rows_for_the_bulk_bar(stub, page, app_url):
    serve(stub)
    page.goto(f'{app_url}/p/default/workspace?view=table')
    rows = page.get_by_test_id('data-table-row')
    expect(rows).to_have_count(4, timeout=ACTION_TIMEOUT_MS)
    rows.nth(0).click()
    page.keyboard.press('Space')
    page.keyboard.press('Shift+ArrowDown')
    page.keyboard.press('Shift+ArrowDown')
    expect(page.locator('[data-testid=data-table-row][aria-selected=true]')).to_have_count(
        3, timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('bulk-summary')).to_be_visible(timeout=ACTION_TIMEOUT_MS)


def test_table_scrolls_inside_itself_at_430(stub, page, app_url):
    serve(stub)
    page.set_viewport_size({'width': 430, 'height': 900})
    page.goto(
        f'{app_url}/p/default/workspace?view=table'
        '&columns=thumbnail,class_name,detector_confidence,image_name'
    )
    expect(page.get_by_test_id('data-table-row')).to_have_count(4, timeout=ACTION_TIMEOUT_MS)
    m = page.evaluate(
        """() => {
          const s = document.querySelector('[data-testid=data-table-scroll]');
          return {page: document.documentElement.scrollWidth, vw: window.innerWidth,
                  inner: s.scrollWidth, box: s.clientWidth};
        }"""
    )
    assert m['page'] <= m['vw'], m
    assert m['inner'] > m['box'], m


def test_screens(stub, page, app_url):
    """Detections, a sorted column and selected rows at 1600 and 430 (viewed by hand)."""
    serve(stub)
    out = SCREENS / 'workflow_3f'
    out.mkdir(parents=True, exist_ok=True)
    for width in (1600, 430):
        page.set_viewport_size({'width': width, 'height': 1000})
        page.goto(f'{app_url}/p/default/workspace?view=table&sort_table=detector_confidence:asc')
        rows = page.get_by_test_id('data-table-row')
        expect(rows).to_have_count(4, timeout=ACTION_TIMEOUT_MS)
        page.screenshot(path=str(out / f'table-sorted-{width}.png'), full_page=True)
        rows.nth(0).click()
        page.keyboard.press('Space')
        page.keyboard.press('Shift+ArrowDown')
        expect(page.locator('[data-testid=data-table-row][aria-selected=true]')).to_have_count(2)
        page.screenshot(path=str(out / f'table-selected-{width}.png'), full_page=True)
        page.get_by_test_id('table-rows-images').click()
        expect(page.get_by_test_id('table-rows-images')).to_have_attribute(
            'aria-pressed', 'true', timeout=ACTION_TIMEOUT_MS
        )
        expect(rows).to_have_count(4, timeout=ACTION_TIMEOUT_MS)
        page.screenshot(path=str(out / f'table-images-{width}.png'), full_page=True)
