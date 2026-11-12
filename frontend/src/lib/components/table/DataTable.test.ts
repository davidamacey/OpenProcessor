import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import DataTable from './DataTable.svelte';
import type { TableColumn, TableRow } from '$lib/types_table';

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLElement | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  target?.remove();
  instance = undefined;
});

const COLUMNS: TableColumn[] = [
  { id: 'thumbnail', label: '', sortable: false, align: 'start', kind: 'thumbnail' },
  { id: 'class_name', label: 'Class', sortable: true, align: 'start', kind: 'text' },
  { id: 'conf', label: 'Confidence', sortable: true, align: 'end', kind: 'number' },
];
const ROWS: TableRow[] = [
  {
    crop_id: 'c1',
    image_id: 'i1',
    thumbnail_url: '/t/c1',
    cells: { class_name: { text: 'class_a' }, conf: { text: '0.91', value: 0.91 } },
  },
  {
    crop_id: 'c2',
    image_id: 'i2',
    thumbnail_url: '/t/c2',
    cells: { class_name: { text: 'class_b', tone: 'muted' }, conf: { text: '0.40' } },
  },
];

function render(over: Record<string, unknown> = {}) {
  const props = {
    columns: COLUMNS,
    rows: ROWS,
    sortApplied: 'conf:desc',
    sortOptions: [
      { id: 'conf:desc', label: 'Confidence, high first' },
      { id: 'conf:asc', label: 'Confidence, low first' },
    ],
    selectedIds: new Set<string>(['c2']),
    focusedId: 'c1',
    onsort: vi.fn(),
    onselect: vi.fn(),
    onfocus: vi.fn(),
    onopen: vi.fn(),
    ...over,
  };
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(DataTable, { target, props });
  flushSync();
  return { props, el: target };
}

describe('DataTable', () => {
  it('shows served cell text verbatim, in the served order', () => {
    const { el } = render();
    const rows = [...el.querySelectorAll('[data-testid="data-table-row"]')];
    expect(rows.map((r) => r.textContent?.replace(/\s+/g, ' ').trim())).toEqual([
      'class_a0.91',
      'class_b0.40',
    ]);
  });

  it('aria-sort follows the served sort_applied and the thumbnail header has a label', () => {
    const { el } = render();
    const th = [...el.querySelectorAll('thead th')];
    expect(th[2].getAttribute('aria-sort')).toBe('none');
    expect(th[3].getAttribute('aria-sort')).toBe('descending');
    expect(th[1].textContent).toContain('Preview');
    expect(th.every((h) => (h.textContent ?? '').trim() !== '')).toBe(true);
  });

  it('a header click reports the column and a row click its key', () => {
    const { el, props } = render();
    (
      el.querySelector('[data-testid="data-table-sort-class_name"]') as HTMLElement
    ).click();
    expect(props.onsort).toHaveBeenCalledWith('class_name');
    (el.querySelectorAll('[data-testid="data-table-row"]')[1] as HTMLElement).click();
    expect(props.onfocus).toHaveBeenCalledWith('c2');
  });

  it('marks selected rows', () => {
    const { el } = render();
    const rows = el.querySelectorAll('[data-testid="data-table-row"]');
    expect(rows[0].getAttribute('aria-selected')).toBe('false');
    expect(rows[1].getAttribute('aria-selected')).toBe('true');
  });
});
