import { existsSync, readFileSync } from 'node:fs';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { setScopedPrefix } from '$lib/api';
import {
  createTableController,
  type TableUrlState,
} from '$lib/table/tableController.svelte';
import type { NearDupFilter } from '$lib/types_images';
import type { TableColumns, TablePage, TableRow } from '$lib/types_table';

const fetchMock = vi.fn();

function ok(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

const VOCAB: TableColumns = {
  rows: 'detections',
  default: ['thumbnail', 'detector_confidence'],
  columns: [
    { id: 'thumbnail', label: '', sortable: false, align: 'start', kind: 'thumbnail' },
    {
      id: 'detector_confidence',
      label: 'Confidence',
      sortable: true,
      align: 'end',
      kind: 'number',
    },
    { id: 'class_name', label: 'Class', sortable: true, align: 'start', kind: 'text' },
  ],
};

function row(id: string, text: string): TableRow {
  return {
    crop_id: id,
    image_id: `img-${id}`,
    thumbnail_url: `/t/${id}`,
    cells: { detector_confidence: { text, value: 0.5 } },
  };
}

function page(
  items: TableRow[],
  next: string | null,
  applied = 'detector_confidence:desc',
): TablePage {
  return {
    rows: 'detections',
    columns: VOCAB.columns.slice(0, 2),
    items,
    next_cursor: next,
    total: 3,
    limit: 2,
    sort_applied: applied,
    sort_options: [
      { id: 'detector_confidence:desc', label: 'Confidence, high first' },
      { id: 'detector_confidence:asc', label: 'Confidence, low first' },
    ],
    hidden: { test_holdout: 0, excluded: 0 },
    join_truncated: false,
  };
}

function make(
  initial: Partial<TableUrlState> = {},
  nearDup: () => NearDupFilter | null = () => null,
) {
  const writeUrl = vi.fn();
  const c = createTableController({
    initial: { rows: 'detections', columns: [], sort: null, ...initial },
    itemFilter: () => ({}),
    nearDup,
    writeUrl,
  });
  return { c, writeUrl };
}

function urlOf(call: number): URL {
  return new URL(fetchMock.mock.calls[call][0], 'http://x');
}

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal('fetch', fetchMock);
  setScopedPrefix('/curation/projects/default');
});

describe('table controller', () => {
  it('cells render verbatim', async () => {
    fetchMock
      .mockResolvedValueOnce(ok(VOCAB))
      .mockResolvedValueOnce(ok(page([row('a', '0.91 (served)')], null)));
    const { c } = make();
    await c.load();
    expect(c.rows[0].cells.detector_confidence.text).toBe('0.91 (served)');
    expect(c.sortApplied).toBe('detector_confidence:desc');
  });

  it('header click sends the served sort id', async () => {
    fetchMock
      .mockResolvedValueOnce(ok(VOCAB))
      .mockResolvedValueOnce(ok(page([row('a', '1')], null)))
      .mockResolvedValueOnce(ok(page([row('a', '1')], null, 'detector_confidence:asc')));
    const { c, writeUrl } = make();
    await c.load();
    // desc is applied, so the next click on the column takes the other served direction.
    await c.sortBy('detector_confidence');
    expect(urlOf(2).searchParams.get('sort')).toBe('detector_confidence:asc');
    expect(writeUrl).toHaveBeenLastCalledWith(
      expect.objectContaining({ sort: 'detector_confidence:asc' }),
    );
  });

  it('sends the near-duplicate role of the URL, and none when it is off', async () => {
    fetchMock
      .mockResolvedValueOnce(ok(VOCAB))
      .mockResolvedValueOnce(ok(page([], null)))
      .mockResolvedValueOnce(ok(page([], null)));
    let role: NearDupFilter | null = 'collapsed';
    const { c } = make({}, () => role);
    await c.load();
    expect(urlOf(1).searchParams.get('near_dup')).toBe('collapsed');
    role = null;
    await c.reload();
    expect(urlOf(2).searchParams.has('near_dup')).toBe(false);
  });

  it('appends by cursor', async () => {
    fetchMock
      .mockResolvedValueOnce(ok(VOCAB))
      .mockResolvedValueOnce(ok(page([row('a', '1'), row('b', '2')], 'CUR1')))
      .mockResolvedValueOnce(ok(page([row('c', '3')], null)));
    const { c } = make();
    await c.load();
    await c.loadMore();
    expect(urlOf(2).searchParams.get('cursor')).toBe('CUR1');
    expect(c.rows.map((r) => r.crop_id)).toEqual(['a', 'b', 'c']);
  });

  it('column choice writes columns to the URL', async () => {
    fetchMock
      .mockResolvedValueOnce(ok(VOCAB))
      .mockResolvedValueOnce(ok(page([], null)))
      .mockResolvedValueOnce(ok(page([], null)));
    const { c, writeUrl } = make();
    await c.load();
    await c.setColumns(['thumbnail', 'class_name']);
    expect(writeUrl).toHaveBeenLastCalledWith(
      expect.objectContaining({ columns: ['thumbnail', 'class_name'] }),
    );
    expect(urlOf(2).searchParams.get('columns')).toBe('thumbnail,class_name');
  });

  it('starts from the columns, sort and rows in the URL', async () => {
    fetchMock
      .mockResolvedValueOnce(ok({ ...VOCAB, rows: 'images' }))
      .mockResolvedValueOnce(ok(page([], null)));
    const { c } = make({
      rows: 'images',
      columns: ['thumbnail', 'class_name'],
      sort: 'class_name:asc',
    });
    await c.load();
    const u = urlOf(1);
    expect(u.searchParams.get('rows')).toBe('images');
    expect(u.searchParams.get('columns')).toBe('thumbnail,class_name');
    expect(u.searchParams.get('sort')).toBe('class_name:asc');
  });

  it('no client sort', () => {
    for (const f of [
      'src/lib/table/tableController.svelte.ts',
      'src/lib/components/table/DataTable.svelte',
      'src/lib/workspace/views/TableView.svelte',
    ]) {
      if (!existsSync(f)) continue;
      expect(readFileSync(f, 'utf8'), f).not.toMatch(
        /\.sort\(|\.toSorted\(|localeCompare|\.reverse\(/,
      );
    }
  });
});
