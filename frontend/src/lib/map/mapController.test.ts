import { describe, expect, it, vi } from 'vitest';
import {
  CELL_CARDS,
  MapController,
  type MapDeps,
  type MapHost,
} from './mapController.svelte';
import { mapMeta } from '$lib/test/fixtures/mapMeta';
import type { ImagePage } from '$lib/types_images';
import type { SelectionSummary } from '$lib/types_itemFilter';
import type { MapCellCollection, MapMeta } from '$lib/types_map';

const META: MapMeta = mapMeta({ bounds: { west: 7, south: 46, east: 9, north: 48 } });
const CELLS: MapCellCollection = {
  type: 'FeatureCollection',
  features: [],
  precision: 9,
  truncated: false,
  join_truncated: false,
  n_located_matching: 0,
  n_unlocated_matching: 0,
};
const BOUNDS = { west: 8.4375, south: 47.5, east: 8.4551, north: 47.51 };
const CELL = { tile: '9/267/179', count: 7, bounds: BOUNDS };
const SUMMARY = {
  summary_text: '7 detections in 4 images',
  images: 4,
} as SelectionSummary;
const PAGE = { items: [], total: 4 } as unknown as ImagePage;

function host() {
  return {
    itemQuery: vi.fn(() => ({ class_name: ['widget'], review_tab: 'unreviewed' })),
    itemBody: vi.fn(() => ({ class_names: ['widget'], review_tab: 'unreviewed' })),
    setFocus: vi.fn(),
    applyFilter: vi.fn(),
    filter: { geoBbox: null, hasLocation: null },
  } as unknown as MapHost & {
    itemQuery: ReturnType<typeof vi.fn>;
    setFocus: ReturnType<typeof vi.fn>;
    applyFilter: ReturnType<typeof vi.fn>;
  };
}

function deps(over: Partial<MapDeps> = {}): MapDeps {
  return {
    meta: vi.fn().mockResolvedValue(META),
    cells: vi.fn().mockResolvedValue(CELLS),
    summary: vi.fn().mockResolvedValue(SUMMARY),
    images: vi.fn().mockResolvedValue(PAGE),
    ...over,
  };
}

describe('MapController', () => {
  it('loads the served meta, or the served error and never a made-up map', async () => {
    const ok = new MapController(host(), deps());
    await ok.loadMeta();
    expect(ok.meta).toEqual(META);
    const bad = new MapController(
      host(),
      deps({ meta: vi.fn().mockRejectedValue(new Error('boom')) }),
    );
    await bad.loadMeta();
    expect(bad.meta).toBeNull();
    expect(bad.metaError).toBe('boom');
  });

  it('sends the workspace filter and queue', async () => {
    const d = deps();
    await new MapController(host(), d).loadCells({ zoom: 6, bbox: '7,46,9,48' });
    expect(d.cells).toHaveBeenCalledWith(
      { zoom: 6, bbox: '7,46,9,48' },
      { class_name: ['widget'], review_tab: 'unreviewed' },
      expect.any(AbortSignal),
    );
  });

  it('a newer viewport cancels the older request and keeps only the newer answer', async () => {
    const answers: Array<(v: MapCellCollection) => void> = [];
    const signals: AbortSignal[] = [];
    const ctl = new MapController(
      host(),
      deps({
        cells: vi.fn().mockImplementation(
          (_v, _f, signal: AbortSignal) =>
            new Promise<MapCellCollection>((resolve) => {
              signals.push(signal);
              answers.push(resolve);
            }),
        ),
      }),
    );
    const first = ctl.loadCells({ zoom: 3 });
    const second = ctl.loadCells({ zoom: 4 });
    expect(signals[0].aborted).toBe(true);
    answers[1]({ ...CELLS, precision: 7 });
    answers[0]({ ...CELLS, precision: 6 });
    await Promise.all([first, second]);
    expect(ctl.cells?.precision).toBe(7);
    expect(ctl.loadingCells).toBe(false);
  });

  it('a cell click focuses the cell and asks for its summary with the cell bounds', async () => {
    const h = host();
    const d = deps();
    const ctl = new MapController(h, d);
    await ctl.focusCell(CELL);
    expect(h.setFocus).toHaveBeenCalledWith({ kind: 'cell', id: '9/267/179' });
    expect(d.summary).toHaveBeenCalledWith(
      { filter: { class_names: ['widget'], review_tab: 'unreviewed', geo_bbox: BOUNDS } },
      expect.any(AbortSignal),
    );
    expect(d.images).toHaveBeenCalledTimes(1);
    expect(d.images).toHaveBeenCalledWith(
      {
        class_name: ['widget'],
        review_tab: 'unreviewed',
        geo_bbox: '8.4375,47.5,8.4551,47.51',
        limit: CELL_CARDS,
      },
      null,
      expect.any(AbortSignal),
    );
    expect(ctl.summary).toEqual(SUMMARY);
    expect(ctl.cell).toEqual(CELL);
  });

  it('a pin opens the image details and drops the cell', async () => {
    const h = host();
    const ctl = new MapController(h, deps());
    await ctl.focusCell(CELL);
    ctl.focusPin('img-9');
    expect(h.setFocus).toHaveBeenLastCalledWith({ kind: 'image', id: 'img-9' });
    expect(ctl.cell).toBeNull();
  });

  it('a filter change reloads the last viewport and the focused cell', async () => {
    const d = deps();
    const ctl = new MapController(host(), d);
    await ctl.loadCells({ zoom: 5, bbox: '1,2,3,4' });
    await ctl.focusCell(CELL);
    await ctl.refresh();
    expect(d.cells).toHaveBeenCalledTimes(2);
    expect(d.summary).toHaveBeenCalledTimes(2);
  });

  it('shows the served error when a cell cannot be summarised', async () => {
    const ctl = new MapController(
      host(),
      deps({ summary: vi.fn().mockRejectedValue(new Error('nope')) }),
    );
    await ctl.focusCell(CELL);
    expect((ctl.cellFailure as Error).message).toBe('nope');
    expect(ctl.summary).toBeNull();
  });

  it('retries the focused cell after a failure and clears the failure', async () => {
    const summary = vi
      .fn()
      .mockRejectedValueOnce(new Error('nope'))
      .mockResolvedValue({ images: 3, summary_text: 'ok' });
    const ctl = new MapController(host(), deps({ summary }));
    await ctl.focusCell(CELL);
    expect(ctl.cellFailure).not.toBeNull();
    await ctl.retryCell();
    expect(ctl.cellFailure).toBeNull();
    expect(ctl.summary?.images).toBe(3);
  });

  it('show-in applies the cell bounds as the area filter and the view in one navigation', () => {
    const h = host();
    new MapController(h, deps()).showCell(CELL, 'gallery');
    expect(h.filter.geoBbox).toEqual(BOUNDS);
    expect(h.applyFilter).toHaveBeenCalledWith('gallery');
  });

  it('the no-location link filters to photos without a position', () => {
    const h = host();
    new MapController(h, deps()).showWithoutLocation();
    expect(h.filter.hasLocation).toBe(false);
    expect(h.applyFilter).toHaveBeenCalledWith();
  });
});
