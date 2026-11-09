import { describe, expect, it, vi } from 'vitest';
import { MapController, OPEN_CELL_MAX, type MapDeps } from './mapController.svelte';
import type { MapCellCollection, MapMeta, MapTileImages } from '$lib/types_map';

const META: MapMeta = {
  available: true,
  location_policy: 'store',
  n_located: 3,
  bounds: { west: 7, south: 46, east: 9, north: 48 },
  tile_url: 'https://tiles.example/{z}/{x}/{y}.png',
  tile_attribution: 'osm',
  max_zoom: 12,
};
const CELLS: MapCellCollection = {
  type: 'FeatureCollection',
  features: [],
  precision: 9,
  truncated: false,
  join_truncated: false,
};
const IMAGES: MapTileImages = {
  tile: '9/267/179',
  count: 2,
  image_ids: ['a', 'b'],
  truncated: false,
};

function deps(over: Partial<MapDeps> = {}): MapDeps {
  return {
    meta: vi.fn().mockResolvedValue(META),
    cells: vi.fn().mockResolvedValue(CELLS),
    images: vi.fn().mockResolvedValue(IMAGES),
    ...over,
  };
}

describe('MapController', () => {
  it('loads the served meta', async () => {
    const ctl = new MapController(deps());
    await ctl.loadMeta();
    expect(ctl.meta).toEqual(META);
    expect(ctl.loadingMeta).toBe(false);
  });

  it('shows the served error when the meta read fails, never a made-up map', async () => {
    const ctl = new MapController(
      deps({ meta: vi.fn().mockRejectedValue(new Error('boom')) }),
    );
    await ctl.loadMeta();
    expect(ctl.meta).toBeNull();
    expect(ctl.metaError).toBe('boom');
  });

  it('sends the shared filter and the capture days with every cells request', async () => {
    const d = deps();
    const ctl = new MapController(d);
    ctl.filter.setValue('class_name', ['class_a']);
    ctl.takenFrom = '2024-03-05';
    await ctl.loadCells({ zoom: 6, bbox: '7,46,9,48' });
    expect(d.cells).toHaveBeenCalledWith(
      { zoom: 6, bbox: '7,46,9,48' },
      { class_name: ['class_a'], taken_from: '2024-03-05' },
      expect.any(AbortSignal),
    );
    expect(ctl.cells).toEqual(CELLS);
  });

  it('a newer viewport cancels the older request and keeps only the newer answer', async () => {
    const answers: Array<(v: MapCellCollection) => void> = [];
    const signals: AbortSignal[] = [];
    const d = deps({
      cells: vi.fn().mockImplementation(
        (_v, _f, signal: AbortSignal) =>
          new Promise<MapCellCollection>((resolve) => {
            signals.push(signal);
            answers.push(resolve);
          }),
      ),
    });
    const ctl = new MapController(d);
    const first = ctl.loadCells({ zoom: 3 });
    const second = ctl.loadCells({ zoom: 4 });
    expect(signals[0].aborted).toBe(true);
    answers[1]({ ...CELLS, precision: 7 });
    answers[0]({ ...CELLS, precision: 6 });
    await Promise.all([first, second]);
    expect(ctl.cells?.precision).toBe(7);
    expect(ctl.loadingCells).toBe(false);
  });

  it('a filter change reloads the last viewport and closes the open cell', async () => {
    const d = deps();
    const ctl = new MapController(d);
    await ctl.loadCells({ zoom: 5, bbox: '1,2,3,4' });
    await ctl.openCell('9/1/1');
    ctl.takenTo = '2024-03-06';
    await ctl.refresh();
    expect(ctl.openTile).toBeNull();
    expect(d.cells).toHaveBeenCalledTimes(2);
    expect(vi.mocked(d.cells).mock.calls[1][0]).toEqual({ zoom: 5, bbox: '1,2,3,4' });
  });

  it('opens the list of a small cell and zooms in on a large one, except at the deepest zoom', async () => {
    const ctl = new MapController(deps());
    await ctl.loadMeta();
    expect(ctl.opensList(OPEN_CELL_MAX, 4)).toBe(true);
    expect(ctl.opensList(OPEN_CELL_MAX + 1, 4)).toBe(false);
    expect(ctl.opensList(OPEN_CELL_MAX + 1, 12)).toBe(true);
  });

  it("narrows the list to the open cell's photos, in both filter spellings", async () => {
    const ctl = new MapController(deps());
    ctl.takenFrom = '2024-03-05';
    await ctl.openCell('9/267/179');
    expect(ctl.listQuery()).toEqual({ taken_from: '2024-03-05', image_id: ['a', 'b'] });
    expect(ctl.listBody()).toEqual({ taken_from: '2024-03-05', image_ids: ['a', 'b'] });
  });

  it('shows the served error when a cell cannot be opened', async () => {
    const ctl = new MapController(
      deps({ images: vi.fn().mockRejectedValue(new Error('nope')) }),
    );
    await ctl.openCell('9/1/1');
    expect(ctl.openTile).toBeNull();
    expect(ctl.openError).toBe('nope');
  });
});
