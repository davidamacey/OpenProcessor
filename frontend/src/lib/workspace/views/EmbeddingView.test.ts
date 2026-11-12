/**
 * The Embedding view mounted for real (jsdom) over a stubbed `GET /viz/projection`: the served
 * legend, points the filter leaves out drawn dimmed, a lasso that selects only matching items,
 * a click that focuses an item, and a drawn box that becomes a `viz_box` filter chip.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { computeScale } from '$lib/embeddingPlot';
import { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';
import { NO_SELECTION } from '$lib/workspace/selection';
import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

const getVizProjection = vi.fn();

vi.mock('$lib/api_viz', async () => {
  const actual = await vi.importActual<typeof import('$lib/api_viz')>('$lib/api_viz');
  return { ...actual, getVizProjection: (...a: unknown[]) => getVizProjection(...a) };
});
vi.mock('$components/workspace/BulkBar.svelte', () => ({ default: () => {} }));

const { default: EmbeddingView } = await import('./EmbeddingView.svelte');

const WIDTH = 800;
const HEIGHT = 520;
const PADDING = 20;

const POINTS = [
  { crop_id: 'in-1', x: 0, y: 0, matched: true, colour_index: 0, class_name: 'gizmo' },
  { crop_id: 'in-2', x: 1, y: 1, matched: true, colour_index: 0, class_name: 'gizmo' },
  { crop_id: 'out-1', x: 9, y: 9, matched: false, colour_index: 1, class_name: 'truck' },
].map((p) => ({ cluster_uid: null, class_source: null, ...p }));
const LEGEND = [
  { class_name: 'gizmo', colour_index: 0, count: 2 },
  { class_name: 'truck', colour_index: 1, count: 1 },
];
const SCALE = computeScale(POINTS, WIDTH, HEIGHT, PADDING);

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;
const fills: { alpha: number; style: string }[] = [];

function fakeWs() {
  return {
    filter: new ItemFilterState(),
    selection: NO_SELECTION,
    itemQuery: () => ({ class_name: ['gizmo'] }),
    select: vi.fn(),
    applyFilter: vi.fn(),
    setFocus: vi.fn(),
  };
}

async function open(ws = fakeWs()) {
  instance = mount(EmbeddingView, {
    target,
    props: { ws: ws as unknown as WorkspaceController },
  });
  await vi.waitFor(() => {
    flushSync();
    expect(target.querySelector('canvas')).not.toBeNull();
  });
  return { ws, canvas: target.querySelector('canvas')! };
}

function pointer(
  canvas: HTMLElement,
  type: string,
  at: { x: number; y: number },
  init: MouseEventInit = {},
): void {
  canvas.dispatchEvent(
    new MouseEvent(type, {
      bubbles: true,
      button: 0,
      clientX: at.x,
      clientY: at.y,
      ...init,
    }),
  );
  flushSync();
}

beforeEach(() => {
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
    },
  );
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(WIDTH);
  fills.length = 0;
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function () {
    const ctx: Record<string, unknown> = {
      globalAlpha: 1,
      fillStyle: '',
      fill() {
        fills.push({ alpha: ctx.globalAlpha as number, style: String(ctx.fillStyle) });
      },
    };
    for (const fn of [
      'clearRect',
      'fillRect',
      'beginPath',
      'arc',
      'stroke',
      'strokeRect',
      'setLineDash',
      'moveTo',
      'lineTo',
    ]) {
      ctx[fn] = () => {};
    }
    return ctx as unknown as CanvasRenderingContext2D;
  } as never);
  getVizProjection.mockResolvedValue({
    built: true,
    points: POINTS,
    legend: LEGEND,
    total: POINTS.length,
    stale: false,
    fitted_at: null,
    projection_version: null,
  });
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  getVizProjection.mockReset();
});

describe('Embedding view', () => {
  it('asks the projection with the Workspace query', async () => {
    await open();
    expect(getVizProjection.mock.calls[0]![0]).toEqual({ class_name: ['gizmo'] });
  });

  it('lists the served legend with its served counts', async () => {
    await open();
    const entries = [...target.querySelectorAll('[data-testid=embedding-legend-entry]')];
    expect(entries.map((e) => e.textContent?.replace(/\s+/g, ' ').trim())).toEqual([
      'gizmo (2)',
      'truck (1)',
    ]);
  });

  it('unmatched points render dimmed', async () => {
    await open();
    const dim = fills.filter((f) => f.alpha < 0.2);
    const bright = fills.filter((f) => f.alpha > 0.5);
    expect(dim).toHaveLength(1);
    expect(bright).toHaveLength(2);
  });

  it('a drawn box becomes a viz_box chip', async () => {
    const { ws, canvas } = await open();
    const from = SCALE.toScreen(-0.5, 1.5);
    const to = SCALE.toScreen(1.5, -0.5);
    pointer(canvas, 'pointerdown', from, { shiftKey: true });
    pointer(canvas, 'pointermove', to);
    pointer(canvas, 'pointerup', to);
    expect(ws.filter.vizBox).not.toBeNull();
    const [x1, y1, x2, y2] = ws.filter.vizBox!;
    expect([x1, y1, x2, y2].map((v) => Math.round(v * 10) / 10)).toEqual([
      -0.5, -0.5, 1.5, 1.5,
    ]);
    expect(ws.filter.chips().map((c) => c.param)).toContain('viz_box');
    expect(ws.applyFilter).toHaveBeenCalledTimes(1);
  });

  it('the "Draw a box" button arms a plain drag as a box', async () => {
    const { ws, canvas } = await open();
    (target.querySelector('[data-testid=embedding-box-tool]') as HTMLElement).click();
    flushSync();
    pointer(canvas, 'pointerdown', { x: 100, y: 100 });
    pointer(canvas, 'pointermove', { x: 300, y: 300 });
    pointer(canvas, 'pointerup', { x: 300, y: 300 });
    expect(ws.filter.vizBox).not.toBeNull();
    expect(ws.select).not.toHaveBeenCalled();
  });

  it('a lasso selects the matching items only', async () => {
    const { ws, canvas } = await open();
    const a = SCALE.toScreen(-1, 2);
    const path = [a, SCALE.toScreen(2, 2), SCALE.toScreen(2, -1), SCALE.toScreen(-1, -1)];
    pointer(canvas, 'pointerdown', path[0]!);
    for (const p of path.slice(1)) pointer(canvas, 'pointermove', p);
    pointer(canvas, 'pointerup', path[3]!);
    expect(ws.select).toHaveBeenCalledWith({
      kind: 'items',
      cropIds: ['in-1', 'in-2'],
    });
  });

  it('a lasso around only a dimmed point selects nothing', async () => {
    const { ws, canvas } = await open();
    const o = SCALE.toScreen(9, 9);
    const path = [
      { x: o.x - 15, y: o.y - 15 },
      { x: o.x + 15, y: o.y - 15 },
      { x: o.x + 15, y: o.y + 15 },
      { x: o.x - 15, y: o.y + 15 },
    ];
    pointer(canvas, 'pointerdown', path[0]!);
    for (const p of path.slice(1)) pointer(canvas, 'pointermove', p);
    pointer(canvas, 'pointerup', path[3]!);
    expect(ws.select).toHaveBeenCalledWith(NO_SELECTION);
  });

  it('a click on a dot focuses its item', async () => {
    const { ws, canvas } = await open();
    const at = SCALE.toScreen(1, 1);
    pointer(canvas, 'pointerdown', at);
    pointer(canvas, 'pointerup', at);
    expect(ws.setFocus).toHaveBeenCalledWith({ kind: 'item', id: 'in-2' });
  });
});
