import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import type { XYXY } from '$lib/annotations/types';
import { flushSync, mount, unmount } from 'svelte';
import { imageBox } from '$lib/test/fixtures/imageDetail';
import { pointer as rawPointer, stubCanvasLayout } from '$lib/test/canvasLayout';
import type { ImageBox } from '$lib/types_images';
import EditorCanvas, { type EditorTool } from './EditorCanvas.svelte';

const W = 800;
const H = 600;

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

beforeEach(() => stubCanvasLayout(W, H));

afterEach(() => {
  vi.restoreAllMocks();
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const ITEM: ImageBox = imageBox({ crop_id: 'c1', bbox_norm: [0.2, 0.2, 0.6, 0.6] });

interface Calls {
  ondraw: Mock<(box: XYXY) => void>;
  onmove: Mock<(cropId: string, box: XYXY) => void>;
  onselect: Mock<(cropId: string | null) => void>;
  ongesture: Mock<(active: boolean) => void>;
}

function render(
  over: {
    tool?: EditorTool;
    boxes?: ImageBox[];
    selectedCropId?: string | null;
    locks?: Record<string, string | null>;
    pending?: [number, number, number, number] | null;
  } = {},
): Calls & { canvas: Record<string, (...a: never[]) => unknown> } {
  const calls: Calls = {
    ondraw: vi.fn<(box: XYXY) => void>(),
    onmove: vi.fn<(cropId: string, box: XYXY) => void>(),
    onselect: vi.fn<(cropId: string | null) => void>(),
    ongesture: vi.fn<(active: boolean) => void>(),
  };
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(EditorCanvas, {
    target,
    props: {
      imageUrl: (d: number) => `/img?d=${d}`,
      width: W,
      height: H,
      boxes: over.boxes ?? [ITEM],
      selectedCropId: over.selectedCropId ?? null,
      locks: over.locks ?? {},
      tool: over.tool ?? 'select',
      pending: over.pending ?? null,
      showLabels: true,
      ...calls,
    },
  });
  flushSync();
  return { ...calls, canvas: instance as never };
}

const canvasEl = () =>
  target.querySelector('[data-testid="editor-canvas"]') as HTMLElement;
const frameEl = () => target.querySelector('[data-testid="editor-frame"]') as HTMLElement;

function pointer(el: Element, type: string, x: number, y: number): void {
  rawPointer(el, type, x, y);
  flushSync();
}

function px(el: HTMLElement, name: 'left' | 'top' | 'width' | 'height'): number {
  return parseFloat(el.style.getPropertyValue(name)) || 0;
}

describe('EditorCanvas drawing', () => {
  it('a drag draws a clamped box', () => {
    const c = render({ tool: 'draw' });
    pointer(canvasEl(), 'pointerdown', 100, 150);
    pointer(canvasEl(), 'pointermove', 500, 300);
    expect(target.querySelector('[data-testid="draft-box"]')).not.toBeNull();
    pointer(canvasEl(), 'pointermove', 1200, 900);
    pointer(canvasEl(), 'pointerup', 1200, 900);
    expect(c.ondraw).toHaveBeenCalledTimes(1);
    expect(c.ondraw).toHaveBeenCalledWith([0.125, 0.25, 1, 1]);
    expect(target.querySelector('[data-testid="draft-box"]')).toBeNull();
  });

  it('two clicks draw a box', () => {
    const c = render({ tool: 'draw' });
    pointer(canvasEl(), 'pointerdown', 80, 60);
    pointer(canvasEl(), 'pointerup', 80, 60);
    expect(c.ondraw).not.toHaveBeenCalled();
    expect(c.ongesture).toHaveBeenLastCalledWith(true);
    pointer(canvasEl(), 'pointermove', 400, 300);
    expect(target.querySelector('[data-testid="draft-box"]')).not.toBeNull();
    pointer(canvasEl(), 'pointerdown', 400, 300);
    pointer(canvasEl(), 'pointerup', 400, 300);
    expect(c.ondraw).toHaveBeenCalledOnce();
    expect(c.ondraw).toHaveBeenCalledWith([0.1, 0.1, 0.5, 0.5]);
    expect(c.ongesture).toHaveBeenLastCalledWith(false);
  });

  it('a click that never grows into a box sends nothing', () => {
    const c = render({ tool: 'draw' });
    pointer(canvasEl(), 'pointerdown', 80, 60);
    pointer(canvasEl(), 'pointerup', 80, 60);
    pointer(canvasEl(), 'pointerdown', 80, 60);
    expect(c.ondraw).not.toHaveBeenCalled();
  });

  it('cancelGesture drops a half-drawn box and says whether there was one', () => {
    const c = render({ tool: 'draw' });
    expect(c.canvas.cancelGesture()).toBe(false);
    pointer(canvasEl(), 'pointerdown', 100, 100);
    pointer(canvasEl(), 'pointermove', 300, 300);
    expect(c.canvas.cancelGesture()).toBe(true);
    flushSync();
    expect(target.querySelector('[data-testid="draft-box"]')).toBeNull();
    pointer(canvasEl(), 'pointerup', 300, 300);
    expect(c.ondraw).not.toHaveBeenCalled();
  });
});

describe('EditorCanvas moving and resizing', () => {
  it('a handle drag resizes and reports once, at the end', () => {
    const c = render({ selectedCropId: 'c1' });
    const se = target.querySelector('[data-handle="se"]')!;
    pointer(se, 'pointerdown', 480, 360);
    pointer(canvasEl(), 'pointermove', 520, 390);
    pointer(canvasEl(), 'pointermove', 560, 420);
    expect(c.onmove).not.toHaveBeenCalled();
    pointer(canvasEl(), 'pointerup', 560, 420);
    expect(c.onmove).toHaveBeenCalledTimes(1);
    const [id, box] = c.onmove.mock.calls[0] as [string, number[]];
    expect(id).toBe('c1');
    expect(box.map((n) => +n.toFixed(6))).toEqual([0.2, 0.2, 0.7, 0.7]);
  });

  it('dragging the box moves it without changing its size', () => {
    const c = render({ selectedCropId: 'c1' });
    const body = target.querySelector('[data-testid="selected-box"]')!;
    pointer(body, 'pointerdown', 300, 200);
    pointer(canvasEl(), 'pointermove', 380, 260);
    pointer(canvasEl(), 'pointerup', 380, 260);
    const [, box] = c.onmove.mock.calls[0] as [string, number[]];
    expect(box.map((n) => +n.toFixed(6))).toEqual([0.3, 0.3, 0.7, 0.7]);
  });

  it('a click on the selected box does not send a write', () => {
    const c = render({ selectedCropId: 'c1' });
    const body = target.querySelector('[data-testid="selected-box"]')!;
    pointer(body, 'pointerdown', 300, 200);
    pointer(canvasEl(), 'pointerup', 300, 200);
    expect(c.onmove).not.toHaveBeenCalled();
  });

  it('a locked box shows its lock and has no handles or move', () => {
    const c = render({
      selectedCropId: 'c1',
      locks: { c1: 'Placed by a person' },
    });
    expect(target.querySelectorAll('[data-handle]')).toHaveLength(0);
    const body = target.querySelector('[data-testid="selected-box"]')!;
    expect(body.getAttribute('title')).toBe('Placed by a person');
    pointer(body, 'pointerdown', 300, 200);
    pointer(canvasEl(), 'pointermove', 380, 260);
    pointer(canvasEl(), 'pointerup', 380, 260);
    expect(c.onmove).not.toHaveBeenCalled();
  });

  it('draws eight handles on an unlocked selected box', () => {
    render({ selectedCropId: 'c1' });
    expect(target.querySelectorAll('[data-handle]')).toHaveLength(8);
  });

  it('a click on the empty picture deselects', () => {
    const c = render({ selectedCropId: 'c1' });
    pointer(canvasEl(), 'pointerdown', 700, 500);
    expect(c.onselect).toHaveBeenCalledWith(null);
  });

  it('a click on another box selects it', () => {
    const other = imageBox({ crop_id: 'c2', bbox_norm: [0.7, 0.7, 0.9, 0.9] });
    const c = render({ boxes: [ITEM, other], selectedCropId: 'c1' });
    (target.querySelector('[data-crop-id="c2"]') as HTMLElement).click();
    expect(c.onselect).toHaveBeenCalledWith('c2');
  });
});

describe('EditorCanvas zoom and pan', () => {
  it('wheel zooms at the cursor', () => {
    render();
    const before = canvasEl();
    const ev = new WheelEvent('wheel', {
      deltaY: -400,
      clientX: 200,
      clientY: 150,
      bubbles: true,
      cancelable: true,
    });
    before.dispatchEvent(ev);
    flushSync();
    expect(ev.defaultPrevented).toBe(true);
    const f = frameEl();
    expect(px(f, 'width')).toBeGreaterThan(W);
    // The picture point under the cursor (a quarter in each way) has not moved.
    expect((200 - px(f, 'left')) / px(f, 'width')).toBeCloseTo(0.25, 3);
    expect((150 - px(f, 'top')) / px(f, 'height')).toBeCloseTo(0.25, 3);
  });

  it('zoomFit returns to the fitted picture', () => {
    const c = render();
    c.canvas.zoomBy(3 as never);
    flushSync();
    expect(px(frameEl(), 'width')).toBeGreaterThan(W);
    c.canvas.zoomFit();
    flushSync();
    expect(px(frameEl(), 'width')).toBe(W);
    expect(px(frameEl(), 'left')).toBe(0);
  });

  it('zoomActual shows one picture pixel per screen pixel', () => {
    const c = render();
    c.canvas.zoomActual();
    flushSync();
    expect(px(frameEl(), 'width')).toBe(W);
  });

  it('asks for a sharper picture once zoomed past the one on screen', () => {
    const c = render();
    expect(target.querySelector('img')!.getAttribute('src')).toBe('/img?d=1280');
    c.canvas.zoomBy(4 as never);
    flushSync();
    expect(target.querySelector('img')!.getAttribute('src')).toBe('/img?d=4096');
  });

  it('the pan tool drags a zoomed picture', () => {
    const c = render({ tool: 'pan' });
    c.canvas.zoomBy(2 as never);
    flushSync();
    const left = px(frameEl(), 'left');
    pointer(canvasEl(), 'pointerdown', 400, 300);
    pointer(canvasEl(), 'pointermove', 340, 300);
    pointer(canvasEl(), 'pointerup', 340, 300);
    expect(px(frameEl(), 'left')).toBe(left - 60);
  });

  it('holding space pans temporarily', () => {
    const c = render();
    c.canvas.zoomBy(2 as never);
    flushSync();
    const left = px(frameEl(), 'left');
    canvasEl().dispatchEvent(
      new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true }),
    );
    flushSync();
    pointer(canvasEl(), 'pointerdown', 400, 300);
    pointer(canvasEl(), 'pointermove', 350, 300);
    pointer(canvasEl(), 'pointerup', 350, 300);
    expect(px(frameEl(), 'left')).toBe(left - 50);
    canvasEl().dispatchEvent(
      new KeyboardEvent('keyup', { key: ' ', code: 'Space', bubbles: true }),
    );
  });

  it('reports the zoom as a percentage of the picture pixels', () => {
    const onzoom = vi.fn();
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(EditorCanvas, {
      target,
      props: {
        imageUrl: () => '/img',
        width: 1600,
        height: 1200,
        boxes: [],
        selectedCropId: null,
        locks: {},
        tool: 'select',
        showLabels: true,
        ondraw: vi.fn(),
        onmove: vi.fn(),
        onselect: vi.fn(),
        onzoom,
      },
    });
    flushSync();
    expect(onzoom).toHaveBeenLastCalledWith(50);
  });
});

describe('EditorCanvas pending box', () => {
  it('keeps a box waiting for its class on screen', () => {
    render({ pending: [0.2, 0.2, 0.4, 0.4] });
    expect(target.querySelector('[data-testid="pending-box"]')).not.toBeNull();
  });
});

describe('EditorCanvas labels', () => {
  it('shows the selected box label, and hides it when labels are off', () => {
    render({ selectedCropId: 'c1' });
    expect(target.textContent).toContain('widget_a');
    unmount(instance!);
    target.remove();
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(EditorCanvas, {
      target,
      props: {
        imageUrl: () => '/img',
        width: W,
        height: H,
        boxes: [ITEM],
        selectedCropId: 'c1',
        locks: {},
        tool: 'select',
        showLabels: false,
        ondraw: vi.fn(),
        onmove: vi.fn(),
        onselect: vi.fn(),
      },
    });
    flushSync();
    expect(target.textContent).not.toContain('widget_a');
  });
});

describe('EditorCanvas keyboard', () => {
  it('registers no listener on window', () => {
    const add = vi.spyOn(window, 'addEventListener');
    render();
    expect(add.mock.calls.filter(([type]) => type === 'keydown')).toEqual([]);
    expect(add.mock.calls.filter(([type]) => String(type).startsWith('pointer'))).toEqual(
      [],
    );
  });
});
