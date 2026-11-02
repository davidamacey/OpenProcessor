/** The picker hands the chosen picture, and the region dragged on it, to the search. */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ImageSearchPicker from './ImageSearchPicker.svelte';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;
const onSearch = vi.fn();

beforeEach(() => {
  URL.createObjectURL = vi.fn(() => 'blob:preview');
  URL.revokeObjectURL = vi.fn();
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ImageSearchPicker, { target, props: { onSearch } });
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
  onSearch.mockReset();
});

function q(id: string): HTMLElement {
  return target.querySelector(`[data-testid="${id}"]`) as HTMLElement;
}

function choose(file: File): void {
  q('image-search-toggle').click();
  flushSync();
  const input = q('image-search-file') as HTMLInputElement;
  Object.defineProperty(input, 'files', { value: [file], configurable: true });
  input.dispatchEvent(new Event('change', { bubbles: true }));
  flushSync();
}

const png = new File([new Uint8Array([1])], 'q.png', { type: 'image/png' });

describe('ImageSearchPicker', () => {
  it('starts closed and cannot search before a picture is chosen', () => {
    expect(q('image-search-panel')).toBeNull();
    q('image-search-toggle').click();
    flushSync();
    expect((q('image-search-run') as HTMLButtonElement).disabled).toBe(true);
  });

  it('stays inside a narrow viewport instead of hanging off the button', () => {
    q('image-search-toggle').click();
    flushSync();
    // Anchored to the button the 20rem panel runs past a 430px screen's right edge.
    const classes = q('image-search-panel').className.split(/\s+/);
    expect(classes).toContain('max-sm:fixed');
    expect(classes).toContain('max-sm:inset-x-2');
    // top-full on a fixed box would be a full viewport below the button, off screen.
    expect(classes).toContain('max-sm:top-14');
    // A tall picture must scroll inside the panel, not push Search off the screen.
    expect(classes).toContain('max-sm:overflow-y-auto');
  });

  it('searches by the whole picture when no region is drawn', () => {
    choose(png);
    q('image-search-run').click();
    expect(onSearch).toHaveBeenCalledWith(png, null);
  });

  it('ignores a file that is not a picture', () => {
    choose(new File(['x'], 'a.txt', { type: 'text/plain' }));
    expect(q('image-search-preview')).toBeNull();
  });

  it('sends a dragged region as fractions of the preview', () => {
    choose(png);
    const preview = q('image-search-preview');
    preview.getBoundingClientRect = () =>
      ({ left: 10, top: 20, width: 200, height: 100 }) as DOMRect;
    preview.setPointerCapture = vi.fn();
    const ev = (type: string, x: number, y: number) => {
      const e = new Event(type, { bubbles: true }) as Event & Record<string, number>;
      e.clientX = x;
      e.clientY = y;
      e.pointerId = 1;
      return e;
    };
    preview.dispatchEvent(ev('pointerdown', 110, 45));
    preview.dispatchEvent(ev('pointermove', 210, 120));
    preview.dispatchEvent(ev('pointerup', 210, 120));
    flushSync();
    q('image-search-run').click();
    const [, box] = onSearch.mock.calls[0];
    expect(box[0]).toBeCloseTo(0.5);
    expect(box[1]).toBeCloseTo(0.25);
    expect(box[2]).toBeCloseTo(1);
    expect(box[3]).toBeCloseTo(1);
  });
});
