/**
 * Mount-based test for the CropDetailModal narrow-viewport layout fix
 * (visual-audit follow-up, 2026-09-25): at 800px the modal used to switch
 * to its two-column layout at the `md` (768px) breakpoint, and the left
 * "Source" column stayed `flex-1` — sized to the shared grid row's height
 * (driven by the taller meta column) rather than its own content. In a
 * narrow column that produced a tall, mostly-empty black box around a
 * small `object-contain`'d image. The fix: the two-column split now
 * starts at `lg` (1024px) so 800px stays single-column/stacked (each
 * column sized to its own content, no shared-row stretch), and the source
 * image area caps its own height with `max-h` below `lg` instead of
 * `flex-1`, so a two-column layout narrower than 1024px (if one ever
 * exists) can't reintroduce the same stretch. `lg:flex-1`/`lg:max-h-[78vh]`
 * restore the original 1600px-verified sizing at the two-column breakpoint.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import CropDetailModal from './CropDetailModal.svelte';
import type { Crop } from '$lib/types';

let target: HTMLDivElement;
let instance: unknown;

function crop(over: Partial<Crop> = {}): Crop {
  return {
    id: 'crop-1',
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.5, h: 0.5 },
    class_id: null,
    class_name: null,
    cluster_id: 7,
    label_confidence: null,
    label_source: null,
    class_source: null,
    updated_at: '',
    ...over,
  } as Crop;
}

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  vi.unstubAllGlobals();
});

function render(): HTMLDivElement {
  // CropMetaPanel and SourceImageOverlay both fire fetches on mount; this
  // test only looks at the modal's own layout classes.
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network in test')));
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropDetailModal, {
    target,
    props: { crop: crop(), onclose: () => {} },
  } as never);
  flushSync();
  return target;
}

describe('CropDetailModal narrow-viewport layout (visual-audit follow-up)', () => {
  it('splits into two columns at lg, not md — 800px (below lg) stays single-column', () => {
    const el = render();
    const grid = el.querySelector('[role="dialog"] > div');
    expect(grid?.className).toContain('lg:grid-cols-[1fr_320px]');
    expect(grid?.className).not.toContain('md:grid-cols-[1fr_320px]');
    expect(grid?.className).toContain('grid-cols-1');
  });

  it('the source image area has a definite height and no stretched black backdrop', () => {
    const el = render();
    const box = el.querySelector('[data-testid="source-box"]') as HTMLElement | null;
    expect(box).not.toBeNull();
    expect(box!.className).toContain('h-[50vh]');
    expect(box!.className).toContain('lg:h-[62vh]');
    expect(box!.className).not.toContain('bg-black');
    expect(box!.className).not.toMatch(/\bflex-1\b/);
  });
});
