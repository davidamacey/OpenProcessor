/**
 * The crop-frame candidate view: the served parent-frame geometry drawn as
 * an SVG layer over the crop thumbnail, dropped candidates faint. Nothing
 * is projected here.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { candidateShapes } from '$lib/configTest/overlayShapes';
import { legsFixture } from '$lib/test/fixtures/configTest';
import CropFrameShapes from './CropFrameShapes.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(shapes = candidateShapes(legsFixture(), 'parent')) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropFrameShapes, { target, props: { cropId: 'c_123', shapes } });
  flushSync();
  return target;
}

describe('CropFrameShapes', () => {
  it('draws the crop thumbnail with a unit-box SVG layer', () => {
    const el = render();
    expect(el.querySelector('img')?.getAttribute('src')).toContain(
      '/crops/c_123/thumbnail',
    );
    const svg = el.querySelector('svg')!;
    expect(svg.getAttribute('viewBox')).toBe('0 0 1 1');
    expect(svg.getAttribute('preserveAspectRatio')).toBe('none');
  });

  it('draws each served box at its parent-frame coordinates, dropped ones dimmed', () => {
    const el = render();
    const rects = [...el.querySelectorAll('[data-testid="crop-frame-box"]')];
    expect(rects).toHaveLength(3);
    expect(rects[0]!.getAttribute('x')).toBe('0.1');
    expect(rects[0]!.getAttribute('width')).toBe('0.8');
    expect(rects[0]!.getAttribute('opacity')).toBe('1');
    expect(rects[0]!.getAttribute('data-dimmed')).toBe('false');
    expect(rects[1]!.getAttribute('x')).toBe('0.6');
    expect(rects[1]!.getAttribute('data-dimmed')).toBe('true');
    expect(rects[1]!.getAttribute('opacity')).toBe('0.4');
    expect(rects[1]!.querySelector('title')?.textContent).toContain('Below min score');
  });

  it('draws the mask polygon from the parent-frame points', () => {
    const el = render();
    const poly = el.querySelector('[data-testid="crop-frame-polygon"]')!;
    expect(poly.getAttribute('points')).toBe('0.1,0.1 0.9,0.1 0.5,0.9');
    expect(poly.getAttribute('data-dimmed')).toBe('false');
  });

  it('draws nothing but the thumbnail for no shapes', () => {
    const el = render([]);
    expect(el.querySelector('[data-testid="crop-frame-box"]')).toBeNull();
    expect(el.querySelector('[data-testid="crop-frame-polygon"]')).toBeNull();
  });
});
