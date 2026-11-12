import { describe, expect, it } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import Icon from './Icon.svelte';
import { ICONS } from './iconRegistry';

describe('Icon', () => {
  it('is an SVG hidden from assistive tech when it has no label', () => {
    const svg = mountUi(Icon, { name: 'check' }).querySelector('svg')!;
    expect(svg).not.toBeNull();
    expect(svg.getAttribute('aria-hidden')).toBe('true');
    expect(svg.getAttribute('width')).toBe('16');
  });

  it('is a named image when it has a label, at the requested size and class', () => {
    const svg = mountUi(Icon, {
      name: 'x',
      label: 'Close',
      size: 12,
      class: 'text-fg-2',
    }).querySelector('svg')!;
    expect(svg.getAttribute('role')).toBe('img');
    expect(svg.getAttribute('aria-label')).toBe('Close');
    expect(svg.getAttribute('height')).toBe('12');
    expect(svg.getAttribute('class')).toContain('text-fg-2');
    expect(svg.getAttribute('class')).toContain('shrink-0');
  });

  it('draws a different shape for each name', () => {
    const shapes = new Set(
      (Object.keys(ICONS) as Array<keyof typeof ICONS>).map(
        (name) => mountUi(Icon, { name }).querySelector('svg')!.innerHTML,
      ),
    );
    expect(shapes.size).toBe(Object.keys(ICONS).length);
  });
});
