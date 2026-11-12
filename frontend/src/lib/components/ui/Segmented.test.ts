import { describe, expect, it, vi } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import Segmented from './Segmented.svelte';

const options = [
  { value: 'grid', label: 'Grid' },
  { value: 'list', label: 'List' },
  { value: 'map', label: 'Map', disabled: true },
];

describe('Segmented', () => {
  it('is a radiogroup with one checked radio, the only tab stop', () => {
    const el = mountUi(Segmented, {
      options,
      value: 'list',
      label: 'View',
      onchange: vi.fn(),
    });
    expect(el.querySelector('[role="radiogroup"]')!.getAttribute('aria-label')).toBe(
      'View',
    );
    const radios = el.querySelectorAll<HTMLButtonElement>('[role="radio"]');
    expect([...radios].map((r) => r.getAttribute('aria-checked'))).toEqual([
      'false',
      'true',
      'false',
    ]);
    expect([...radios].map((r) => r.tabIndex)).toEqual([-1, 0, -1]);
    expect(radios[1]!.className).toContain('text-accent-text');
  });

  it('a click chooses, and the arrow keys move over the enabled options only', () => {
    const onchange = vi.fn();
    const el = mountUi(Segmented, { options, value: 'list', label: 'View', onchange });
    el.querySelectorAll<HTMLButtonElement>('[role="radio"]')[0]!.click();
    expect(onchange).toHaveBeenLastCalledWith('grid');
    const group = el.querySelector('[role="radiogroup"]')!;
    group.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }),
    );
    expect(onchange).toHaveBeenLastCalledWith('grid');
    group.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }),
    );
    expect(onchange).toHaveBeenLastCalledWith('grid');
  });

  it('vertical orientation stacks and answers up/down', () => {
    const onchange = vi.fn();
    const el = mountUi(Segmented, {
      options,
      value: 'grid',
      label: 'Tool',
      onchange,
      orientation: 'vertical',
    });
    const group = el.querySelector('[role="radiogroup"]')!;
    expect(group.className).toContain('flex-col');
    group.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }),
    );
    expect(onchange).toHaveBeenLastCalledWith('list');
    onchange.mockClear();
    group.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }),
    );
    expect(onchange).not.toHaveBeenCalled();
  });

  it('an icon-only option is named by its title', () => {
    const el = mountUi(Segmented, {
      options: [{ value: 'a', icon: 'hand', title: 'Pan' }],
      value: 'a',
      label: 'Tool',
      onchange: vi.fn(),
    });
    expect(el.querySelector('[role="radio"]')!.getAttribute('aria-label')).toBe('Pan');
  });
});
