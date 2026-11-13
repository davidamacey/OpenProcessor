import { describe, expect, it } from 'vitest';
import { mountUi, snip } from '$lib/test/mountUi';
import Kbd from './Kbd.svelte';

describe('Kbd', () => {
  it('spells a letter and a combo with capitals', () => {
    expect(mountUi(Kbd, { combo: 'l' }).querySelector('kbd')!.textContent).toBe('L');
    expect(mountUi(Kbd, { combo: 'shift+space' }).querySelector('kbd')!.textContent).toBe(
      'Shift+Space',
    );
    expect(mountUi(Kbd, { combo: 'escape' }).querySelector('kbd')!.textContent).toBe(
      'Esc',
    );
  });

  it('shows children as they are when no combo is given', () => {
    expect(mountUi(Kbd, { children: snip('/') }).querySelector('kbd')!.textContent).toBe(
      '/',
    );
  });

  it('adds the caller class', () => {
    expect(
      mountUi(Kbd, { combo: '1', class: 'text-caption' }).querySelector('kbd')!.className,
    ).toBe('text-caption');
  });
});
