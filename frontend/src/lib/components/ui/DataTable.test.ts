import { flushSync } from 'svelte';
import { describe, expect, it } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import DataTableHarness from './DataTableHarness.test.svelte';

const names = (el: HTMLElement) =>
  [...el.querySelectorAll('tbody tr td:first-child')].map((c) => c.textContent?.trim());

describe('DataTable', () => {
  it('has a sticky header and renders one row per item through the cell snippet', () => {
    const el = mountUi(DataTableHarness);
    expect(el.querySelector('thead')!.className).toContain('sticky');
    expect(names(el)).toEqual(['car', 'bus', 'person']);
  });

  it('a header click sorts ascending, again descending, and marks aria-sort', () => {
    const el = mountUi(DataTableHarness);
    const th = (id: string) =>
      [...el.querySelectorAll('th')].find((h) => h.textContent?.includes(id))!;
    th('Class').querySelector('button')!.click();
    flushSync();
    expect(names(el)).toEqual(['bus', 'car', 'person']);
    expect(th('Class').getAttribute('aria-sort')).toBe('ascending');
    th('Class').querySelector('button')!.click();
    flushSync();
    expect(names(el)).toEqual(['person', 'car', 'bus']);
    expect(th('Class').getAttribute('aria-sort')).toBe('descending');
  });

  it('a numeric column sorts descending first and numerically', () => {
    const el = mountUi(DataTableHarness);
    [...el.querySelectorAll('th')]
      .find((h) => h.textContent?.includes('Items'))!
      .querySelector('button')!
      .click();
    flushSync();
    expect(names(el)).toEqual(['person', 'car', 'bus']);
  });

  it('a column without a value is not sortable', () => {
    const el = mountUi(DataTableHarness);
    const plain = [...el.querySelectorAll('th')].find((h) =>
      h.textContent?.includes('Note'),
    )!;
    expect(plain.querySelector('button')).toBeNull();
    expect(plain.hasAttribute('aria-sort')).toBe(false);
  });
});
