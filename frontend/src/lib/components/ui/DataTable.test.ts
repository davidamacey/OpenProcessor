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
    expect(names(el)).toEqual(['cog', 'bolt', 'person']);
  });

  it('a header click sorts ascending, again descending, and marks aria-sort', () => {
    const el = mountUi(DataTableHarness);
    const th = (id: string) =>
      [...el.querySelectorAll('th')].find((h) => h.textContent?.includes(id))!;
    th('Class').querySelector('button')!.click();
    flushSync();
    expect(names(el)).toEqual(['bolt', 'cog', 'person']);
    expect(th('Class').getAttribute('aria-sort')).toBe('ascending');
    th('Class').querySelector('button')!.click();
    flushSync();
    expect(names(el)).toEqual(['person', 'cog', 'bolt']);
    expect(th('Class').getAttribute('aria-sort')).toBe('descending');
  });

  it('a numeric column sorts descending first and numerically', () => {
    const el = mountUi(DataTableHarness);
    [...el.querySelectorAll('th')]
      .find((h) => h.textContent?.includes('Items'))!
      .querySelector('button')!
      .click();
    flushSync();
    expect(names(el)).toEqual(['person', 'cog', 'bolt']);
  });

  it('a column without a value is not sortable', () => {
    const el = mountUi(DataTableHarness);
    const plain = [...el.querySelectorAll('th')].find((h) =>
      h.textContent?.includes('Note'),
    )!;
    expect(plain.querySelector('button')).toBeNull();
    expect(plain.hasAttribute('aria-sort')).toBe(false);
  });

  it('numeric columns align their figures: tabular-nums on the header and every cell', () => {
    const el = mountUi(DataTableHarness);
    const th = [...el.querySelectorAll('th')].find((h) =>
      h.textContent?.includes('Items'),
    )!;
    expect(th.className).toContain('tabular-nums');
    const cells = el.querySelectorAll('tbody tr td:nth-child(2)');
    expect(cells.length).toBe(3);
    for (const c of cells) expect(c.className).toContain('tabular-nums');
    expect(el.querySelector('tbody tr td:nth-child(1)')!.className).not.toContain(
      'tabular-nums',
    );
  });

  it('keeps the first column in view while a narrow screen scrolls sideways', () => {
    const el = mountUi(DataTableHarness);
    for (const c of el.querySelectorAll('thead th:first-child, tbody td:first-child')) {
      expect(c.className, c.textContent ?? '').toContain('max-md:sticky');
      expect(c.className).toContain('max-md:left-0');
    }
    expect(el.querySelector('tbody td:nth-child(2)')!.className).not.toContain('sticky');
  });
});
