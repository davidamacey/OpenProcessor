import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import SortSelect from './SortSelect.svelte';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;

const OPTIONS = [
  { id: 'size_desc', label: 'Largest first' },
  { id: 'updated_desc', label: 'Recently updated' },
];

function render(props: Record<string, unknown>): HTMLElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SortSelect, { target, props } as never);
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

describe('SortSelect', () => {
  it('lists the served options in the served order and marks the applied one', () => {
    const el = render({ options: OPTIONS, applied: 'updated_desc', onchange: vi.fn() });
    const select = el.querySelector('select')!;
    expect([...select.options].map((o) => o.textContent)).toEqual([
      'Largest first',
      'Recently updated',
    ]);
    expect(select.value).toBe('updated_desc');
  });

  it('reports the chosen id', () => {
    const onchange = vi.fn();
    const el = render({ options: OPTIONS, applied: 'size_desc', onchange });
    const select = el.querySelector('select')!;
    select.value = 'updated_desc';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onchange).toHaveBeenCalledWith('updated_desc');
  });

  it('shows the served reason when the server applied another sort', () => {
    const el = render({
      options: OPTIONS,
      applied: 'size_desc',
      fallbackReason: 'Date order needs capture dates.',
      onchange: vi.fn(),
    });
    expect(el.querySelector('[data-testid="sort-fallback"]')?.textContent).toBe(
      'Date order needs capture dates.',
    );
  });
});
