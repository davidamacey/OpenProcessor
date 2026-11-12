import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ColumnPicker from './ColumnPicker.svelte';
import type { TableColumn } from '$lib/types_table';

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLElement | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  target?.remove();
  instance = undefined;
});

const VOCAB: TableColumn[] = ['a', 'b', 'c'].map((id) => ({
  id,
  label: id.toUpperCase(),
  sortable: true,
  align: 'start',
  kind: 'text',
}));

function render(chosen: string[]) {
  const onchange = vi.fn();
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ColumnPicker, {
    target,
    props: { vocabulary: VOCAB, chosen, onchange },
  });
  flushSync();
  return { onchange, el: target };
}

describe('ColumnPicker', () => {
  it('sends the chosen ids in the served vocabulary order', () => {
    const { el, onchange } = render(['c']);
    (el.querySelectorAll('input')[0] as HTMLInputElement).click();
    expect(onchange).toHaveBeenCalledWith(['a', 'c']);
  });

  it('never lets the last column go', () => {
    const { el } = render(['b']);
    expect((el.querySelectorAll('input')[1] as HTMLInputElement).disabled).toBe(true);
  });
});
