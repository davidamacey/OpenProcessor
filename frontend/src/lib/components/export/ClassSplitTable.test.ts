import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ClassSplitTable from './ClassSplitTable.svelte';

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLElement | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  target?.remove();
  instance = undefined;
});

describe('ClassSplitTable', () => {
  it('lists each class with grouped counts and flags one missing from train or val', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ClassSplitTable, {
      target,
      props: {
        class: 'max-h-64',
        rows: [
          {
            class_id: 1,
            export_id: 0,
            class_name: 'car',
            train: 1200,
            val: 30,
            test: 10,
          },
          { class_id: 2, export_id: 1, class_name: 'bus', train: 5, val: 0, test: 1 },
        ],
      },
    });
    flushSync();
    const rows = [...target.querySelectorAll('tbody tr')];
    expect(rows).toHaveLength(2);
    expect(rows[0]!.textContent).toContain((1200).toLocaleString());
    expect(rows[0]!.className).not.toContain('bg-red-500/10');
    expect(rows[1]!.className).toContain('bg-red-500/10');
    expect(target.firstElementChild!.className).toContain('max-h-64');
  });
});
