import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ClassRegistryTable from './ClassRegistryTable.svelte';

const noop = async () => {};
let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement | null = null;

function render(allClasses: never[]): string {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ClassRegistryTable, {
    target,
    props: {
      allClasses,
      activeRows: [],
      groups: [],
      busy: false,
      testHeldOutByClass: new Map(),
      editingId: null,
      editName: '',
      startEdit: () => {},
      cancelEdit: () => {},
      commitRename: noop,
      changeGroup: noop,
      setHotkey: noop,
      deprecateClassAction: noop,
    },
  });
  flushSync();
  return target.textContent ?? '';
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

describe('ClassRegistryTable empty state', () => {
  it('a project with no classes says so instead of blaming a filter', () => {
    const text = render([]);
    expect(text).toContain('No classes yet');
    expect(text).not.toContain('match this filter');
  });
});
