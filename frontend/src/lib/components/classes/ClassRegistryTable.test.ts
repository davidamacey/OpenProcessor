import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ClassRegistryTable from './ClassRegistryTable.svelte';

const noop = async () => {};
let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement | null = null;

function render(allClasses: never[], activeRows: never[] = []): string {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ClassRegistryTable, {
    target,
    props: {
      allClasses,
      activeRows,
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

describe('ClassRegistryTable row controls are named for assistive tech', () => {
  it('names the group select, the hotkey input and the actions column', () => {
    const row = {
      id: 3,
      name: 'widget',
      group: 'tools',
      count: 1,
      validated_count: 0,
      cluster_size: 1,
      hotkey_letter: null,
      added_at: '2026-10-01T00:00:00Z',
    } as never;
    render([row], [row]);
    expect(target!.querySelector('select')?.getAttribute('aria-label')).toBe(
      'Group for widget',
    );
    expect(target!.querySelector('input')?.getAttribute('aria-label')).toBe(
      'Hotkey for widget',
    );
    const headers = [...target!.querySelectorAll('th')];
    expect(headers.every((th) => (th.textContent ?? '').trim() !== '')).toBe(true);
  });
});
