import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { ObjectRow } from '$lib/editor/editorController.svelte';
import ObjectList from './ObjectList.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const row = (over: Partial<ObjectRow> = {}): ObjectRow => ({
  cropId: 'c1',
  label: 'car',
  tone: 'validated',
  className: 'car',
  lockLabel: null,
  locked: false,
  deleted: false,
  regionCount: 0,
  ...over,
});

function render(rows: ObjectRow[], selectedCropId: string | null = null) {
  const calls = { onselect: vi.fn(), onclass: vi.fn(), ondelete: vi.fn() };
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ObjectList, { target, props: { rows, selectedCropId, ...calls } });
  flushSync();
  return calls;
}

describe('ObjectList', () => {
  it('lists the served labels in the served order', () => {
    render([row({ cropId: 'a', label: 'car' }), row({ cropId: 'b', label: 'bus' })]);
    expect(
      [...target.querySelectorAll('li[data-crop-id]')].map((l) => l.textContent?.trim()),
    ).toEqual(['car', 'bus']);
  });

  it('says so when there are no objects', () => {
    render([]);
    expect(target.querySelector('[data-testid="object-list-empty"]')).not.toBeNull();
  });

  it('a click selects the row', () => {
    const c = render([row({ cropId: 'a' }), row({ cropId: 'b', label: 'bus' })]);
    (target.querySelector('li[data-crop-id="b"] button') as HTMLElement).click();
    expect(c.onselect).toHaveBeenCalledWith('b');
  });

  it('shows the lock glyph with the served words only on a locked row', () => {
    render([
      row({ cropId: 'a', locked: true, lockLabel: 'Placed by a person' }),
      row({ cropId: 'b' }),
    ]);
    const locks = target.querySelectorAll('[data-testid="object-locked"]');
    expect(locks).toHaveLength(1);
    expect(locks[0].getAttribute('title')).toBe('Placed by a person');
  });

  it('marks a deleted object and offers no actions on it', () => {
    render([row({ deleted: true })], 'c1');
    expect(target.querySelector('[data-testid="object-deleted"]')).not.toBeNull();
    expect(target.textContent).not.toContain('Change class');
  });

  it('the selected row offers change class and delete', () => {
    const c = render([row()], 'c1');
    const buttons = [...target.querySelectorAll('li button')] as HTMLElement[];
    buttons.find((b) => b.textContent?.includes('Change class'))!.click();
    buttons.find((b) => b.textContent?.includes('Delete'))!.click();
    expect(c.onclass).toHaveBeenCalledWith('c1');
    expect(c.ondelete).toHaveBeenCalledWith('c1');
  });

  it('registers no keydown listener on window', () => {
    const add = vi.spyOn(window, 'addEventListener');
    render([row()]);
    expect(add.mock.calls.filter(([t]) => t === 'keydown')).toEqual([]);
    add.mockRestore();
  });
});
