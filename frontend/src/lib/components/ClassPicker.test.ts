import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

const api = vi.hoisted(() => ({
  addClass: vi.fn(),
}));
vi.mock('$lib/api_classes', () => api);

import { classesStore } from '$stores/classes.svelte';
import type { RegistryClass } from '$lib/types_classes';
import ClassPicker from './ClassPicker.svelte';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;
const onpick = vi.fn();
const onclose = vi.fn();

function cls(
  id: number,
  name: string,
  extra: Partial<RegistryClass> = {},
): RegistryClass {
  return {
    id,
    name,
    group: 'g',
    count: id * 10,
    validated_count: id,
    added_at: '2026-01-01',
    adequacy: 'ok',
    kind: 'item',
    trainable: 0,
    trainable_gap: 0,
    ...extra,
  };
}

function render(props: Record<string, unknown> = {}): HTMLElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ClassPicker, {
    target,
    props: { open: true, mode: 'assign', onpick, onclose, ...props },
  } as never);
  flushSync();
  return target.ownerDocument.body;
}

function type(text: string): HTMLInputElement {
  const input = document.querySelector('input[role="combobox"]') as HTMLInputElement;
  input.value = text;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
  return input;
}

function key(input: HTMLElement, k: string): void {
  input.dispatchEvent(
    new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }),
  );
  flushSync();
}

beforeEach(() => {
  api.addClass
    .mockReset()
    .mockResolvedValue({ class_id: 9, class_name: 'school_bus', group: 'g' });
  onpick.mockReset();
  onclose.mockReset();
  classesStore.classes = [
    cls(1, 'widget'),
    cls(2, 'gadget'),
    cls(3, 'old_truck', { deprecated: true }),
    cls(4, 'region_x', { kind: 'region' }),
    cls(5, 'traffic_light'),
  ];
  vi.spyOn(classesStore, 'clearAndRefetch').mockImplementation(async () => {
    classesStore.classes = [...classesStore.classes, cls(9, 'school_bus')];
  });
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
  vi.restoreAllMocks();
});

describe('ClassPicker', () => {
  it('hides deprecated classes', () => {
    render();
    const names = [...document.querySelectorAll('[role="option"]')].map(
      (o) => o.textContent,
    );
    expect(names.some((n) => n?.includes('old_truck'))).toBe(false);
    expect(names.some((n) => n?.includes('widget'))).toBe(true);
  });

  it('offers create when nothing matches', () => {
    render();
    type('school bus');
    expect(
      document.querySelector('[data-testid="class-picker-create"]')?.textContent,
    ).toContain('Create new class "school bus"');
  });

  it('offers no create row for a name that exists', () => {
    render();
    type('widget');
    expect(document.querySelector('[data-testid="class-picker-create"]')).toBeNull();
  });

  it('picks the class under the slug the server answered for a typed name', async () => {
    render();
    const input = type('School Bus');
    key(input, 'Enter');
    await vi.waitFor(() => expect(onpick).toHaveBeenCalledTimes(1));
    expect(api.addClass).toHaveBeenCalledWith({ name: 'School Bus' });
    expect(onpick.mock.calls[0]![0].name).toBe('school_bus');
  });

  it('assigns the created class in the same gesture', async () => {
    render();
    const input = type('school_bus');
    key(input, 'Enter');
    await vi.waitFor(() => expect(onpick).toHaveBeenCalledTimes(1));
    expect(api.addClass).toHaveBeenCalledWith({ name: 'school_bus' });
    expect(onpick.mock.calls[0]![0]).toMatchObject({ id: 9, name: 'school_bus' });
    expect(onclose).toHaveBeenCalled();
  });

  it.each(['Traffic light', 'traffic-light', 'TRAFFIC  LIGHT', ' traffic_light '])(
    'a natural spelling of an existing class (%s) selects it and offers no create',
    (typed) => {
      render();
      const input = type(typed);
      expect(document.querySelector('[data-testid="class-picker-create"]')).toBeNull();
      expect(document.querySelector('[role="option"]')?.textContent).toContain(
        'traffic_light',
      );
      key(input, 'Enter');
      expect(onpick.mock.calls[0]![0]).toMatchObject({ id: 5, name: 'traffic_light' });
      expect(api.addClass).not.toHaveBeenCalled();
    },
  );

  it('picks the highlighted class with Enter', () => {
    render();
    const input = type('ga');
    key(input, 'Enter');
    expect(onpick.mock.calls[0]![0]).toMatchObject({ name: 'gadget' });
  });

  it('create mode lists no classes', () => {
    render({ mode: 'create' });
    expect(document.querySelectorAll('[role="option"]').length).toBe(0);
    type('van');
    expect(document.querySelector('[data-testid="class-picker-create"]')).not.toBeNull();
  });

  it('adds no window keydown listener', () => {
    const add = vi.spyOn(window, 'addEventListener');
    render();
    expect(add.mock.calls.filter(([t]) => t === 'keydown')).toHaveLength(0);
  });
});
