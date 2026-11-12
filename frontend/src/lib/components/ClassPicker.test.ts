import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

const api = vi.hoisted(() => ({
  addClass: vi.fn(),
  invalidClassNameDetail: vi.fn(),
}));
vi.mock('$lib/api_classes', () => api);

import { classesStore } from '$stores/classes.svelte';
import { ApiError } from '$lib/api_errors';
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
  api.addClass.mockReset();
  api.invalidClassNameDetail.mockReset().mockReturnValue(null);
  onpick.mockReset();
  onclose.mockReset();
  classesStore.classes = [
    cls(1, 'widget'),
    cls(2, 'gadget'),
    cls(3, 'old_truck', { deprecated: true }),
    cls(4, 'region_x', { kind: 'region' }),
  ];
  vi.spyOn(classesStore, 'clearAndRefetch').mockImplementation(async () => {
    classesStore.classes = [...classesStore.classes, cls(9, 'traffic_light')];
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
    type('traffic light');
    expect(
      document.querySelector('[data-testid="class-picker-create"]')?.textContent,
    ).toContain('Create "traffic light"');
  });

  it('offers no create row for a name that exists', () => {
    render();
    type('widget');
    expect(document.querySelector('[data-testid="class-picker-create"]')).toBeNull();
  });

  it('turns a refusal into the served suggestion', async () => {
    api.addClass.mockRejectedValue(new ApiError(422, '/curation/classes', {}));
    api.invalidClassNameDetail.mockReturnValue({
      message: 'Use lower-case letters, digits and underscores only.',
      suggested_name: 'traffic_light',
    });
    render();
    const input = type('Traffic light');
    key(input, 'Enter');
    await vi.waitFor(() => {
      expect(
        document.querySelector('[data-testid="class-picker-create"]')?.textContent,
      ).toContain('Create "traffic_light"');
    });
    expect(document.querySelector('[role="alert"]')?.textContent).toContain('lower-case');
    expect(onpick).not.toHaveBeenCalled();
  });

  it('assigns the created class in the same gesture', async () => {
    render();
    const input = type('traffic_light');
    key(input, 'Enter');
    await vi.waitFor(() => expect(onpick).toHaveBeenCalledTimes(1));
    expect(api.addClass).toHaveBeenCalledWith({ name: 'traffic_light' });
    expect(onpick.mock.calls[0]![0]).toMatchObject({ id: 9, name: 'traffic_light' });
    expect(onclose).toHaveBeenCalled();
  });

  it('picks the highlighted class with Enter', () => {
    render();
    const input = type('bu');
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
