/**
 * V-4: default/exclude selection keyed only off the served
 * `trainable_gap` (and served `kind`), never a client threshold.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { RegistryClass } from '$lib/types';
import {
  classesLackingData,
  defaultTrainSelection,
  excludeLackingData,
} from './trainClassSelection';
import ClassSubsetPicker from '$lib/components/ClassSubsetPicker.svelte';

function cls(id: number, over: Partial<RegistryClass> = {}): RegistryClass {
  return {
    id,
    name: `widget_${id}`,
    group: null,
    count: 0,
    validated_count: 0,
    cluster_size: 0,
    added_at: '2026-01-01T00:00:00Z',
    adequacy: 'block',
    kind: 'item',
    trainable: 0,
    trainable_gap: 0,
    ...over,
  };
}

const registry: RegistryClass[] = [
  cls(0, { trainable: 150, trainable_gap: 0, kind: 'item' }),
  cls(1, { trainable: 90, trainable_gap: 0, kind: 'item' }),
  cls(2, { trainable: 0, trainable_gap: 20, kind: 'item' }),
  cls(3, { trainable: 0, trainable_gap: 20, kind: 'region' }),
  cls(4, { deprecated: true, trainable_gap: 20 }),
];

describe('trainClassSelection', () => {
  it('lacking = non-deprecated item classes with a served trainable_gap > 0', () => {
    expect(classesLackingData(registry).map((c) => c.id)).toEqual([2]);
  });

  it('defaults to the classes with enough data when any class is short', () => {
    expect(defaultTrainSelection(registry)).toEqual([0, 1]);
  });

  it('keeps "all" when no class is short, or the backend serves no trainable_gap', () => {
    expect(defaultTrainSelection([cls(0, { trainable_gap: 0 }), cls(1)])).toBeNull();
    expect(defaultTrainSelection([cls(0), cls(1)])).toBeNull();
  });

  it('exclude drops the lacking classes from an explicit or "all" selection', () => {
    expect(excludeLackingData([0, 2], registry)).toEqual([0]);
    expect(excludeLackingData(null, registry)).toEqual([0, 1, 3]);
  });
});

describe('ClassSubsetPicker "exclude classes without enough data"', () => {
  let target: HTMLDivElement;
  let instance: unknown;
  afterEach(() => {
    if (instance) unmount(instance);
    instance = undefined;
    target?.remove();
  });

  it('is offered while a selected class is short, and sets the served-trainable subset', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    const setSelected = vi.fn();
    instance = mount(ClassSubsetPicker, {
      target,
      props: {
        classes: registry,
        selected: [0, 1, 2],
        setSelected,
        singleCls: false,
        setSingleCls: () => {},
      },
    });
    flushSync();
    expect(
      target.querySelector('[data-testid="lacking-data-row"]')?.textContent,
    ).toContain('widget_2');
    (
      target.querySelector('[data-testid="exclude-lacking"]') as HTMLButtonElement
    ).click();
    expect(setSelected).toHaveBeenCalledWith([0, 1]);
  });

  it('is absent when no selected class is short', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ClassSubsetPicker, {
      target,
      props: {
        classes: registry,
        selected: [0, 1],
        setSelected: () => {},
        singleCls: false,
        setSingleCls: () => {},
      },
    });
    flushSync();
    expect(target.querySelector('[data-testid="lacking-data-row"]')).toBeNull();
  });
});
