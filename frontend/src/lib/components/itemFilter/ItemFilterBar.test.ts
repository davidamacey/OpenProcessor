/**
 * The shared filter bar: a control the route does not honour is absent, a
 * served spec replaces the local label / bounds / options, list values and
 * names have removable chips, the open-vocabulary pair appears only in the
 * Matching view, and a served refusal renders under the bar.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ItemFilterBar from './ItemFilterBar.svelte';
import ServedFilterField from './ServedFilterField.svelte';
import { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';
import { classesStore } from '$stores/classes.svelte';
import type { ReviewFilterSpec } from '$lib/api';
import type { RegistryClass } from '$lib/types';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;
const prevClasses = classesStore.classes;

function cls(id: number, name: string, deprecated = false): RegistryClass {
  return {
    id,
    name,
    deprecated,
    group: null,
    count: 0,
    validated_count: 0,
  } as RegistryClass;
}

beforeEach(() => {
  classesStore.classes = [cls(1, 'widget'), cls(2, 'gadget'), cls(3, 'old', true)];
  target = document.createElement('div');
  document.body.appendChild(target);
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
  classesStore.classes = prevClasses;
});

const q = (sel: string) => target.querySelector<HTMLElement>(sel);
const has = (param: string) => q(`[data-testid="served-filter-${param}"]`) != null;

describe('ItemFilterBar', () => {
  it('draws every shared control by default, without the open-vocabulary pair', () => {
    instance = mount(ItemFilterBar, { target, props: { state: new ItemFilterState() } });
    flushSync();
    for (const p of [
      'class_name',
      'exclude_class_name',
      'conf_min',
      'conf_max',
      'min_area',
      'max_area',
      'max_rank',
      'origin',
      'embedding_state',
      'review_status',
    ]) {
      expect(has(p), p).toBe(true);
    }
    expect(has('open_vocab_set')).toBe(false);
    expect(has('source_prompt')).toBe(false);
  });

  it('only the class control is inline; the rest sit in a More filters disclosure that opens when one is active', () => {
    const state = new ItemFilterState();
    instance = mount(ItemFilterBar, { target, props: { state } });
    flushSync();
    const more = q('[data-testid="item-filter-more"]') as HTMLDetailsElement;
    expect(more.open).toBe(false);
    expect(more.textContent).toContain('More filters');
    expect(more.querySelector('[data-testid="served-filter-origin"]')).not.toBeNull();
    expect(more.querySelector('[data-testid="served-filter-class_name"]')).toBeNull();
    expect(q('[data-testid="served-filter-class_name"]')!.closest('details')).toBeNull();
    state.origin = ['sam3'];
    flushSync();
    expect(more.open).toBe(true);
    expect(more.textContent).toContain('More filters (1)');
  });

  it('showOpenVocab adds the set and prompt text controls', () => {
    instance = mount(ItemFilterBar, {
      target,
      props: { state: new ItemFilterState(), showOpenVocab: true },
    });
    flushSync();
    expect(has('open_vocab_set')).toBe(true);
    expect(has('source_prompt')).toBe(true);
  });

  it('a param the route does not honour is absent, not disabled', () => {
    instance = mount(ItemFilterBar, {
      target,
      props: {
        state: new ItemFilterState(),
        visible: (p: string) => p !== 'origin' && p !== 'conf_min',
      },
    });
    flushSync();
    expect(has('origin')).toBe(false);
    expect(has('conf_min')).toBe(false);
    expect(has('conf_max')).toBe(true);
    expect(target.querySelectorAll('[disabled]')).toHaveLength(0);
  });

  it('a served spec replaces the local label, bounds and help text', () => {
    const served: ReviewFilterSpec[] = [
      {
        param: 'conf_min',
        kind: 'number',
        label: 'Served min confidence',
        options: [],
        min: 0.1,
        max: 0.9,
        description: 'Served help',
      },
    ];
    instance = mount(ItemFilterBar, {
      target,
      props: { state: new ItemFilterState(), served },
    });
    flushSync();
    const field = q('[data-testid="served-filter-conf_min"]')!;
    expect(field.textContent).toContain('Served min confidence');
    expect(field.getAttribute('title')).toBe('Served help');
    const input = field.querySelector('input')!;
    expect(input.min).toBe('0.1');
    expect(input.max).toBe('0.9');
  });

  it('class names get a removable chip and an add-a-class select over live classes', () => {
    const state = new ItemFilterState();
    state.classNames = ['widget'];
    const onchange = vi.fn();
    instance = mount(ItemFilterBar, { target, props: { state, onchange } });
    flushSync();
    const chip = q('[data-testid="item-filter-chip"]')!;
    expect(chip.textContent).toContain('widget');
    const options = [
      ...q('[data-testid="served-filter-class_name"] select')!.querySelectorAll('option'),
    ].map((o) => o.value);
    // already chosen and deprecated classes are not offered again
    expect(options).toEqual(['', 'gadget']);
    chip.click();
    flushSync();
    expect(state.classNames).toEqual([]);
    expect(onchange).toHaveBeenCalled();
  });

  it('toggling an origin chip writes the state and Clear filters empties it', () => {
    const state = new ItemFilterState();
    instance = mount(ItemFilterBar, { target, props: { state } });
    flushSync();
    const buttons = [
      ...q('[data-testid="served-filter-origin"]')!.querySelectorAll('button'),
    ];
    buttons.find((b) => b.textContent?.includes('Sam3'))!.click();
    flushSync();
    expect(state.origin).toEqual(['sam3']);
    q('[data-testid="item-filter-clear"]')!.click();
    flushSync();
    expect(state.isEmpty).toBe(true);
    expect(q('[data-testid="item-filter-clear"]')).toBeNull();
  });

  it('a URL-seeded open-vocabulary set shows as a removable chip', () => {
    const state = new ItemFilterState();
    state.openVocabSet = 'tags';
    instance = mount(ItemFilterBar, { target, props: { state, showOpenVocab: true } });
    flushSync();
    const chip = q('[data-testid="item-filter-chip"]')!;
    expect(chip.textContent).toContain('tags');
    chip.click();
    flushSync();
    expect(state.openVocabSet).toBeNull();
  });

  it("shows the server's refusal under the bar", () => {
    instance = mount(ItemFilterBar, {
      target,
      props: { state: new ItemFilterState(), error: 'conf_min must be <= conf_max' },
    });
    flushSync();
    expect(q('[data-testid="item-filter-error"]')!.textContent).toContain(
      'conf_min must be <= conf_max',
    );
  });
});

describe('ServedFilterField kinds', () => {
  const base = { options: [], min: null, max: null, description: '' };
  function field(
    spec: Partial<ReviewFilterSpec> & Pick<ReviewFilterSpec, 'param' | 'kind'>,
    value: string | string[] = '',
  ) {
    const onchange = vi.fn();
    instance = mount(ServedFilterField, {
      target,
      props: { spec: { ...base, label: 'L', ...spec }, value, onchange },
    });
    flushSync();
    return onchange;
  }

  it('enum: a select of the served options, changes emit the value', () => {
    const onchange = field({
      param: 'p',
      kind: 'enum',
      options: [
        { value: 'a', label: 'Alpha' },
        { value: 'b', label: 'Beta' },
      ],
    });
    const sel = target.querySelector('select')!;
    expect([...sel.options].map((o) => o.textContent?.trim())).toEqual(['Alpha', 'Beta']);
    sel.value = 'b';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onchange).toHaveBeenCalledWith('p', 'b');
  });

  it('multi_enum: toggles add and remove', () => {
    const onchange = field(
      {
        param: 'p',
        kind: 'multi_enum',
        options: [
          { value: 'a', label: 'Alpha' },
          { value: 'b', label: 'Beta' },
        ],
      },
      ['a'],
    );
    const [a, b] = [...target.querySelectorAll('button')];
    expect(a!.getAttribute('aria-pressed')).toBe('true');
    b!.click();
    expect(onchange).toHaveBeenLastCalledWith('p', ['a', 'b']);
    a!.click();
    expect(onchange).toHaveBeenLastCalledWith('p', []);
  });

  it('bool: any / yes / no emits true / false / empty', () => {
    const onchange = field({ param: 'p', kind: 'bool' });
    const sel = target.querySelector('select')!;
    expect([...sel.options].map((o) => o.value)).toEqual(['', 'true', 'false']);
    sel.value = 'false';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onchange).toHaveBeenCalledWith('p', 'false');
  });

  it('integer / number: the served bounds, integer steps by one', () => {
    field({ param: 'p', kind: 'integer', min: 1, max: 9 });
    const input = target.querySelector('input')!;
    expect([input.min, input.max, input.step]).toEqual(['1', '9', '1']);
  });

  it('text: a text input', () => {
    const onchange = field({ param: 'p', kind: 'text' });
    const input = target.querySelector('input')!;
    expect(input.type).toBe('text');
    input.value = 'blue widget';
    input.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onchange).toHaveBeenCalledWith('p', 'blue widget');
  });

  it('class_names: adds the picked class to the list', () => {
    const onchange = field({ param: 'p', kind: 'class_names' }, ['widget']);
    const sel = target.querySelector('select')!;
    sel.value = 'gadget';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onchange).toHaveBeenCalledWith('p', ['widget', 'gadget']);
  });

  describe('ItemFilterBar chips', () => {
    it('draws a chip per entry even when a class name is listed twice', () => {
      const state = new ItemFilterState();
      state.classNames = ['widget', 'widget'];
      instance = mount(ItemFilterBar, { target, props: { state } });
      flushSync();
      expect(target.querySelectorAll('button.chip').length).toBeGreaterThanOrEqual(2);
      expect(target.textContent).toContain('widget');
    });
  });
});
