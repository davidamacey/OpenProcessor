/**
 * Mount-based behavior test for the holdout-aware summary
 * (`/train`'s "Classes to train" header): `validated_count` includes
 * test-holdout crops training never sees, so the summary shows the
 * served `GET {API_PREFIX}/test_holdout/stats` figure *alongside* the
 * validated count — never subtracted client-side — and omits the
 * clause entirely when the stats haven't loaded.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import type { RegistryClass, TestHoldoutStats } from '$lib/types';
import ClassSubsetPicker from './ClassSubsetPicker.svelte';

let target: HTMLDivElement;
let instance: unknown;

function makeClass(id: number, name: string, validated: number): RegistryClass {
  return {
    id,
    name,
    group: null,
    count: validated + 10,
    validated_count: validated,
    cluster_size: 0,
    added_at: '2026-01-01T00:00:00Z',
  };
}

function renderPicker(props: {
  classes: RegistryClass[];
  selected: number[] | null;
  holdout?: TestHoldoutStats | null;
}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ClassSubsetPicker, {
    target,
    props: {
      classes: props.classes,
      selected: props.selected,
      setSelected: () => {},
      singleCls: false,
      setSingleCls: () => {},
      presets: [],
      holdout: props.holdout ?? null,
    },
  });
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
});

describe('ClassSubsetPicker — holdout-aware summary', () => {
  const classes = [
    makeClass(1, 'miata', 35),
    makeClass(2, 'vw', 35),
    makeClass(3, 'mustang', 34),
  ];

  it('shows only the validated count when holdout stats are unavailable (null)', () => {
    const el = renderPicker({ classes, selected: [1, 2, 3], holdout: null });
    expect(el.textContent).toContain('3 classes selected · 104 validated crops');
    expect(el.textContent).not.toContain('held out');
  });

  it('shows the served holdout total alongside — never subtracted from — validated crops', () => {
    const holdout: TestHoldoutStats = {
      total: 15,
      by_class: [
        { key: 1, doc_count: 5 },
        { key: 2, doc_count: 5 },
        { key: 3, doc_count: 5 },
      ],
    };
    const el = renderPicker({ classes, selected: [1, 2, 3], holdout });
    expect(el.textContent).toContain(
      '3 classes selected · 104 validated crops (15 held out for test)',
    );
  });

  it('sums holdout only for the selected classes, not every class in the registry', () => {
    const holdout: TestHoldoutStats = {
      total: 15,
      by_class: [
        { key: 1, doc_count: 5 },
        { key: 2, doc_count: 5 },
        { key: 3, doc_count: 5 },
      ],
    };
    const el = renderPicker({ classes, selected: [1], holdout });
    expect(el.textContent).toContain(
      '1 class selected · 35 validated crops (5 held out for test)',
    );
  });

  it('omits the holdout clause when the served total for the selection is 0', () => {
    const holdout: TestHoldoutStats = {
      total: 15,
      by_class: [{ key: 4, doc_count: 15 }], // none of these are selected
    };
    const el = renderPicker({ classes, selected: [1], holdout });
    expect(el.textContent).toContain('1 class selected · 35 validated crops');
    expect(el.textContent).not.toContain('held out');
  });

  it('applies the holdout clause in "all classes" mode too', () => {
    const holdout: TestHoldoutStats = {
      total: 15,
      by_class: [
        { key: 1, doc_count: 5 },
        { key: 2, doc_count: 5 },
        { key: 3, doc_count: 5 },
      ],
    };
    const el = renderPicker({ classes, selected: null, holdout });
    expect(el.textContent).toContain(
      'All 3 classes · 104 validated crops (15 held out for test)',
    );
  });
});
