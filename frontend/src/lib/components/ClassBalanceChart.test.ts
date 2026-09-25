/**
 * Mount test for `/dashboard`'s class balance chart (visual audit
 * 2026-09-24, D2): zero classes drew a visible 2% bar each, every class
 * past 30 was silently dropped, and counts included frozen test crops.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ClassBalanceChart from './ClassBalanceChart.svelte';

let instance: unknown;
let target: HTMLDivElement;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(props: Record<string, unknown>): HTMLDivElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ClassBalanceChart, { target, props } as never);
  flushSync();
  return target;
}

const row = (id: number, name: string, validated: number, count = 100) => ({
  class_id: id,
  class_name: name,
  count,
  validated_count: validated,
  adequacy: 'warn',
});

describe('ClassBalanceChart (D2)', () => {
  it('shows trainable = validated minus the served holdout, with the test count alongside', () => {
    const el = render({
      rows: [row(52, 'porsche', 35), row(10, 'bmw', 12)],
      holdout: { total: 5, by_class: [{ key: 52, doc_count: 5 }] },
    });
    const items = [...el.querySelectorAll('[data-testid="class-balance-bars"] li')].map(
      (li) => (li.textContent ?? '').replace(/\s+/g, ' ').trim(),
    );
    expect(items[0]).toBe('porsche 30 trainable · 5 test');
    expect(items[1]).toBe('bmw 12 trainable');
  });

  it('collapses zero classes into one line instead of drawing a bar for each', () => {
    const el = render({
      rows: [row(1, 'a', 3), row(2, 'b', 0), row(3, 'c', 0), row(4, 'd', 0)],
      holdout: null,
    });
    expect(el.querySelectorAll('[data-testid="class-balance-bars"] li')).toHaveLength(1);
    expect(el.querySelector('[data-testid="class-balance-zero"]')?.textContent).toContain(
      '3 classes with 0 validated',
    );
  });

  it('m6: the legend renders the served thresholds', () => {
    const el = render({
      rows: [row(1, 'a', 3)],
      holdout: null,
      thresholds: { block_below: 20, warn_below: 500 },
    });
    expect((el.textContent ?? '').replace(/\s+/g, ' ')).toContain(
      'green ≥500 · orange 20–499 · red <20',
    );
  });

  it('counts the classes past the limit rather than dropping them silently', () => {
    const rows = Array.from({ length: 5 }, (_, i) => row(i + 1, `c${i}`, 10 + i));
    const el = render({ rows, holdout: null, limit: 3 });
    expect(el.querySelectorAll('[data-testid="class-balance-bars"] li')).toHaveLength(3);
    expect(el.querySelector('[data-testid="class-balance-more"]')?.textContent).toContain(
      '+2 more classes',
    );
  });
});
