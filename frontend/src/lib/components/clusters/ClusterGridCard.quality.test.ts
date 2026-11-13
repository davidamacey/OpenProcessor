/**
 * The card shows the served quality fields sparingly: the labelled share and the detector's top
 * class on one line, at most one status chip (served flag label, served description as the
 * tooltip), and the rest in the hover text.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

import ClusterGridCard from './ClusterGridCard.svelte';
import type { Cluster } from '$lib/types';
import type { ClusterFlag } from '$lib/types_clusterList';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;

const VOCAB: ClusterFlag[] = [
  {
    id: 'mixed_detector_classes',
    label: 'Mixed detector classes',
    description: 'The detector gave different classes.',
  },
  { id: 'unvalidated', label: 'Unvalidated', description: 'No member is validated.' },
];

function render(extra: Partial<Cluster> = {}): HTMLElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  const c = {
    uid: '0b9f9a38-0000-4000-8000-000000000007',
    size: 40,
    label_agreement: 0.75,
    dominant_pct: 0.75,
    dominant_class_name: 'class_a',
    representative_crop_ids: [],
    detector_top_class: 'widget_x',
    detector_top_share: 0.62,
    detector_conf_mean: 0.814,
    validation_state: 'partial',
    review_counts: { all: 7, outliers: 3 },
    needs_review_count: 7,
    flags: ['mixed_detector_classes', 'unvalidated'],
    ...extra,
  } as unknown as Cluster;
  instance = mount(ClusterGridCard, { target, props: { c, flagVocabulary: VOCAB } });
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

const q = (root: HTMLElement, id: string) =>
  root.querySelector(`[data-testid="${id}"]`) as HTMLElement | null;

describe('ClusterGridCard quality fields', () => {
  it('shows the labelled share and the detector top class on one meta line', () => {
    const root = render({
      labelled_share: 1,
      display_name: 'cup',
      display_name_kind: 'class',
    });
    expect(q(root, 'cluster-meta')!.textContent).toBe(
      '100% labelled \u00b7 detector: widget_x 62%',
    );
  });

  it('shows one status chip, the review count first, in a warn tone', () => {
    const root = render();
    const chips = root.querySelectorAll('[data-testid="cluster-status"]');
    expect(chips).toHaveLength(1);
    expect(chips[0]!.textContent).toContain('7 to review');
    expect(chips[0]!.className).toContain('chip-warn');
    expect(chips[0]!.getAttribute('title')).toContain('outliers: 3');
  });

  it('falls to the served mixed-classes flag, then to unvalidated in an info tone', () => {
    const mixed = q(render({ needs_review_count: 0 }), 'cluster-status')!;
    expect(mixed.textContent?.trim()).toBe('Mixed detector classes');
    expect(mixed.getAttribute('title')).toBe('The detector gave different classes.');
    if (instance) unmount(instance);
    target.remove();
    const un = q(
      render({ needs_review_count: 0, flags: [], validation_state: 'none' }),
      'cluster-status',
    )!;
    expect(un.textContent?.trim()).toBe('Unvalidated');
    expect(un.className).toContain('chip-info');
  });

  it('keeps agreement and detector confidence in the hover text, not on the card', () => {
    const root = render({ detector_conf_mean: 0.814, validated_count: 4 });
    const btn = root.querySelector('button')!;
    expect(btn.getAttribute('title')).toContain('agreement 75%');
    expect(btn.getAttribute('title')).toContain('detector confidence 0.81');
    expect(btn.getAttribute('title')).toContain('4 of 40 validated');
    expect(root.textContent).not.toContain('agreement');
    expect(q(root, 'cluster-agreement')).toBeNull();
  });

  it('shows no status when nothing needs attention', () => {
    const root = render({
      needs_review_count: 0,
      flags: [],
      validation_state: 'complete',
    });
    expect(q(root, 'cluster-status')).toBeNull();
  });

  it('nests no interactive control inside the card button', () => {
    const root = render();
    expect(
      root.querySelectorAll('button button, button a, button input, button select')
        .length,
    ).toBe(0);
  });
});
