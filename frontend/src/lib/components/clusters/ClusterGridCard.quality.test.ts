/**
 * The card shows the served quality fields: the detector's top class with its
 * share and mean confidence, the validation state, the review count and the
 * flags (served label as the chip, served description as the tooltip).
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
    id: 7,
    size: 40,
    purity_tier: 'mixed',
    purity: 0.5,
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
  it('shows the detector top class with its share and mean confidence', () => {
    const el = q(render(), 'cluster-detector')!;
    expect(el.textContent).toContain('widget_x');
    expect(el.textContent).toContain('62%');
    expect(el.textContent).toContain('0.81');
  });

  it('shows the validation state and the review count', () => {
    const root = render();
    expect(q(root, 'cluster-validation')!.textContent).toMatch(/partly validated/);
    expect(q(root, 'cluster-needs-review')!.textContent).toContain('7');
    expect(q(root, 'cluster-needs-review')!.getAttribute('title')).toContain(
      'outliers: 3',
    );
  });

  it('renders each flag with its served label and description', () => {
    const flags = [...render().querySelectorAll('[data-testid="cluster-flag"]')];
    expect(flags.map((f) => f.textContent?.trim())).toEqual([
      'Mixed detector classes',
      'Unvalidated',
    ]);
    expect(flags[0]!.getAttribute('title')).toBe('The detector gave different classes.');
  });

  it('omits the detector line, the review chip and flags when none are served', () => {
    const root = render({
      detector_top_class: null,
      needs_review_count: 0,
      flags: [],
    });
    expect(q(root, 'cluster-detector')).toBeNull();
    expect(q(root, 'cluster-needs-review')).toBeNull();
    expect(root.querySelector('[data-testid="cluster-flag"]')).toBeNull();
  });

  it('nests no interactive control inside the card button', () => {
    const root = render();
    expect(
      root.querySelectorAll('button button, button a, button input, button select')
        .length,
    ).toBe(0);
  });
});
