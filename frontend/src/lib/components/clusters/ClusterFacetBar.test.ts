/**
 * The facet chips render the served counts verbatim (zeros included) and a
 * click goes to the controller with the served value.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ClusterFacetBar from './ClusterFacetBar.svelte';
import type { ClusterFacets } from '$lib/types_clusterList';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;

const FACETS: ClusterFacets = {
  cluster_id: 7,
  total: 30,
  detector_classes: [
    { value: 'widget_x', count: 20 },
    { value: 'widget_y', count: 0 },
  ],
  class_sources: [],
  validation: { validated: 4, unvalidated: 26 },
  review_tabs: { outliers: 3, detector_disagreements: 0 },
  detector_confidence: [
    { min: 0, max: 0.1, count: 0 },
    { min: 0.9, max: null, count: 12 },
  ],
  detector_confidence_missing: 2,
};

function fakeView(over: Record<string, unknown> = {}) {
  return {
    facets: FACETS,
    facetsError: null,
    facetDetectorClass: null,
    facetBand: null,
    facetReviewTab: null,
    facetValidated: null,
    detectorSort: 'default',
    toggleDetectorClass: vi.fn(),
    toggleBand: vi.fn(),
    toggleReviewTab: vi.fn(),
    toggleValidated: vi.fn(),
    ...over,
  };
}

function render(view = fakeView()) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ClusterFacetBar, { target, props: { view: view as never } });
  flushSync();
  return { view, root: target };
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

const texts = (root: HTMLElement, id: string) =>
  [...root.querySelectorAll(`[data-testid="${id}"]`)].map((e) =>
    (e.textContent ?? '').replace(/\s+/g, ' ').trim(),
  );

describe('ClusterFacetBar', () => {
  it('renders every served count verbatim, a served zero included', () => {
    const { root } = render();
    expect(texts(root, 'facet-detector-class')).toEqual(['widget_x 20', 'widget_y 0']);
    expect(texts(root, 'facet-validated')).toEqual(['validated 4']);
    expect(texts(root, 'facet-unvalidated')).toEqual(['not validated 26']);
    expect(texts(root, 'facet-review-tab').map((t) => t.split(' ').at(-1))).toEqual([
      '3',
      '0',
    ]);
    expect(texts(root, 'facet-confidence-bin')).toEqual(['0.0–0.1 0', '0.9+ 12']);
    expect(texts(root, 'facet-confidence-missing')).toEqual(['none: 2']);
  });

  it('sends the served value of the clicked chip', () => {
    const { root, view } = render();
    const click = (id: string, i = 0) =>
      (root.querySelectorAll(`[data-testid="${id}"]`)[i] as HTMLButtonElement).click();
    click('facet-detector-class', 1);
    expect(view.toggleDetectorClass).toHaveBeenCalledWith('widget_y');
    click('facet-validated');
    expect(view.toggleValidated).toHaveBeenCalledWith(true);
    click('facet-unvalidated');
    expect(view.toggleValidated).toHaveBeenCalledWith(false);
    click('facet-review-tab');
    expect(view.toggleReviewTab).toHaveBeenCalledWith('outliers');
    click('facet-confidence-bin', 1);
    expect(view.toggleBand).toHaveBeenCalledWith({ min: 0.9, max: null, count: 12 });
  });

  it('marks the selected chips pressed', () => {
    const { root } = render(
      fakeView({ facetDetectorClass: 'widget_x', facetValidated: false }),
    );
    const pressed = [...root.querySelectorAll('[aria-pressed="true"]')].map((e) =>
      e.getAttribute('data-testid'),
    );
    expect(pressed.sort()).toEqual(['facet-detector-class', 'facet-unvalidated']);
  });

  it('shows the error instead of counts when the facets failed', () => {
    const { root } = render(fakeView({ facets: null, facetsError: 'API 500' }));
    expect(
      root.querySelector('[data-testid="cluster-facets-error"]')?.textContent,
    ).toContain('API 500');
    expect(root.querySelector('[data-testid="cluster-facets"]')).toBeNull();
  });

  it('draws nothing before the facets arrive', () => {
    const { root } = render(fakeView({ facets: null }));
    expect(root.querySelector('[data-testid="cluster-facets"]')).toBeNull();
  });
});
