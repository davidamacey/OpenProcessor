import { describe, expect, it, vi } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import ClusterDetails from './ClusterDetails.svelte';

function ws(card: Record<string, unknown> | null) {
  return { scopeCard: card, startLabelCluster: vi.fn() };
}
const card = (over: Record<string, unknown> = {}) => ({
  uid: 'u1',
  display_no: 3,
  size: 12,
  dominant_class_name: null,
  display_name: 'bus',
  display_name_kind: 'detector',
  ...over,
});

describe('ClusterDetails', () => {
  it("names the cluster with the served name and says when it is the detector's", () => {
    const el = mountUi(ClusterDetails, { ws: ws(card()), id: 'u1' });
    const name = el.querySelector('[data-testid="cluster-details-name"]')!.textContent!;
    expect(name).toContain('bus');
    expect(name).toContain('from the detector');
  });

  it('says "not named yet" when the server served no name', () => {
    const el = mountUi(ClusterDetails, {
      ws: ws(card({ display_name: '', display_name_kind: 'none' })),
      id: 'u1',
    });
    expect(el.textContent).toContain('not named yet');
  });

  it('keeps a secondary copy of the whole-cluster label, through the same label flow', () => {
    const w = ws(card());
    const el = mountUi(ClusterDetails, { ws: w, id: 'u1' });
    const btn = el.querySelector(
      '[data-testid="label-cluster-details"]',
    ) as HTMLButtonElement;
    expect(btn.textContent).toContain('Label all 12 as');
    expect(btn.className).not.toContain('btn-primary');
    btn.click();
    expect(w.startLabelCluster).toHaveBeenCalledOnce();
  });

  it('has no label action before a cluster is open', () => {
    const el = mountUi(ClusterDetails, { ws: ws(null), id: null });
    expect(el.querySelector('[data-testid="label-cluster-details"]')).toBeNull();
  });
});
