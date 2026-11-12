import { describe, expect, it, vi } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import ClusterDetails from './ClusterDetails.svelte';

function ws(card: Record<string, unknown> | null) {
  return { scopeCard: card, confirmSelectAllMatching: vi.fn(), pickerOpen: false };
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
    expect(name).toContain("detector's most common class");
  });

  it('says "not named yet" when the server served no name', () => {
    const el = mountUi(ClusterDetails, {
      ws: ws(card({ display_name: '', display_name_kind: 'none' })),
      id: 'u1',
    });
    expect(el.textContent).toContain('not named yet');
  });

  it('labels the whole cluster in one action: select everything it matches, open the picker', () => {
    const w = ws(card());
    const el = mountUi(ClusterDetails, { ws: w, id: 'u1' });
    const btn = el.querySelector('[data-testid="label-cluster"]') as HTMLButtonElement;
    expect(btn.textContent).toContain('Label all 12');
    btn.click();
    expect(w.confirmSelectAllMatching).toHaveBeenCalledOnce();
    expect(w.pickerOpen).toBe(true);
  });

  it('has no label action before a cluster is open', () => {
    const el = mountUi(ClusterDetails, { ws: ws(null), id: null });
    expect(el.querySelector('[data-testid="label-cluster"]')).toBeNull();
  });
});
