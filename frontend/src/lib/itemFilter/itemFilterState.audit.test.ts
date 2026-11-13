import { describe, expect, it } from 'vitest';
import { ItemFilterState } from './itemFilterState.svelte';

describe('ItemFilterState audit_outcome', () => {
  it('reads the link from an audit report and sends it on every query and body', () => {
    const s = new ItemFilterState();
    s.fromUrl(
      new URLSearchParams(
        'audit_outcome=both_wrong&audit_outcome=agree&audit_outcome=bogus',
      ),
    );
    expect(s.auditOutcomes).toEqual(['both_wrong', 'agree']);
    expect(s.toQuery().audit_outcome).toEqual(['both_wrong', 'agree']);
    expect(s.toBody().audit_outcome).toEqual(['both_wrong', 'agree']);
    const url = new URLSearchParams();
    s.toUrl(url);
    expect(url.getAll('audit_outcome')).toEqual(['both_wrong', 'agree']);
    expect(s.isEmpty).toBe(false);
  });

  it('shows one removable chip per outcome and clears with the rest', () => {
    const s = new ItemFilterState();
    s.auditOutcomes = ['vlm_wrong'];
    const chips = s.chips();
    expect(chips.map((c) => c.label)).toEqual(['Audit: VLM wrong']);
    chips[0].clear();
    expect(s.auditOutcomes).toEqual([]);
    s.auditOutcomes = ['agree'];
    s.clear();
    expect(s.auditOutcomes).toEqual([]);
  });
});
