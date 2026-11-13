import { describe, expect, it } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import { datasetStatsFixture } from '$lib/test/fixtures/datasetStats';
import LabeledBySourceCard from './LabeledBySourceCard.svelte';

describe('LabeledBySourceCard', () => {
  it('shows the served total, one row per served source and the ground-truth note', () => {
    const stats = datasetStatsFixture({ total_crops: 937, validated: 102 });
    const el = mountUi(LabeledBySourceCard, { stats });
    expect(el.querySelector('[data-testid="labeled-total"]')?.textContent).toContain(
      stats.labeled.total.toLocaleString(),
    );
    expect(el.querySelectorAll('li').length).toBe(stats.labeled.rows.length);
    expect(
      el.querySelector('[data-testid="labeled-not-ground-truth"]')?.textContent,
    ).toContain('102 of 937 crops are validated');
  });

  it('renders nothing until the stats arrive', () => {
    expect(
      mountUi(LabeledBySourceCard, { stats: null }).querySelector(
        '[data-testid="labeled-by-source"]',
      ),
    ).toBeNull();
  });
});
