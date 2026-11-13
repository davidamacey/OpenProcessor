import { describe, expect, it } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import { datasetStatsFixture } from '$lib/test/fixtures/datasetStats';
import DashboardKpis from './DashboardKpis.svelte';

describe('DashboardKpis', () => {
  it('shows the served counts and the served detector row label', () => {
    const base = datasetStatsFixture({ total_crops: 285, validated: 120 });
    const stats = {
      ...base,
      labeled: {
        ...base.labeled,
        rows: [
          ...base.labeled.rows,
          { key: 'by_detector', label: 'Detector, unchecked', count: 1131, share: 0.5 },
        ],
      },
    };
    const el = mountUi(DashboardKpis, { stats });
    const text = (id: string) =>
      el.querySelector(`[data-kpi="${id}"] dd`)?.textContent?.trim();
    expect(text('objects')).toBe(stats.total_crops.toLocaleString());
    expect(text('validated')).toBe(stats.validated.toLocaleString());
    expect(text('no-class')).toBe(stats.unlabeled.no_label_source.toLocaleString());
    expect(el.querySelector('[data-kpi="detector"] dt')?.textContent).toBe(
      'Detector, unchecked',
    );
    expect(text('detector')).toBe('1,131');
  });

  it('renders nothing until the stats arrive', () => {
    expect(mountUi(DashboardKpis, { stats: null }).querySelector('dl')).toBeNull();
  });
});
