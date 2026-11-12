import { describe, expect, it } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import type { ClusterRunController } from '$lib/clusters/clusterRunController.svelte';
import ClusterRunPanel from './ClusterRunPanel.svelte';

const ctl = {
  serving: null,
  pending: null,
  loadError: null,
  busy: false,
  jobStatus: null,
  message: null,
  actionError: null,
  load: async () => {},
} as unknown as ClusterRunController;

describe('ClusterRunPanel', () => {
  it('"Cluster items" is the primary button by default', () => {
    const el = mountUi(ClusterRunPanel, { ctl });
    expect(el.querySelector('[data-testid="cluster-run-start"]')!.className).toContain(
      'btn-primary',
    );
  });

  it('is secondary when the served next step belongs to another action', () => {
    const el = mountUi(ClusterRunPanel, { ctl, primary: false });
    const start = el.querySelector('[data-testid="cluster-run-start"]')!;
    expect(start.className).not.toContain('btn-primary');
    expect(start.textContent).toContain('Cluster items');
  });
});
