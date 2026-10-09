import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { AuditController, type AuditDeps } from './auditController.svelte';
import type { AuditReport, AuditStartResponse } from '$lib/types_labelConfirmation';

const REPORT: AuditReport = {
  audited: 12,
  pending: 40,
  min_per_class: 30,
  detector: [],
  vlm: [],
  confusion: {},
  outcomes: { agree: 9, detector_wrong: 3 },
};
const QUEUE = { items: [], total: 40, page: 1, pageSize: 30 };
const STARTED: AuditStartResponse = {
  batch_id: 'b1',
  min_per_class: 30,
  requested: 300,
  sampled: 120,
  strata: [
    { detector_class: 'widget_a', available: 500, sampled: 30, short_of_floor: false },
  ],
};

function make(over: Partial<AuditDeps> = {}) {
  const deps: AuditDeps = {
    report: vi.fn().mockResolvedValue(REPORT),
    queue: vi.fn().mockResolvedValue(QUEUE),
    start: vi.fn().mockResolvedValue(STARTED),
    ...over,
  };
  return { audit: new AuditController(deps), deps };
}

describe('AuditController', () => {
  it('loads the served report and the first queue page', async () => {
    const { audit, deps } = make();
    await audit.load();
    expect(audit.report).toEqual(REPORT);
    expect(audit.queue?.total).toBe(40);
    expect(audit.loading).toBe(false);
    expect(deps.queue).toHaveBeenCalledWith(1, 30, null, undefined);
  });

  it('shows the served error when a read fails', async () => {
    const { audit } = make({
      report: vi.fn().mockRejectedValue(new ApiError(503, 'u', { detail: 'index down' })),
    });
    await audit.load();
    expect(audit.loadError).toBe('index down');
    expect(audit.report).toBeNull();
  });

  it('draws with only the inputs the operator filled, then re-reads', async () => {
    const { audit, deps } = make();
    audit.sampleSize = 120;
    expect(await audit.start()).toBe(true);
    expect(deps.start).toHaveBeenCalledWith({
      min_per_class: undefined,
      sample_size: 120,
    });
    expect(audit.started).toEqual(STARTED);
    expect(deps.report).toHaveBeenCalledTimes(1);
  });

  it('shows the served refusal of a draw with no candidates and keeps the old report', async () => {
    const { audit } = make({
      start: vi.fn().mockRejectedValue(
        new ApiError(409, 'u', {
          detail: { error: 'audit_no_candidates', message: 'no crop is eligible' },
        }),
      ),
    });
    await audit.load();
    expect(await audit.start()).toBe(false);
    expect(audit.startLines).toEqual(['no crop is eligible']);
    expect(audit.started).toBeNull();
    expect(audit.report).toEqual(REPORT);
    expect(audit.starting).toBe(false);
  });

  it('pages the queue', async () => {
    const queue = vi.fn().mockResolvedValue({ ...QUEUE, page: 3 });
    const { audit } = make({ queue });
    await audit.goToPage(3);
    expect(queue).toHaveBeenCalledWith(3, 30, null);
    expect(audit.queue?.page).toBe(3);
    expect(audit.queuePage).toBe(3);
  });
});
