import { describe, expect, it } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import type { PreflightReport } from '$lib/types_train';
import TrainPreflightPanel from './TrainPreflightPanel.svelte';

const check = (name: string, severity: 'ok' | 'info' | 'warn' | 'block') => ({
  name,
  label: `${name[0]!.toUpperCase()}${name.slice(1)} check`,
  severity,
  message: `${name} message`,
  detail: {},
});

const report = {
  blocked: false,
  summary: '1 warning',
  checks: [
    check('gpu', 'ok'),
    check('disk', 'ok'),
    check('holdout', 'warn'),
    check('classes', 'ok'),
  ],
} as unknown as PreflightReport;

describe('TrainPreflightPanel', () => {
  it('lists the warnings and blockers and folds the OK rows into one disclosure', () => {
    const el = mountUi(TrainPreflightPanel, { preflight: report, preflighting: false });
    const open = [...el.querySelectorAll('[data-testid="preflight-open-rows"] > li')].map(
      (li) => li.textContent ?? '',
    );
    expect(open.some((t) => t.includes('Holdout check'))).toBe(true);
    expect(open.some((t) => t.includes('Gpu check'))).toBe(false);
    const folded = el.querySelector('[data-testid="preflight-ok-rows"]')!;
    expect(folded.tagName).toBe('DETAILS');
    expect(folded.querySelector('summary')!.textContent).toContain('3 checks passed');
    expect(folded.textContent).toContain('Gpu check');
  });

  it('shows the served label, never the check id, and the severity as a word', () => {
    const el = mountUi(TrainPreflightPanel, { preflight: report, preflighting: false });
    const row = el.querySelector('[data-testid="preflight-open-rows"] > li')!;
    expect(row.querySelector('[data-testid="preflight-label"]')!.textContent).toBe(
      'Holdout check',
    );
    expect(row.querySelector('[data-testid="preflight-severity"]')!.textContent).toBe(
      'Warning',
    );
    expect(row.querySelector('code')).toBeNull();
  });

  it('with only OK rows there is no open list, just the disclosure', () => {
    const el = mountUi(TrainPreflightPanel, {
      preflight: { ...report, checks: [check('gpu', 'ok')] } as PreflightReport,
      preflighting: false,
    });
    expect(el.querySelector('[data-testid="preflight-open-rows"]')).toBeNull();
    expect(
      el.querySelector('[data-testid="preflight-ok-rows"] summary')!.textContent,
    ).toContain('1 check passed');
  });

  it('never shows a bare header: no report and an empty report each say so', () => {
    const none = mountUi(TrainPreflightPanel, { preflight: null, preflighting: false });
    expect(none.querySelector('[data-testid="preflight-empty"]')!.textContent).toContain(
      'Not checked yet',
    );
    const empty = mountUi(TrainPreflightPanel, {
      preflight: { blocked: false, summary: '', checks: [] } as PreflightReport,
      preflighting: false,
    });
    expect(empty.querySelector('[data-testid="preflight-empty"]')!.textContent).toContain(
      'No checks',
    );
  });
});
