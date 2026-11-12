import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import DataQualityCard from './DataQualityCard.svelte';

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });

const AUDIT = {
  audited: 52,
  pending: 31,
  min_per_class: 30,
  detector: [],
  vlm: [],
  confusion: {},
  outcomes: {},
};
const NEAR = {
  profile: null,
  levels: [
    {
      level: 'image',
      pending_count: 4,
      stale_reasons: ['pending'],
      current_run: {
        dedup_groups: 12,
        leak_groups: 3,
        finished_at: '2026-01-01T00:00:00Z',
      },
    },
    { level: 'item', pending_count: 0, stale_reasons: [], current_run: null },
  ],
};

function serve(audit: () => Response, near: () => Response): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const u = String(url);
      if (u.includes('/audit/report')) return audit();
      if (u.includes('/near_duplicates/status')) return near();
      return json({}, 404);
    }),
  );
}
afterEach(() => vi.unstubAllGlobals());
const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);

async function render(): Promise<HTMLElement> {
  const el = mountUi(DataQualityCard, {});
  await vi.waitFor(() => {
    flushSync();
    expect(
      el.querySelector('[data-testid="dq-audit"],[data-testid="error-state"]'),
    ).not.toBeNull();
  });
  return el;
}

describe('DataQualityCard', () => {
  it('shows the served audit and duplicate counts and links to Data quality', async () => {
    serve(
      () => json(AUDIT),
      () => json(NEAR),
    );
    const el = await render();
    expect(q('dq-audit')!.textContent).toContain('52');
    expect(q('dq-audit')!.textContent).toContain('31');
    await vi.waitFor(() => {
      flushSync();
      expect(q('dq-duplicates')!.textContent).toContain('12');
    });
    expect(q('dq-duplicates')!.textContent).toContain('3');
    const hrefs = [...el.querySelectorAll('a')].map((a) => a.getAttribute('href'));
    expect(hrefs.some((h) => h?.endsWith('/quality/audit'))).toBe(true);
    expect(hrefs.some((h) => h?.endsWith('/quality/near-duplicates'))).toBe(true);
  });

  it('an unaudited project and a never-run duplicate scan say so, with no invented zero', async () => {
    serve(
      () => json({ ...AUDIT, audited: 0, pending: 0 }),
      () => json({ profile: null, levels: [{ ...NEAR.levels[1], level: 'image' }] }),
    );
    await render();
    expect(q('dq-audit')!.textContent).toContain('No audit yet');
    await vi.waitFor(() => {
      flushSync();
      expect(q('dq-duplicates')!.textContent).toContain('Never run');
    });
  });

  it('a failed read is an ErrorState with the request id and a Retry', async () => {
    let calls = 0;
    serve(
      () => {
        calls += 1;
        return calls === 1
          ? json({ detail: 'Not Found' }, 404, { 'x-request-id': 'req-8' })
          : json(AUDIT);
      },
      () => json(NEAR),
    );
    const el = await render();
    expect(el.querySelector('[data-testid="error-state-request-id"]')!.textContent).toBe(
      'req-8',
    );
    [...el.querySelectorAll('button')]
      .find((b) => b.textContent?.includes('Retry'))!
      .click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('dq-audit')).not.toBeNull();
    });
  });
});
