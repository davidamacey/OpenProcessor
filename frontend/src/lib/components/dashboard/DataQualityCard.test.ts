import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import { projectsStore } from '$stores/projects.svelte';
import { testProject } from '$lib/test/fixtures/projects';
import type { DatasetReadiness } from '$lib/types_quality';
import DataQualityCard from './DataQualityCard.svelte';

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });

const READINESS: DatasetReadiness = {
  verdict: 'blocked',
  computed_at: '2026-11-12T18:20:00Z',
  coverage: [],
  checks: [
    {
      id: 'test_holdout',
      label: 'Test set',
      severity: 'block',
      message: '2 classes have under 5 test examples.',
      count: 2,
      link: { section: 'export', query: null },
      classes: [],
    },
    {
      id: 'near_dup',
      label: 'Near-duplicate groups',
      severity: 'ok',
      message: 'Up to date.',
      count: null,
      link: null,
      classes: [],
    },
  ],
};

function serve(answer: () => Response): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) =>
      String(url).includes('/quality/readiness') ? answer() : json({}, 404),
    ),
  );
}
afterEach(() => vi.unstubAllGlobals());
const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);

async function render(): Promise<HTMLElement> {
  projectsStore.current = testProject({ slug: 'demo' });
  const el = mountUi(DataQualityCard, {});
  await vi.waitFor(() => {
    flushSync();
    expect(
      el.querySelector('[data-testid="dq-verdict"],[data-testid="error-state"]'),
    ).not.toBeNull();
  });
  return el;
}

describe('DataQualityCard', () => {
  it('shows the served verdict and only the checks that still need work', async () => {
    serve(() => json(READINESS));
    await render();
    expect(q('dq-verdict')!.textContent).toBe('Not ready');
    const open = q('dq-open-checks')!;
    expect(open.textContent).toContain('Test set');
    expect(open.textContent).toContain('2 classes have under 5 test examples.');
    expect(open.textContent).not.toContain('Near-duplicate groups');
  });

  it('links into Data quality and into the Workspace Duplicates view', async () => {
    serve(() => json(READINESS));
    await render();
    expect(q('dq-open-ready')!.getAttribute('href')).toBe('/p/demo/quality/ready');
    expect(q('dq-open-duplicates')!.getAttribute('href')).toBe(
      '/p/demo/workspace?view=gallery&near_dup=collapsed',
    );
  });

  it('says every check passed when none needs work', async () => {
    serve(() => json({ ...READINESS, verdict: 'ready', checks: [READINESS.checks[1]] }));
    const el = await render();
    expect(q('dq-verdict')!.textContent).toBe('Ready to train');
    expect(el.textContent).toContain('Every check passed.');
  });

  it('a failed read is an ErrorState with the request id and a Retry', async () => {
    let calls = 0;
    serve(() => {
      calls += 1;
      return calls === 1
        ? json({ detail: 'Not Found' }, 404, { 'x-request-id': 'req-8' })
        : json(READINESS);
    });
    const el = await render();
    expect(el.querySelector('[data-testid="error-state-request-id"]')!.textContent).toBe(
      'req-8',
    );
    [...el.querySelectorAll('button')]
      .find((b) => b.textContent?.includes('Retry'))!
      .click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('dq-verdict')).not.toBeNull();
    });
  });
});
