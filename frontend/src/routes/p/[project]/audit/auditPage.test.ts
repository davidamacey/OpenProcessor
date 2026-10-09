/**
 * /audit mounted against stubbed served data (#119): the report tables, the
 * confusion matrix, the queue, and the empty and failed states.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { makeItem } from '$lib/test/makeItem';
import AuditPage from './+page.svelte';

let target: HTMLDivElement;
let instance: unknown;

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });

const REPORT = {
  audited: 52,
  pending: 31,
  min_per_class: 30,
  detector: [
    {
      name: 'widget_a',
      n: 40,
      correct: 38,
      precision: 0.95,
      ci_low: 0.83,
      ci_high: 0.99,
      insufficient_sample: false,
    },
    {
      name: 'widget_b',
      n: 12,
      correct: 6,
      precision: 0.5,
      ci_low: 0.25,
      ci_high: 0.75,
      insufficient_sample: true,
    },
  ],
  vlm: [
    {
      name: 'widget_a',
      n: 30,
      correct: 27,
      precision: 0.9,
      ci_low: 0.74,
      ci_high: 0.97,
      insufficient_sample: false,
    },
  ],
  confusion: {
    widget_a: { widget_a: 38, widget_b: 2 },
    widget_b: { widget_b: 6, widget_a: 6 },
  },
  outcomes: { agree: 44, detector_wrong: 5, vlm_wrong: 2, both_wrong: 1 },
};

function stub(over: { report?: () => Response; queue?: () => Response } = {}) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((url: string) => {
      if (url.includes('/audit/report'))
        return Promise.resolve((over.report ?? (() => json(REPORT)))());
      if (url.includes('/audit/queue')) {
        return Promise.resolve(
          (
            over.queue ??
            (() =>
              json({
                items: [makeItem({ crop_id: 'q1', detector_class_name: 'widget_b' })],
                total: 31,
                page: 1,
                page_size: 30,
              }))
          )(),
        );
      }
      return Promise.resolve(json({}));
    }),
  );
}

async function mountPage(): Promise<void> {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(AuditPage, { target } as never);
  await vi.waitFor(
    () => {
      flushSync();
      expect(
        target.querySelector(
          '[data-testid="audit-load-error"],[data-testid="audit-summary"]',
        ),
      ).not.toBeNull();
    },
    { timeout: 8000 },
  );
}

const q = (sel: string) => target.querySelector(sel) as HTMLElement | null;

afterEach(() => {
  if (instance) unmount(instance as never);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
});

describe('/audit', () => {
  it('shows the served counts, both precision tables, the matrix and the queue', async () => {
    stub();
    await mountPage();
    expect(q('[data-testid="audit-audited"]')?.textContent).toBe('52');
    expect(q('[data-testid="audit-pending"]')?.textContent).toBe('31');
    expect(q('[data-testid="audit-outcomes"]')?.textContent).toContain('Detector wrong');
    expect(q('[data-testid="audit-outcomes"]')?.textContent).toContain('Both wrong');
    expect(
      q('[data-testid="audit-detector"]')?.querySelectorAll(
        '[data-testid="audit-class-row"]',
      ),
    ).toHaveLength(2);
    expect(
      q('[data-testid="audit-vlm"]')?.querySelectorAll('[data-testid="audit-class-row"]'),
    ).toHaveLength(1);
    expect(
      q('[data-testid="audit-confusion"]')?.querySelectorAll(
        '[data-testid="confusion-cell"]',
      ),
    ).toHaveLength(4);
    expect(q('[data-testid="audit-queue-total"]')?.textContent).toContain('31');
  });

  it('marks the class with too few audited crops as an insufficient sample', async () => {
    stub();
    await mountPage();
    const rows = [
      ...q('[data-testid="audit-detector"]')!.querySelectorAll<HTMLElement>(
        '[data-testid="audit-class-row"]',
      ),
    ];
    expect(rows.map((r) => r.dataset.insufficient)).toEqual(['false', 'true']);
  });

  it('shows the served error and a retry when the report cannot be read', async () => {
    stub({ report: () => json({ detail: 'audit index missing' }, 503) });
    await mountPage();
    expect(q('[data-testid="audit-load-error"]')?.textContent).toContain(
      'audit index missing',
    );
    expect(q('[data-testid="audit-summary"]')).toBeNull();
  });
});
