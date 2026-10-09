/**
 * The audit components (#119) mounted against served report data: the
 * insufficient-sample state, the confusion matrix, the start form and the
 * queue list.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { AuditController } from '$lib/labelConfirmation/auditController.svelte';
import { classSourcesStore } from '$stores/classSources.svelte';
import { makeItem } from '$lib/test/makeItem';
import { mapRawCrop } from '$lib/api';
import AuditClassTable from './AuditClassTable.svelte';
import AuditQueueList from './AuditQueueList.svelte';
import AuditStartForm from './AuditStartForm.svelte';
import ConfusionMatrix from './ConfusionMatrix.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

function render(component: unknown, props: Record<string, unknown>) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(component as never, { target, props } as never);
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const stat = (over = {}) => ({
  name: 'widget_a',
  n: 40,
  correct: 38,
  precision: 0.95,
  ci_low: 0.835,
  ci_high: 0.986,
  insufficient_sample: false,
  ...over,
});

describe('AuditClassTable', () => {
  it('prints the served precision with its interval', () => {
    const el = render(AuditClassTable, {
      title: 'Detector precision',
      blurb: 'b',
      stats: [stat()],
      minPerClass: 30,
      testId: 'audit-detector',
    });
    const row = el.querySelector('[data-testid="audit-class-row"]') as HTMLElement;
    expect(row.textContent).toContain('widget_a');
    expect(row.textContent).toContain('95.0%');
    expect(row.textContent).toContain('83.5% to 98.6%');
    expect(row.dataset.insufficient).toBe('false');
    expect(el.querySelector('[data-testid="insufficient-sample"]')).toBeNull();
  });

  it('flags an insufficient sample and does not present its precision as trustworthy', () => {
    const el = render(AuditClassTable, {
      title: 'VLM precision',
      blurb: 'b',
      stats: [
        stat({
          name: 'widget_b',
          n: 4,
          correct: 4,
          precision: 1,
          insufficient_sample: true,
        }),
      ],
      minPerClass: 30,
      testId: 'audit-vlm',
    });
    const row = el.querySelector('[data-testid="audit-class-row"]') as HTMLElement;
    expect(row.dataset.insufficient).toBe('true');
    expect(
      row.querySelector('[data-testid="insufficient-sample"]')?.getAttribute('title'),
    ).toContain('Fewer than 30');
  });

  it('prints a dash for a class with no precision and says when nothing is audited', () => {
    let el = render(AuditClassTable, {
      title: 't',
      blurb: 'b',
      stats: [stat({ n: 0, correct: 0, precision: null, insufficient_sample: true })],
      minPerClass: 30,
      testId: 'x',
    });
    expect(el.querySelector('[data-testid="audit-class-row"]')?.textContent).toContain(
      '—',
    );
    unmount(instance!);
    target.remove();
    el = render(AuditClassTable, {
      title: 't',
      blurb: 'b',
      stats: [],
      minPerClass: 30,
      testId: 'x',
    });
    expect(el.textContent).toContain('No audited crops yet');
  });
});

describe('ConfusionMatrix', () => {
  const confusion = {
    widget_a: { widget_a: 7, widget_b: 2 },
    widget_b: { widget_b: 5 },
  };

  it('renders each served cell, with zero where a pair never occurred', () => {
    const el = render(ConfusionMatrix, { confusion });
    const cell = (d: string, h: string) =>
      el.querySelector(`[data-detector="${d}"][data-human="${h}"]`)?.textContent?.trim();
    expect(cell('widget_a', 'widget_a')).toBe('7');
    expect(cell('widget_a', 'widget_b')).toBe('2');
    expect(cell('widget_b', 'widget_a')).toBe('0');
    expect(cell('widget_b', 'widget_b')).toBe('5');
    expect(el.querySelectorAll('[data-testid="confusion-cell"]')).toHaveLength(4);
  });

  it('says so when nothing is audited', () => {
    const el = render(ConfusionMatrix, { confusion: {} });
    expect(el.textContent).toContain('No audited crops yet');
    expect(el.querySelector('table')).toBeNull();
  });
});

describe('AuditStartForm', () => {
  const makeAudit = (start = vi.fn()) =>
    new AuditController({
      report: vi.fn().mockResolvedValue({
        audited: 0,
        pending: 0,
        min_per_class: 30,
        detector: [],
        vlm: [],
        confusion: {},
        outcomes: {},
      }),
      queue: vi.fn().mockResolvedValue({ items: [], total: 0, page: 1, pageSize: 30 }),
      start,
    });

  it('draws with the typed values and shows the served strata, flagging a short one', async () => {
    const start = vi.fn().mockResolvedValue({
      batch_id: 'b',
      min_per_class: 30,
      requested: 300,
      sampled: 38,
      strata: [
        {
          detector_class: 'widget_a',
          available: 500,
          sampled: 30,
          short_of_floor: false,
        },
        { detector_class: 'widget_b', available: 90, sampled: 8, short_of_floor: true },
      ],
    });
    const audit = makeAudit(start);
    const el = render(AuditStartForm, { audit });
    const size = el.querySelector(
      '[data-testid="audit-sample-size"]',
    ) as HTMLInputElement;
    size.value = '300';
    size.dispatchEvent(new Event('input', { bubbles: true }));
    (el.querySelector('[data-testid="audit-start-button"]') as HTMLButtonElement).click();
    await vi.waitFor(() => {
      flushSync();
      expect(el.querySelector('[data-testid="audit-started"]')).not.toBeNull();
    });
    expect(start).toHaveBeenCalledWith({ min_per_class: undefined, sample_size: 300 });
    const strata = [
      ...el.querySelectorAll('[data-testid="audit-stratum"]'),
    ] as HTMLElement[];
    expect(strata.map((s) => s.dataset.short)).toEqual(['false', 'true']);
    expect(el.textContent).toContain('Drew 38 of the 300 requested crops');
  });

  it('shows the served refusal', async () => {
    const { ApiError } = await import('$lib/api');
    const audit = makeAudit(
      vi.fn().mockRejectedValue(
        new ApiError(409, 'u', {
          detail: { error: 'audit_no_candidates', message: 'no crop is eligible' },
        }),
      ),
    );
    const el = render(AuditStartForm, { audit });
    (el.querySelector('[data-testid="audit-start-button"]') as HTMLButtonElement).click();
    await vi.waitFor(() => {
      flushSync();
      expect(
        el.querySelector('[data-testid="audit-start-error"]')?.textContent,
      ).toContain('no crop is eligible');
    });
  });
});

describe('AuditQueueList', () => {
  const queue = (n: number, total = n) => ({
    items: Array.from({ length: n }, (_, i) =>
      mapRawCrop(
        makeItem({
          crop_id: `q${i}`,
          class_name: 'widget_a',
          class_source: 'vlm',
          detector_class_name: 'widget_b',
          detector_confidence: 0.61,
        }),
      ),
    ),
    total,
    page: 1,
    pageSize: 30,
  });

  it('lists each drawn crop with its class, the detector class and a Review link', () => {
    classSourcesStore.list = [
      { id: 'vlm', label: 'VLM', role: 'vlm', short_label: 'VLM' },
    ] as never;
    const el = render(AuditQueueList, { queue: queue(2), ongoto: () => {} });
    const rows = [...el.querySelectorAll('[data-testid="audit-queue-item"]')];
    expect(rows).toHaveLength(2);
    expect(rows[0]!.textContent).toContain('widget_a');
    expect(
      rows[0]!.querySelector('[data-testid="audit-queue-detector"]')?.textContent,
    ).toContain('widget_b');
    const href = rows[0]!
      .querySelector('[data-testid="audit-queue-open"]')!
      .getAttribute('href')!;
    expect(href).toContain('/review?tab=all&crop_id=q0');
  });

  it('pages by the served total and page size', () => {
    const ongoto = vi.fn();
    const el = render(AuditQueueList, { queue: queue(30, 65), ongoto });
    expect(el.textContent).toContain('Page 1 of 3');
    const next = [...el.querySelectorAll('button')].find(
      (b) => b.textContent === 'Next',
    )!;
    next.click();
    expect(ongoto).toHaveBeenCalledWith(2);
  });

  it('says nothing is waiting when the queue is empty', () => {
    const el = render(AuditQueueList, { queue: queue(0), ongoto: () => {} });
    expect(el.textContent).toContain('Nothing is waiting');
  });
});
