import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

vi.mock('$lib/api_select', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_select')>('$lib/api_select');
  return { ...actual, batchExcludeSelection: vi.fn(), getSelectionSummary: vi.fn() };
});
vi.mock('$lib/api_cropWrites', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_cropWrites')>('$lib/api_cropWrites');
  return { ...actual, getExcludeReasons: vi.fn() };
});

import { getExcludeReasons } from '$lib/api_cropWrites';
import { batchExcludeSelection } from '$lib/api_select';
import { excludeReasonsStore } from '$stores/excludeReasons.svelte';
import BulkBar from './BulkBar.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;
const tick = () => new Promise((r) => setTimeout(r, 0));
const q = (sel: string) => target.querySelector<HTMLElement>(sel);

function fakeWs(summaryText: string | null) {
  return {
    selection: { kind: 'items', cropIds: ['a', 'b'] },
    pendingSelectAll: null,
    summary: {
      summary: summaryText ? { summary_text: summaryText } : null,
      error: null,
    },
    itemSelection: () => ({ crop_ids: ['a', 'b'] }),
    clearSelection: vi.fn(),
    confirmSelectAllMatching: vi.fn(),
  };
}

function render(ws: ReturnType<typeof fakeWs>): void {
  instance = mount(BulkBar, { target, props: { ws } as never });
  flushSync();
}

beforeEach(() => {
  excludeReasonsStore.reset();
  vi.mocked(getExcludeReasons).mockResolvedValue([
    { id: 'ignore', label: 'Ignore (generic)' },
    { id: 'blurry', label: 'Blurry' },
  ]);
  target = document.createElement('div');
  document.body.appendChild(target);
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
  vi.clearAllMocks();
});

describe('BulkBar', () => {
  it('shows the served summary text verbatim', () => {
    render(fakeWs('183 detections in 41 images · 7 locked, skipped'));
    expect(q('[data-testid="bulk-summary"]')!.textContent).toBe(
      '183 detections in 41 images · 7 locked, skipped',
    );
  });

  it('dry-runs before confirm', async () => {
    vi.mocked(batchExcludeSelection).mockResolvedValue({ dry_run: true, selected: 2 });
    render(fakeWs('2 detections in 1 image'));
    q('[data-testid="ignore-selected"]')!.click();
    await tick();
    flushSync();
    expect(batchExcludeSelection).toHaveBeenCalledTimes(1);
    expect(vi.mocked(batchExcludeSelection).mock.calls[0]).toEqual([
      { crop_ids: ['a', 'b'] },
      'ignore',
      true,
      expect.anything(),
    ]);
  });

  it('renders the served exclude reasons', async () => {
    render(fakeWs('2 detections in 1 image'));
    await tick();
    flushSync();
    target.querySelector<HTMLElement>('[aria-label="Choose ignore reason"]')!.click();
    flushSync();
    const labels = [
      ...target.querySelectorAll('[data-testid="ignore-reasons"] button'),
    ].map((b) => b.textContent?.trim());
    expect(labels).toEqual(['Ignore (generic)', 'Blurry']);
  });

  it('offers Label whole group with nothing selected inside an open duplicate group', () => {
    const ws = {
      ...fakeWs(null),
      selection: { kind: 'none' },
      inNearDupGroup: true,
      itemBody: () => ({ near_dup_group: 'g1' }),
    };
    render(ws as never);
    expect(q('[data-testid="bulk-bar"]')).toBeNull();
    expect(
      q('[data-testid="group-bar"] [data-testid="label-whole-group"]'),
    ).not.toBeNull();
  });

  it('shows no bar with nothing selected outside a duplicate group', () => {
    render({
      ...fakeWs(null),
      selection: { kind: 'none' },
      inNearDupGroup: false,
    } as never);
    expect(q('[data-testid="bulk-bar"]')).toBeNull();
    expect(q('[data-testid="group-bar"]')).toBeNull();
  });
});
