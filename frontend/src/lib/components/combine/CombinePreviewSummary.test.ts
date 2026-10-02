import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import CombinePreviewSummary from './CombinePreviewSummary.svelte';
import { combinePreview } from '$lib/test/fixtures/combine';
import type { CombinePreview } from '$lib/types_combine';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

function render(preview: CombinePreview, stale = false) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CombinePreviewSummary, { target, props: { preview, stale } });
  flushSync();
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const t = (id: string) =>
  target.querySelector(`[data-testid="${id}"]`)?.textContent ?? '';

describe('CombinePreviewSummary', () => {
  it('prints the served counts, target classes, dedup and bytes verbatim', () => {
    render(combinePreview());
    expect(t('combine-preview-verdict')).toContain('No blocking problems');
    const rows = target.querySelectorAll(
      '[data-testid="combine-preview-sources"] tbody tr',
    );
    expect(rows).toHaveLength(2);
    expect(rows[0]!.textContent).toContain('widgets-a');
    // images, items, labeled, test images — each its own served number.
    expect(
      [...rows[0]!.querySelectorAll('td')].slice(1).map((td) => td.textContent?.trim()),
    ).toEqual(['10', '10', '4', '1']);
    expect(
      [...rows[1]!.querySelectorAll('td')].slice(1).map((td) => td.textContent?.trim()),
    ).toEqual(['10', '5', '4', '1']);
    expect(t('combine-preview-target')).toContain('merged');
    expect(t('combine-preview-target')).toContain('18 images');
    expect(t('combine-preview-target')).toContain('15 items');
    expect(t('combine-preview-classes')).toContain('widget');
    expect(t('combine-preview-classes')).toContain(
      'from widgets-a/widget, widgets-b/widget',
    );
    expect(t('combine-preview-dedup')).toContain('2 identical images');
    expect(t('combine-preview-dedup')).toContain('1 conflicts');
    expect(t('combine-preview-dedup')).toContain('near duplicates: not computed');
    expect(
      target.querySelector('[data-testid="combine-conflict-samples"]'),
    ).not.toBeNull();
    expect(t('combine-preview-bytes')).toContain('2.0 KB');
    expect(t('combine-preview-bytes')).toContain('to copy: 0 B');
  });

  it('shows served errors and warnings, and a blocked verdict when ok is false', () => {
    render(
      combinePreview({
        ok: false,
        errors: [{ code: 'slug_taken', message: "'merged' cannot be used" }],
        warnings: [
          { code: 'shard_budget_high', severity: 'warning', message: 'near the budget' },
        ],
        target: { slug: 'merged', slug_available: false },
      }),
    );
    expect(t('combine-preview-verdict')).toContain('cannot start');
    expect(t('combine-errors')).toContain("'merged' cannot be used");
    expect(t('combine-warnings')).toContain('near the budget');
    expect(t('combine-slug-unavailable')).toContain('not available');
  });

  it('missing served numbers read as a dash, never a zero; an estimate prints when served', () => {
    render(
      combinePreview({
        target: { slug: 'merged' },
        dedup: { near_duplicate_pairs_estimate: 7 },
        bytes: {},
      }),
    );
    expect(t('combine-preview-target')).toContain('— images');
    expect(t('combine-preview-dedup')).toContain('— identical images');
    expect(t('combine-preview-dedup')).toContain('7 near-duplicate pairs (estimate)');
    expect(t('combine-preview-bytes')).toContain('to link: —');
    expect(target.querySelector('[data-testid="combine-conflict-samples"]')).toBeNull();
  });

  it('marks a stale preview as updating', () => {
    render(combinePreview(), true);
    expect(t('combine-preview-verdict')).toContain('Updating');
    expect(
      target.querySelector('[data-testid="combine-preview"]')!.getAttribute('data-stale'),
    ).toBe('true');
  });
});
