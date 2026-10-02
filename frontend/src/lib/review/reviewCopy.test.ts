/**
 * docs/design/visual-audit-2026-09-24.md R3 (empty queues gave no reason)
 * and R6 (raw ids in the deep-link toast / VLM empty reason).
 */
import { describe, expect, it } from 'vitest';
import {
  emptyQueueMessage,
  locateMissMessage,
  queuePosition,
  vlmEmptyReasonText,
} from './reviewCopy';
import { humanizeId } from '$lib/humanizeId';

describe('locateMissMessage (R6)', () => {
  it('never shows the raw `not_found` id', () => {
    const msg = locateMissMessage('not_found');
    expect(msg).not.toContain('not_found');
    expect(msg).toContain('no crop with that id exists');
  });

  it('explains `filtered_out` as filters / already reviewed', () => {
    const msg = locateMissMessage('filtered_out');
    expect(msg).not.toContain('filtered_out');
    expect(msg).toMatch(/filters/);
  });

  it('shows an unknown served reason verbatim, and a default with none', () => {
    expect(locateMissMessage('some_new_reason')).toContain('some_new_reason');
    expect(locateMissMessage(null)).toMatch(/may already be reviewed/);
  });
});

describe('emptyQueueMessage (R3)', () => {
  const base = {
    label: 'Uncertainty',
    description: 'High active-learning probe entropy',
    sortFallbackReason: null,
    filtersActive: false,
  };

  it('names the queue and says what it holds (served description)', () => {
    const m = emptyQueueMessage(base);
    expect(m.title).toBe('The Uncertainty queue is empty.');
    expect(m.lines[0]).toBe('This queue holds: High active-learning probe entropy.');
  });

  it("surfaces the served sort-fallback reason as why there's nothing to order", () => {
    const m = emptyQueueMessage({
      ...base,
      sortFallbackReason: "no item has 'probe_pred_entropy' yet",
    });
    expect(m.lines.join(' ')).toContain("no item has 'probe_pred_entropy' yet");
    expect(m.lines.join(' ')).not.toContain('Nothing currently needs review');
  });

  it('points at the operator filters when any are active', () => {
    const m = emptyQueueMessage({ ...base, filtersActive: true });
    expect(m.lines.join(' ')).toMatch(/filters may be hiding items/);
  });

  it('with no reason at all, says nothing needs review (never a bare "Queue empty.")', () => {
    const m = emptyQueueMessage({ ...base, description: null });
    expect(m.lines).toEqual(['Nothing currently needs review here.']);
  });
});

describe('emptyQueueMessage: served empty_reason (#36 item 9)', () => {
  const base = {
    label: 'Uncertainty',
    description: 'High active-learning probe entropy',
    sortFallbackReason:
      "default sort 'uncertainty_entropy' orders by 'probe_pred_entropy', which no item has yet",
    emptyReason: 'no probe predictions — run a probe',
    filtersActive: false,
  };

  it('the served empty_reason takes priority over the sort-fallback note', () => {
    const m = emptyQueueMessage(base);
    expect(m.lines).toContain('no probe predictions — run a probe');
    expect(m.lines.join(' ')).not.toContain("orders by 'probe_pred_entropy'");
  });

  it('falls back to sortFallbackReason when empty_reason is absent (an older backend)', () => {
    const m = emptyQueueMessage({ ...base, emptyReason: null });
    expect(m.lines.join(' ')).toContain("orders by 'probe_pred_entropy'");
  });

  it('links to /train when emptyState says no probe has ever run and the reason mentions a probe', () => {
    const m = emptyQueueMessage({
      ...base,
      emptyState: { has_probe_predictions: false, has_item_scores: true },
    });
    expect(m.link).toEqual({ href: '/train', text: 'Run a probe on /train' });
  });

  it('links to /settings when emptyState says no item scores exist and the reason mentions a score', () => {
    const m = emptyQueueMessage({
      ...base,
      emptyReason: 'no uniqueness score computed yet',
      sortFallbackReason: null,
      emptyState: { has_probe_predictions: true, has_item_scores: false },
    });
    expect(m.link).toEqual({ href: '/settings', text: 'Compute scores on /settings' });
  });

  it('no link when emptyState says the prerequisite IS populated (a genuinely empty queue)', () => {
    const m = emptyQueueMessage({
      ...base,
      emptyState: { has_probe_predictions: true, has_item_scores: true },
    });
    expect(m.link).toBeUndefined();
  });

  describe('the imported tab (W10)', () => {
    const imported = {
      ...base,
      emptyReason: null,
      sortFallbackReason: null,
      importedTab: true,
      datasetsAvailable: true,
      emptyState: {
        has_probe_predictions: true,
        has_item_scores: true,
        has_imported_labels: false,
      },
    };

    it('links to the import page when no import has written labels and W10 is served', () => {
      expect(emptyQueueMessage(imported).link).toEqual({
        href: '/datasets/import',
        text: 'Import a labeled dataset',
      });
    });

    it('no link once labels have been imported (a genuinely empty queue)', () => {
      const m = emptyQueueMessage({
        ...imported,
        emptyState: { ...imported.emptyState, has_imported_labels: true },
      });
      expect(m.link).toBeUndefined();
    });

    it('no link when the import page is not served', () => {
      expect(
        emptyQueueMessage({ ...imported, datasetsAvailable: false }).link,
      ).toBeUndefined();
    });

    it('no link when the flag is absent (an older backend)', () => {
      const m = emptyQueueMessage({
        ...imported,
        emptyState: { has_probe_predictions: true, has_item_scores: true },
      });
      expect(m.link).toBeUndefined();
    });

    it('no link on any other tab, whatever the flag says', () => {
      expect(emptyQueueMessage({ ...imported, importedTab: false }).link).toBeUndefined();
    });
  });

  it('no link when emptyState is absent (an older backend)', () => {
    const m = emptyQueueMessage({ ...base, emptyState: null });
    expect(m.link).toBeUndefined();
  });
});

describe('vlmEmptyReasonText / humanizeId (R6)', () => {
  it('turns a served id into prose', () => {
    expect(vlmEmptyReasonText('no_answer')).toBe('VLM gave no class — No answer');
  });

  it('keeps reader acronyms upper-case (was "Vlm Preferred")', () => {
    expect(humanizeId('vlm_preferred')).toBe('VLM preferred');
    expect(humanizeId('ocr_only')).toBe('OCR only');
    expect(humanizeId('readers_agree')).toBe('Readers agree');
  });
});

describe('queuePosition (F8 D6)', () => {
  it('is the served rank + 1 for a crop located on a later page', () => {
    // /locate: rank 67, page_size 30 -> page 3, index 7 within that page.
    expect(queuePosition(3, 30, 7)).toBe(68);
  });

  it('is the cursor + 1 when the buffer starts at page 1', () => {
    expect(queuePosition(1, 30, 0)).toBe(1);
    expect(queuePosition(1, 30, 45)).toBe(46);
  });
});
