/**
 * docs/design/visual-audit-2026-09-24.md R3 (empty queues gave no reason)
 * and R6 (raw ids in the deep-link toast / VLM empty reason).
 */
import { describe, expect, it } from 'vitest';
import { emptyQueueMessage, locateMissMessage, vlmEmptyReasonText } from './reviewCopy';
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
