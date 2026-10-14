/**
 * How /review words and styles rejections: the rejection_reasons vocabulary,
 * region_bbox_correct, and the per-item reason (region_rejection_reason wins
 * over the generic reason string). Static source scan (no
 * @testing-library/svelte in this repo; see reviewFilterConsistency.test.ts's
 * header comment for the established precedent).
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './+page.svelte'), 'utf-8');

describe("a box's bbox_correct verdict folded into the Status row", () => {
  it('the inline slot panel renders a "model: box wrong" chip gated on the selected box\'s bboxCorrect === false', () => {
    expect(src).toMatch(/selectedSlotBox\?\.bboxCorrect === false/);
    expect(src).toMatch(/model: box wrong/);
  });
});

describe('per-item reason wording — region_rejection_reason wins over the generic reason string', () => {
  it('currentSlotRejectionReason reads the first rejected box (W8 multi-box) or the item-level field (tier-2 single-box)', () => {
    expect(src).toMatch(/const currentSlotRejectionReason = \$derived<string \| null>\(/);
    // the
    // machine reason moved off the item onto each rejected SlotBox.
    expect(src).toMatch(
      /slotOf\(current, activeSlot\)\?\.subBoxes\?\.find\(\(b\) => b\.state === 'rejected'\)\s*\?\.rejectionReason \?\? null/,
    );
    // Tier-2 single-box fallback, unchanged.
    expect(src).toMatch(
      /slotOf\(current, activeSlot\)\?\.lifecycle\?\.rejectionReason \?\? null/,
    );
  });

  it('the Reason row falls back to current.reason ONLY when there is no currentSlotRejectionReason', () => {
    const idx = src.indexOf('{#if currentSlotRejectionReason}');
    expect(idx).toBeGreaterThan(-1);
    const block = src.slice(idx, src.indexOf('{/if}', idx) + 5);
    expect(block).toMatch(/\{:else\}/);
    expect(block).toMatch(/current\.reason \?\? '—'/);
    // The rejection branch must use the vocabulary label, never the raw
    // per-item `reason` string (which always says "verifier rejected
    // this candidate (…)", wrong wording for a needs_human item).
    const rejectionBranch = block.slice(0, block.indexOf('{:else}'));
    expect(rejectionBranch).toMatch(/regionVocabularyStore\.rejectionReasonLabel\(/);
    expect(rejectionBranch).not.toMatch(/current\.reason/);
  });

  it("a needs_human item's Reason-row label is never worded as a rejection", () => {
    const idx = src.indexOf('{#if currentSlotRejectionReason}');
    const block = src.slice(idx, src.indexOf('{:else}', idx));
    expect(block).toMatch(/reasonKind === 'needs_human' \? 'Needs review' : 'Rejection'/);
  });
});

describe('rejected-box badges style by the served rejection kind (never "rejected" for needs_human)', () => {
  it('the W8 multi-box panel badges each rejected box, branching on rejectionReasonKind, not just presence of a reason', () => {
    // no
    // separate "candidate" concept — one badge per rejected SlotBox.
    expect(src).toMatch(/<RejectedBoxChips\s+boxes=\{multiBox\.boxes\}/);
    const chips = readFileSync(
      path.resolve(here, '../../../../lib/components/review/RejectedBoxChips.svelte'),
      'utf-8',
    );
    expect(chips).toMatch(/boxes\.filter\(\(b\) => b\.state === 'rejected'\)/);
    expect(chips).toMatch(/\{@const kind = regionVocabularyStore\.rejectionReasonKind\(/);
    expect(chips).toMatch(
      /kind === 'needs_human' \? 'candidate · needs review' : 'rejected · confirm to accept'/,
    );
  });
});
