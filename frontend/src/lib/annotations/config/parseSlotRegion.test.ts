/**
 * Validator unit tests (docs/design/tier2-annotation-profile-config-plan-
 * 2026-09-20.md §5.2's 67 enumerated cases). Each invalid case asserts
 * `result.slot` is `undefined` AND that `errors` names the offending
 * field — a message that doesn't say what's wrong isn't a usable
 * operator-facing diagnostic.
 *
 * `any` is used, deliberately and only in this file, to mutate deeply
 * nested fixture objects into DELIBERATELY malformed shapes (wrong
 * types, missing required fields, injected forbidden keys) — the exact
 * opposite of what a fully-typed fixture could express. The parser under
 * test is `unknown`-typed throughout and never trusts these shapes.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { describe, expect, it } from 'vitest';
import { parseSlotConfig } from './parseSlotConfig';
import { RING_PRESETS } from './allowLists';
import { expectRejected, fullSlot } from '$lib/test/fixtures/slotConfigInputs';

describe('parseSlotConfig — subBox', () => {
  it('13. neither bboxField nor listField rejected; both rejected', () => {
    const s = fullSlot();
    delete (s.capabilities as any).subBox.bboxField;
    expectRejected(parseSlotConfig(s), /exactly one of listField/);

    const both = fullSlot();
    (both.capabilities as any).subBox.listField = 'region_boxes';
    expectRejected(parseSlotConfig(both), /exactly one of listField/);
  });

  it('13b. a multi-box slot declares listField (+ summary fields) and needs no storedFrame', () => {
    const s = fullSlot();
    const sb = (s.capabilities as any).subBox;
    delete sb.bboxField;
    delete sb.storedFrame;
    sb.listField = 'region_boxes';
    sb.countField = 'region_count';
    sb.rejectedCountField = 'region_rejected_count';
    sb.maxScoreField = 'region_max_score';
    sb.setCompleteField = 'region_set_complete';
    sb.revisionField = 'region_revision';
    const r = parseSlotConfig(s);
    expect(r.errors).toEqual([]);
    const out = r.slot!.capabilities.subBox!;
    expect(out.listField).toBe('region_boxes');
    expect(out.bboxField).toBeUndefined();
    expect(out.revisionField).toBe('region_revision');
    expect(out.setCompleteField).toBe('region_set_complete');
  });

  it('13c. a summary field that is not a wire field is rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).subBox.revisionField = 7;
    expectRejected(parseSlotConfig(s), /revisionField/);
  });

  it('13d. the thumbnail path template may use {boxId}', () => {
    const r = parseSlotConfig(fullSlot());
    expect(r.errors).toEqual([]);
    expect(r.slot!.capabilities.subBox!.thumbnail!.path('c/1', 'b 2', 160)).toBe(
      '/crops/c%2F1/region_thumbnail?box_id=b%202&size=160',
    );
  });

  it('14. storedFrame: "image" rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).subBox.storedFrame = 'image';
    expectRejected(parseSlotConfig(s), /storedFrame/);
  });

  it('15. unknown ring preset rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).subBox.ring = 'rainbow';
    expectRejected(parseSlotConfig(s), /ring/);
  });

  it('16. non-allow-listed ring object rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).subBox.ring = {
      confirmed: 'border-red-500',
      proposed: 'border-red-400',
      rejected: 'border-red-300',
    };
    expectRejected(parseSlotConfig(s), /ring/);
  });

  it('17. ring: "default" resolves to the exact RING_PRESETS.default strings', () => {
    const s = fullSlot();
    const r = parseSlotConfig(s);
    expect(r.slot!.capabilities.subBox!.ring).toEqual(RING_PRESETS.default);
  });

  it('18. absolute-URL thumbnail path rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).subBox.thumbnail.path = 'https://evil.example/x';
    expectRejected(parseSlotConfig(s), /thumbnail\.path/);
  });

  it('19. path-traversal thumbnail path rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).subBox.thumbnail.path = '/crops/../../etc/passwd';
    expectRejected(parseSlotConfig(s), /thumbnail\.path/);
  });

  it('20. placeholder-as-expression rejected, names the placeholder', () => {
    const s = fullSlot();
    (s.capabilities as any).subBox.thumbnail.path = '/crops/{cropId.toUpperCase()}/t';
    const r = parseSlotConfig(s);
    expect(r.slot).toBeUndefined();
    expect(r.errors.some((e) => e.includes('cropId.toUpperCase()'))).toBe(true);
  });

  it('21. non-allow-listed placeholder rejected with "not in the allow-list"', () => {
    const s = fullSlot();
    (s.capabilities as any).subBox.thumbnail.path = '/crops/{userId}/t';
    const r = parseSlotConfig(s);
    expect(r.errors.some((e) => /not in the allow-list/.test(e))).toBe(true);
  });

  it('22. malicious aspect string rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).subBox.thumbnail.aspect = 'url(javascript:alert(1))';
    expectRejected(parseSlotConfig(s), /aspect/);
  });

  it('23. editor.nudgeStep: 0 and editor.thumbSize: "512" rejected', () => {
    const s1 = fullSlot();
    (s1.capabilities as any).subBox.editor.nudgeStep = 0;
    expectRejected(parseSlotConfig(s1), /nudgeStep/);

    const s2 = fullSlot();
    (s2.capabilities as any).subBox.editor.thumbSize = '512';
    expectRejected(parseSlotConfig(s2), /thumbSize/);
  });

  it('24. scoreField: a non-wire-field value rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).subBox.scoreField = 123;
    expectRejected(parseSlotConfig(s), /scoreField/);
  });
});
