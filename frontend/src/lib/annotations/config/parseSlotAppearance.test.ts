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
import { expectRejected, fullSlot } from '$lib/test/fixtures/slotConfigInputs';

describe('parseSlotConfig — text', () => {
  it('25. pattern with only source compiles with empty flags', () => {
    const s = fullSlot();
    (s.capabilities as any).text.pattern = { source: '^[0-9]{18}$' };
    const r = parseSlotConfig(s);
    expect(r.slot).toBeDefined();
    expect(r.slot!.capabilities.text!.pattern).toBeInstanceOf(RegExp);
    expect(r.slot!.capabilities.text!.pattern!.flags).toBe('');
  });

  it('26. pattern.flags "g" rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).text.pattern.flags = 'g';
    expectRejected(parseSlotConfig(s), /pattern\.flags/);
  });

  it('27. pattern.flags "x" rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).text.pattern.flags = 'x';
    expectRejected(parseSlotConfig(s), /pattern\.flags/);
  });

  it('28. pattern.source of 201 chars rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).text.pattern.source = 'a'.repeat(201);
    expectRejected(parseSlotConfig(s), /pattern\.source/);
  });

  it('29. nested-quantifier ReDoS shape rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).text.pattern.source = '(a+)+$';
    expectRejected(parseSlotConfig(s), /pattern\.source/);
  });

  it('30. unbalanced regex rejected by compile try/catch', () => {
    const s = fullSlot();
    (s.capabilities as any).text.pattern.source = '(';
    expectRejected(parseSlotConfig(s), /pattern\.source/);
  });

  it('31. transform: "titlecase" rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).text.transform = 'titlecase';
    expectRejected(parseSlotConfig(s), /transform/);
  });

  it('32. duplicate vocabulary value rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).text.vocabulary = [
      { value: 'a', label: 'A' },
      { value: 'a', label: 'A again' },
    ];
    expectRejected(parseSlotConfig(s), /vocabulary/);
  });
});
describe('parseSlotConfig — a multi-box slot has per-box text and detector (no item-level fields)', () => {
  it('text without valueField and provenance without detectorField parse', () => {
    const s = fullSlot();
    delete (s.capabilities as any).text.valueField;
    delete (s.capabilities as any).provenance.detectorField;
    const r = parseSlotConfig(s);
    expect(r.errors).toEqual([]);
    expect(r.slot!.capabilities.text!.valueField).toBeUndefined();
    expect(r.slot!.capabilities.provenance!.detectorField).toBeUndefined();
    expect(r.slot!.capabilities.provenance!.chainField).toBe('tag_detector_chain');
  });

  it('a present but malformed valueField is still rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).text.valueField = 5;
    expectRejected(parseSlotConfig(s), /valueField/);
  });
});
