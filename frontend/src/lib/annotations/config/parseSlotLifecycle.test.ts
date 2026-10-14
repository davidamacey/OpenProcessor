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

describe('parseSlotConfig — lifecycle', () => {
  it('33. empty states rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).lifecycle.states = [];
    expectRejected(parseSlotConfig(s), /states/);
  });

  it('34. confirmState naming an absent value rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).lifecycle.confirmState = 'nope';
    expectRejected(parseSlotConfig(s), /confirmState/);
  });

  it('35. falsePositiveState naming an absent value rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).lifecycle.falsePositiveState = 'nope';
    expectRejected(parseSlotConfig(s), /falsePositiveState/);
  });

  it('36. states[].role: "maybe" rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).lifecycle.states[0].role = 'maybe';
    expectRejected(parseSlotConfig(s), /role/);
  });

  it('36a. states[].aliases: valid identifier array accepted', () => {
    const s = fullSlot();
    (s.capabilities as any).lifecycle.states[0].aliases = ['legacy_value'];
    const r = parseSlotConfig(s);
    expect(r.slot).toBeDefined();
    expect(r.slot!.capabilities.lifecycle!.states[0].aliases).toEqual(['legacy_value']);
  });

  it('36b. states[].aliases: non-string entry rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).lifecycle.states[0].aliases = [123];
    expectRejected(parseSlotConfig(s), /aliases/);
  });

  it('36c. states[].aliases: invalid identifier rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).lifecycle.states[0].aliases = ['../etc/passwd'];
    expectRejected(parseSlotConfig(s), /aliases/);
  });
});
