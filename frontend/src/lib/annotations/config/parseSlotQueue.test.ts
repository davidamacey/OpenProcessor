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
import type { ParseContext } from './parseCommon';
import { expectRejected, fullSlot } from '$lib/test/fixtures/slotConfigInputs';

describe('parseSlotConfig — queue', () => {
  it('37. endpointId with path traversal rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.endpointId = '../review/all';
    expectRejected(parseSlotConfig(s), /endpointId/);
  });

  it('38. urlId "all" (core tab collision) rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.urlId = 'all';
    expectRejected(parseSlotConfig(s), /urlId/);
  });

  it('39. urlId "mismatches" (preset collision) rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.urlId = 'mismatches';
    expectRejected(parseSlotConfig(s), /urlId/);
  });

  it('40. endpointId colliding with ctx.takenEndpointIds rejected', () => {
    const s = fullSlot();
    const ctx: ParseContext = { takenEndpointIds: new Set(['widget_tags']) };
    expectRejected(parseSlotConfig(s, ctx), /endpointId/);
  });

  it('41. keymap.reject: ["n"] rejected, cites the global skip binding', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = { confirm: ['enter'], reject: ['n'] };
    expectRejected(parseSlotConfig(s), /reserved/);
  });

  it('42. keymap.reject: ["z"] rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = { confirm: ['enter'], reject: ['z'] };
    expectRejected(parseSlotConfig(s), /reserved/);
  });

  it('43. keymap.editBox: ["escape"] rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = { confirm: ['enter'], editBox: ['escape'] };
    expectRejected(parseSlotConfig(s), /reserved/);
  });

  it('44. keymap.reject: ["enter"] rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = { confirm: ['enter'], reject: ['enter'] };
    expectRejected(parseSlotConfig(s), /reserved/);
  });

  it('45. keymap.back: ["arrowright"] rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = { confirm: ['enter'], back: ['arrowright'] };
    expectRejected(parseSlotConfig(s), /reserved/);
  });

  it('46. keymap.confirm: ["x"] rejected — "confirm is always enter"', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = { confirm: ['x'] };
    expectRejected(parseSlotConfig(s), /confirm is always/);
  });

  it('47. keymap.confirm: ["enter"] accepted', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = { confirm: ['enter'] };
    const r = parseSlotConfig(s);
    expect(r.slot).toBeDefined();
  });

  it('48. keymap combo "ctrl+k" rejected (not in the vocabulary)', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = { confirm: ['enter'], reject: ['ctrl+k'] };
    expectRejected(parseSlotConfig(s), /unrecognized combo/);
  });

  it('49. unknown keymap action rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = { confirm: ['enter'], teleport: ['q'] };
    expectRejected(parseSlotConfig(s), /unknown action/);
  });

  it('50. cross-slot: claimed "f" by markFalsePositive, this slot claims editBox: ["f"] -> rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = { confirm: ['enter'], editBox: ['f'] };
    const ctx: ParseContext = { claimedCombos: new Map([['f', 'markFalsePositive']]) };
    expectRejected(parseSlotConfig(s, ctx), /already bound/);
  });

  it('51. cross-slot: claimed "f" by markFalsePositive, this slot ALSO claims markFalsePositive: ["f"] -> accepted', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = {
      confirm: ['enter'],
      markFalsePositive: ['f'],
    };
    const ctx: ParseContext = { claimedCombos: new Map([['f', 'markFalsePositive']]) };
    const r = parseSlotConfig(s, ctx);
    expect(r.slot).toBeDefined();
  });

  it('52. 5 combos for one action rejected (LIMITS.combosPerAction)', () => {
    const s = fullSlot();
    (s.capabilities as any).queue.keymap = {
      confirm: ['enter'],
      reject: ['r', 'g', 'h', 'j', 'k'],
    };
    expectRejected(parseSlotConfig(s), /combosPerAction|non-empty array/);
  });
});
