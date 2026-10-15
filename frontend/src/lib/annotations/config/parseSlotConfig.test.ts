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
import { parseSlotConfig, parseProfileDocument } from './parseSlotConfig';
import {
  expectRejected,
  fullSlot,
  minimalSlot,
} from '$lib/test/fixtures/slotConfigInputs';

function clone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T;
}
describe('parseSlotConfig — accepts', () => {
  it('1. a minimal slot', () => {
    const r = parseSlotConfig(minimalSlot());
    expect(r.slot).toBeDefined();
    expect(r.slot!.key).toBe('widget_tag');
  });

  it('2. the full slot (superset of the shipped pallet_label example, exercised separately in exampleProfile.test.ts)', () => {
    const r = parseSlotConfig(fullSlot());
    expect(r.slot).toBeDefined();
    expect(r.errors).toEqual([]);
  });

  it('3. an unknown key inside capabilities is accepted with a soft warning', () => {
    const s = minimalSlot();
    (s.capabilities as Record<string, unknown>).notARealCapability = { x: 1 };
    const r = parseSlotConfig(s);
    expect(r.slot).toBeDefined();
    expect(r.errors.some((e) => /unknown key.*ignored/.test(e))).toBe(true);
  });

  it('4. bind carrying both className and classId', () => {
    const s = minimalSlot();
    s.bind = { className: 'widget', classId: 7 };
    const r = parseSlotConfig(s);
    expect(r.slot).toBeDefined();
    expect(r.slot!.bind).toEqual({ className: 'widget', classId: 7 });
  });
});

describe('parseSlotConfig — identity / structure', () => {
  it('5. uppercase key rejected', () => {
    const s = minimalSlot();
    s.key = 'Widget_Tag';
    expectRejected(parseSlotConfig(s), /key/);
  });

  it('6. hyphenated key rejected', () => {
    const s = minimalSlot();
    s.key = 'widget-tag';
    expectRejected(parseSlotConfig(s), /key/);
  });

  it('7. __proto__ key rejected', () => {
    const s = minimalSlot();
    s.key = '__proto__';
    expectRejected(parseSlotConfig(s), /key/);
  });

  it('8. a 65-char key rejected', () => {
    const s = minimalSlot();
    s.key = 'a'.repeat(65);
    expectRejected(parseSlotConfig(s), /key/);
  });

  it('9. empty bind rejected', () => {
    const s = minimalSlot();
    s.bind = {};
    expectRejected(parseSlotConfig(s), /bind/);
  });

  it('10. bind.classId non-integer rejected', () => {
    const s = minimalSlot();
    s.bind = { classId: 3.5 };
    expectRejected(parseSlotConfig(s), /classId/);
  });

  it('11. missing label.title rejected', () => {
    const s = minimalSlot();
    s.label = { singular: 'tag', plural: 'tags' };
    expectRejected(parseSlotConfig(s), /label/);
  });

  it('12. capabilities: null rejected', () => {
    const s = minimalSlot();
    s.capabilities = null;
    expectRejected(parseSlotConfig(s), /capabilities/);
  });
});
describe('parseSlotConfig — endpoints / extras', () => {
  it('58. endpoints.batchStatus with a placeholder rejected (no placeholders allowed)', () => {
    const s = fullSlot();
    (s.endpoints as any).batchStatus = '/x/{cropId}';
    expectRejected(parseSlotConfig(s), /batchStatus/);
  });

  it('59. endpoints.patchMeta with the wrong placeholder rejected; setBox/clearBox are not endpoints', () => {
    const s = fullSlot();
    (s.endpoints as any).patchMeta = '/crops/{size}/x';
    expectRejected(parseSlotConfig(s), /patchMeta/);

    const stale = fullSlot();
    (stale.endpoints as any).setBox = '/crops/{cropId}/x';
    const r = parseSlotConfig(stale);
    expect(r.slot!.endpoints).not.toHaveProperty('setBox');
  });

  it('60. extras: [] rejected; extras with a nested __proto__ rejected', () => {
    const s1 = minimalSlot();
    s1.extras = [];
    expectRejected(parseSlotConfig(s1), /extras/);

    const s2 = minimalSlot();
    // Object-literal `{ __proto__: {} }` syntax SETS a's prototype rather
    // than creating an inspectable own key — use JSON.parse, exactly the
    // real attack surface, so the nested key is an actual own property.
    s2.extras = JSON.parse('{"a":{"__proto__":{}}}');
    expectRejected(parseSlotConfig(s2), /extras/);
  });

  it('61. prototype pollution via JSON.parse leaves the shared Object.prototype untouched', () => {
    const parsed = JSON.parse('{"__proto__":{"polluted":true}}');
    const s = minimalSlot();
    s.extras = parsed;
    const r = parseSlotConfig(s);
    expect(r.slot).toBeUndefined(); // extras itself carries the forbidden key -> rejected
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });
});
describe('parseProfileDocument', () => {
  it('62. unknown version rejects the whole document with one warning', () => {
    const r = parseProfileDocument({ version: 2, slots: [] });
    expect(r.slots).toEqual([]);
    expect(r.warnings).toHaveLength(1);
  });

  it('63. a bare array is accepted', () => {
    const r = parseProfileDocument([minimalSlot()]);
    expect(r.slots).toHaveLength(1);
  });

  it('64. more than LIMITS.slots entries rejects the whole document', () => {
    const slots = Array.from({ length: 33 }, (_, i) => {
      const s = minimalSlot();
      s.key = `widget_tag_${i}`;
      return s;
    });
    const r = parseProfileDocument({ version: 1, slots });
    expect(r.slots).toEqual([]);
    expect(r.warnings).toHaveLength(1);
  });

  it('65. two slots claiming the same urlId — first accepted, second rejected', () => {
    const s1 = fullSlot();
    const s2 = clone(fullSlot());
    s2.key = 'widget_tag_2';
    (s2.capabilities as any).queue.endpointId = 'widget_tags_2';
    // urlId left identical to s1's -> collision.
    const r = parseProfileDocument({ version: 1, slots: [s1, s2] });
    expect(r.slots).toHaveLength(1);
    expect(r.slots[0].key).toBe('widget_tag');
    expect(r.warnings.some((w) => /urlId/.test(w))).toBe(true);
  });

  it('66. one bad slot among three -> two installed, one warning', () => {
    const good1 = minimalSlot();
    const bad = minimalSlot();
    bad.key = 'BAD-KEY';
    const good2 = minimalSlot();
    good2.key = 'widget_tag_2';
    const r = parseProfileDocument({ version: 1, slots: [good1, bad, good2] });
    expect(r.slots).toHaveLength(2);
    expect(r.warnings).toHaveLength(1);
  });

  it('67. null / empty-string / number document never throws, one warning each', () => {
    for (const bad of [null, '', 42]) {
      const r = parseProfileDocument(bad);
      expect(r.slots).toEqual([]);
      expect(r.warnings).toHaveLength(1);
    }
  });
});
