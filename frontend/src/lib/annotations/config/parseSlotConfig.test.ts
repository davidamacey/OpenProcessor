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
import {
  parseSlotConfig,
  parseProfileDocument,
  type ParseContext,
} from './parseSlotConfig';
import { RING_PRESETS } from './allowLists';

function clone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T;
}

/** A minimal, fully valid slot — every required field present, no
 *  optional field set. Mutate a clone of this for the "accepts" cases. */
function minimalSlot(): Record<string, unknown> {
  return {
    key: 'widget_tag',
    bind: { className: 'widget' },
    label: { singular: 'tag', plural: 'tags', title: 'Tag' },
    capabilities: {},
    endpoints: {},
  };
}

/** A fully-loaded valid slot exercising every capability, for the
 *  per-field rejection tests to mutate. */
function fullSlot(): Record<string, unknown> {
  return {
    key: 'widget_tag',
    bind: { className: 'widget' },
    label: { singular: 'tag', plural: 'tags', title: 'Tag' },
    capabilities: {
      subBox: {
        bboxField: 'tag_bbox_norm',
        storedFrame: 'source',
        frameField: 'tag_bbox_frame',
        scoreField: 'tag_score',
        visibleField: 'tag_visible',
        envelope: { aspectMin: 0.5, aspectMax: 2.5, maxWidthFrac: 0.7 },
        thumbnail: {
          path: '/crops/{cropId}/region_thumbnail?size={size}',
          aspect: '1 / 1',
          defaultSize: 160,
        },
        ring: 'default',
        editor: { thumbSize: 512, viewPadding: 2.0, nudgeStep: 0.001953125 },
      },
      text: {
        valueField: 'tag_text',
        rawField: 'tag_text_raw',
        sourceField: 'tag_text_source',
        confidenceField: 'tag_text_confidence',
        label: 'Tag text',
        placeholder: '000123',
        transform: 'uppercase',
        monospace: true,
        pattern: { source: '^[0-9]{6}$', flags: '' },
        maxLength: 6,
      },
      provenance: {
        detectorField: 'tag_detector',
        detectorVersionField: 'tag_detector_version',
        chainField: 'tag_detector_chain',
        showChainOnCard: true,
      },
      lifecycle: {
        statusField: 'tag_status',
        verifiedField: 'tag_verified',
        states: [
          { value: 'detected', label: 'detected', humanWritable: true, role: 'proposed' },
          {
            value: 'no_tag',
            label: 'no tag visible',
            humanWritable: true,
            role: 'absent',
          },
          {
            value: 'false_positive',
            label: 'false positive',
            humanWritable: true,
            role: 'falsePositive',
            dim: true,
            badge: 'false pos',
          },
        ],
        confirmState: 'detected',
        rejectState: 'no_tag',
        falsePositiveState: 'false_positive',
      },
      queue: {
        endpointId: 'widget_tags',
        urlId: 'widget_tags',
        tabLabel: 'Widget tags',
        browsePath: '/widget_tags',
        keymap: {
          confirm: ['enter'],
          reject: ['r'],
          markFalsePositive: ['f'],
          editBox: ['e'],
          back: ['arrowleft'],
        },
        textFilter: { param: 'text', label: 'Tag', placeholder: 'e.g. 000123' },
        alwaysVisible: true,
      },
      trainingCohorts: {
        cohorts: [
          {
            id: 'validated_tags',
            label: 'Validated tags',
            description: 'Human-confirmed tags',
            query: {
              kind: 'endpoint',
              path: '/plates/training_candidates',
              params: { class_id: '{classId}' },
            },
            rowKind: 'slot',
            reviewTarget: 'slotQueue',
          },
        ],
      },
    },
    endpoints: {
      setBox: '/crops/{cropId}/tag',
      clearBox: '/crops/{cropId}/tag',
      patchMeta: '/crops/{cropId}/tag_meta',
      batchStatus: '/widget_tags/batch_status',
    },
    stats: {
      key: 'widget_tags',
      panelTitle: 'Widget tags',
      coverageTitle: 'Widget tag coverage',
    },
  };
}

function expectRejected(result: ReturnType<typeof parseSlotConfig>, match: RegExp) {
  expect(result.slot).toBeUndefined();
  expect(result.errors.some((e) => match.test(e))).toBe(true);
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

describe('parseSlotConfig — subBox', () => {
  it('13. missing bboxField rejected', () => {
    const s = fullSlot();
    delete (s.capabilities as any).subBox.bboxField;
    expectRejected(parseSlotConfig(s), /bboxField/);
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

  it('24. envelope.aspectMin > aspectMax rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).subBox.envelope = { aspectMin: 5, aspectMax: 2 };
    expectRejected(parseSlotConfig(s), /envelope/);
  });
});

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

describe('parseSlotConfig — cohorts / endpoints / extras', () => {
  it('53. cohort query.path with {classId} accepted', () => {
    const s = fullSlot();
    const r = parseSlotConfig(s);
    expect(r.slot).toBeDefined();
    expect(r.slot!.capabilities.trainingCohorts!.cohorts[0].query).toMatchObject({
      kind: 'endpoint',
      path: '/plates/training_candidates',
    });
  });

  it('54. cohort query.path using {cropId} rejected (wrong placeholder set)', () => {
    const s = fullSlot();
    (s.capabilities as any).trainingCohorts.cohorts[0].query.path =
      '/plates/training_candidates/{cropId}';
    expectRejected(parseSlotConfig(s), /query\.path/);
  });

  it('55. cohort params key with a space rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).trainingCohorts.cohorts[0].query.params = { 'class id': 1 };
    expectRejected(parseSlotConfig(s), /params key/);
  });

  it('56. cohort params value with ";" rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).trainingCohorts.cohorts[0].query.params = { mode: 'a;b' };
    expectRejected(parseSlotConfig(s), /params\.mode/);
  });

  it('57. cohort query.kind: "sql" rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).trainingCohorts.cohorts[0].query = { kind: 'sql' };
    expectRejected(parseSlotConfig(s), /query\.kind/);
  });

  it('58. endpoints.batchStatus with a placeholder rejected (no placeholders allowed)', () => {
    const s = fullSlot();
    (s.endpoints as any).batchStatus = '/x/{cropId}';
    expectRejected(parseSlotConfig(s), /batchStatus/);
  });

  it('59. endpoints.setBox with the wrong placeholder rejected', () => {
    const s = fullSlot();
    (s.endpoints as any).setBox = '/crops/{size}/x';
    expectRejected(parseSlotConfig(s), /setBox/);
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

describe('parseProfileDocument — envelope', () => {
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
