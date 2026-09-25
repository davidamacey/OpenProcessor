/**
 * The schema proof (contract §9 step 2 / this repo's tier-2 plan §5.3):
 * a JSON-serialized version of the hand-written `aircraftTailNumberSlot`
 * must reconstitute a spec that is structurally and behaviorally
 * identical to the TypeScript original, with NO schema change. If it
 * cannot, the published §4 schema is wrong and this fixture is the bug
 * report — the fix belongs in the schema/parser and a new caveat in
 * docs/annotation-slots-contract-draft.md §5, never in loosening this
 * test's assertions.
 */

import { describe, expect, it, vi } from 'vitest';
import { parseSlotConfig } from './parseSlotConfig';
import { aircraftTailNumberSlot } from '$lib/test/fixtures/aircraftTailNumberSlot';
import { derivedCohorts } from '../cohorts';
import { buildSlotKeymap } from '../../review/slotKeymap';
import type { SlotSpec } from '../types';
import { readExampleDocument } from '$lib/test/fixtures/exampleProfiles';

// The operator-facing example document; its one slot is the fixture.
const fixture = (readExampleDocument('aircraft-tail-number.json') as { slots: unknown[] })
  .slots[0];

describe('round-trip: JSON aircraftTailNumber profile === hand-written TypeScript', () => {
  const result = parseSlotConfig(fixture);

  it('parses cleanly with no errors', () => {
    expect(result.errors).toEqual([]);
    expect(result.slot).toBeDefined();
  });

  const parsed = result.slot as SlotSpec;

  it('every field JSON can express directly matches wholesale', () => {
    const strip = (s: SlotSpec) => ({
      ...s,
      capabilities: {
        ...s.capabilities,
        subBox: s.capabilities.subBox && {
          ...s.capabilities.subBox,
          thumbnail: undefined, // function — compared below
        },
        text: s.capabilities.text && {
          ...s.capabilities.text,
          pattern: undefined, // RegExp — compared below
        },
      },
      endpoints: {}, // functions — compared below
    });
    expect(strip(parsed)).toEqual(strip(aircraftTailNumberSlot));
  });

  it('compiled path closures are byte-identical, including encodeURIComponent on a hostile id', () => {
    const id = 'a/b?c';
    expect(parsed.endpoints.setBox!(id)).toBe(
      aircraftTailNumberSlot.endpoints.setBox!(id),
    );
    expect(parsed.endpoints.patchMeta!(id)).toBe(
      aircraftTailNumberSlot.endpoints.patchMeta!(id),
    );
    expect(parsed.capabilities.subBox!.thumbnail!.path(id, 192)).toBe(
      aircraftTailNumberSlot.capabilities.subBox!.thumbnail!.path(id, 192),
    );
    expect(parsed.endpoints.setBox!(id)).toBe('/crops/a%2Fb%3Fc/tail');
  });

  it('compiled RegExp matches the hand-written one', () => {
    expect(parsed.capabilities.text!.pattern!.source).toBe(
      aircraftTailNumberSlot.capabilities.text!.pattern!.source,
    );
    expect(parsed.capabilities.text!.pattern!.flags).toBe('');
    expect(parsed.capabilities.text!.pattern!.test('N123AB')).toBe(true);
    expect(parsed.capabilities.text!.pattern!.test('123')).toBe(false);
  });

  it('the derived surface (cohorts, keymap) is behaviorally identical', () => {
    expect(
      derivedCohorts(parsed)
        .map((c) => c.id)
        .sort(),
    ).toEqual(
      derivedCohorts(aircraftTailNumberSlot)
        .map((c) => c.id)
        .sort(),
    );
    const handlers = {
      confirm: vi.fn(),
      reject: vi.fn(),
      toggleEdit: vi.fn(),
      back: vi.fn(),
      advance: vi.fn(),
      saveAndExit: vi.fn(),
    };
    expect(buildSlotKeymap(parsed, false, handlers).map((e) => e.combo)).toEqual(
      buildSlotKeymap(aircraftTailNumberSlot, false, handlers).map((e) => e.combo),
    );
  });
});
