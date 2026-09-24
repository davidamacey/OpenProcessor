/**
 * The tier-2 integration gate — the shipped
 * `static/annotation-profiles.example.json` proven end to end through
 * every consumer, the way `secondSlotIntegration.test.ts` proves the
 * capability model end to end for a TypeScript-declared second slot.
 * Reads the file off disk (the shipped example can never rot into an
 * invalid file without CI noticing) and never registers it into the
 * real app registry.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it, vi } from 'vitest';
import { parseProfileDocument } from './parseSlotConfig';
import { resolveSlotRegistry } from '../registry';
import { builtinSlots } from '../registeredSlots';
import { buildReviewTabs } from '../../reviewTabs';
import { buildSlotKeymap } from '../../review/slotKeymap';
import { reservedHotkeyLetters } from '../../classHotkey';
import { cohortsForClass } from '../cohorts';
import { readSlot } from '../readSlot';
import { slotIsPresent } from '../types';

const here = path.dirname(fileURLToPath(import.meta.url));
const examplePath = path.resolve(
  here,
  '../../../../static/annotation-profiles.example.json',
);
const doc = JSON.parse(readFileSync(examplePath, 'utf8'));

describe('the shipped annotation-profiles.example.json — integration gate', () => {
  const { slots, warnings } = parseProfileDocument(doc);

  it('1. parses cleanly: zero warnings, one slot', () => {
    expect(warnings).toEqual([]);
    expect(slots).toHaveLength(1);
  });

  const { registry, warnings: mergeWarnings } = resolveSlotRegistry({
    builtins: builtinSlots,
    deployment: slots,
  });

  it('2. merges over the built-in without disturbing it (identity, not equality)', () => {
    expect(mergeWarnings).toEqual([]);
    expect(registry.all).toHaveLength(builtinSlots.length + 1);
    for (const builtin of builtinSlots) {
      expect(registry.byKey(builtin.key)).toBe(builtin);
    }
    expect(registry.byKey('pallet_label')).toBeDefined();
  });

  const palletSlot = registry.byKey('pallet_label')!;

  it('3. forClass resolves the pallet slot case-insensitively', () => {
    const found = registry.forClass(42, new Map([[42, 'Wooden_Pallet']]));
    expect(found).toContain(palletSlot);
  });

  const tabs = buildReviewTabs(registry.all);

  it('4. buildReviewTabs produces two tabs, the new one shaped correctly', () => {
    expect(tabs).toHaveLength(
      builtinSlots.filter((s) => s.capabilities.queue).length + 1,
    );
    const palletTab = tabs.find((t) => t.id === 'slot:pallet_label');
    expect(palletTab).toBeDefined();
    expect(palletTab!.urlId).toBe('pallet_labels');
    expect(palletTab!.endpointId).toBe('pallet_labels');
    expect(palletTab!.label).toBe('Pallet labels');
  });

  const handlers = {
    confirm: vi.fn(),
    reject: vi.fn(),
    markFalsePositive: vi.fn(),
    toggleEdit: vi.fn(),
    back: vi.fn(),
    advance: vi.fn(),
    saveAndExit: vi.fn(),
  };

  it('5. scan-mode keymap is exactly enter/r/f/e/arrowleft/b/arrowright', () => {
    const entries = buildSlotKeymap(palletSlot, false, handlers);
    expect(entries.map((e) => e.combo)).toEqual([
      'enter',
      'r',
      'f',
      'e',
      'arrowleft',
      'b',
      'arrowright',
    ]);
  });

  it('6. edit-mode keymap is exactly enter/escape', () => {
    const entries = buildSlotKeymap(palletSlot, true, handlers);
    expect(entries.map((e) => e.combo)).toEqual(['enter', 'escape']);
  });

  it('7. reservedHotkeyLetters grows to cover r/f/e/b/d with zero code change', () => {
    const reserved = reservedHotkeyLetters(registry);
    for (const letter of ['r', 'f', 'e', 'b', 'd']) {
      expect(reserved.has(letter)).toBe(true);
    }
  });

  it('8. cohortsForClass includes the declared cohort with a compiled query', () => {
    const classesById = new Map([[42, 'wooden_pallet']]);
    const cohorts = cohortsForClass(42, 'wooden_pallet', registry, classesById, false);
    const validated = cohorts.find((c) => c.id === 'validated_labels');
    expect(validated).toBeDefined();
    expect(validated!.query).toMatchObject({
      kind: 'endpoint',
      params: { class_id: '42' },
    });
    // The 4 CORE_COHORTS are still present alongside the slot's own.
    expect(cohorts.length).toBeGreaterThanOrEqual(5);
  });

  it('9. readSlot() renders the SlotCard-consumed data shape correctly', () => {
    const raw = {
      label_bbox_norm: [0.4, 0.4, 0.52, 0.52],
      label_bbox_frame: 'source',
      label_score: 0.81,
      label_visible: true,
      label_text: '000123456700000000',
      label_text_source: 'gemma',
      label_detector: 'sam3',
      label_detector_chain: ['tag_detector_v1:miss', 'sam3:hit'],
      label_status: 'false_positive',
      label_verified: false,
    };
    const d = readSlot(raw, palletSlot, [0.2, 0.2, 0.8, 0.8]);
    expect(d.subBox!.parent!.w).toBeCloseTo(0.2, 5);
    expect(d.text!.value).toBe('000123456700000000');
    expect(d.provenance!.chain).toHaveLength(2);
    expect(d.lifecycle!.state!.badge).toBe('false pos');
    expect(d.lifecycle!.state!.dim).toBe(true);
    expect(slotIsPresent(d)).toBe(true);
    expect(palletSlot.capabilities.lifecycle!.falsePositiveState).toBe('false_positive');
    expect(palletSlot.capabilities.queue!.textFilter!.label).toBe('SSCC');
  });

  it('10. the ship gate: the example domain never appears in non-test production source', () => {
    const srcRoot = path.resolve(here, '../../../../src');
    function walk(dir: string, out: string[] = []): string[] {
      for (const entry of readdirSync(dir)) {
        const full = path.join(dir, entry);
        if (statSync(full).isDirectory()) walk(full, out);
        else out.push(full);
      }
      return out;
    }
    const files = walk(srcRoot).filter(
      (f) => (f.endsWith('.ts') || f.endsWith('.svelte')) && !f.endsWith('.test.ts'),
    );
    const offenders: string[] = [];
    for (const f of files) {
      const content = readFileSync(f, 'utf-8').toLowerCase();
      if (content.includes('pallet')) offenders.push(path.relative(srcRoot, f));
    }
    expect(offenders).toEqual([]);
  });
});
