/**
 * Closed vocabularies the tier-2 deployment-profile parser validates
 * against (docs/annotation-slots-contract-draft.md §5, and this repo's
 * docs/design/tier2-annotation-profile-config-plan-2026-09-20.md §2.2).
 *
 * `static/annotation-profiles.json` is supplied by a deployment
 * OPERATOR, not by a reviewed pull request. It is untrusted input at a
 * system boundary. Every list here is an allow-list, never a deny-list:
 * anything not named here is rejected, so a future JSON key can never
 * fail open.
 *
 * This module is intentionally free of imports from application code so
 * it can never break anything by existing.
 */

import type { SubBoxRing } from '../types';

/** Placeholders legal in a `{API_PREFIX}`-relative path template.
 *  Mirrors `cohorts.ts:22-25`'s rule: "no arbitrary expressions, no
 *  field access." */
export const PATH_PLACEHOLDERS: readonly string[] = ['cropId', 'size'];

/** Placeholders legal in a training-cohort endpoint query. Same closed
 *  set `compileCohortQuery` (`../cohorts.ts:253-261`) substitutes. */
export const COHORT_PLACEHOLDERS: readonly string[] = ['classId', 'slotKey'];

/**
 * Named ring presets a JSON config may reference by name.
 *
 * These strings are written LITERALLY here, in a `.ts` file under
 * `src/`, because Tailwind's JIT statically scans source for class
 * names. A class string that exists only inside a runtime-mounted JSON
 * file is invisible to that scan and renders unstyled — see
 * `types.ts:69-71` and contract §5.4.
 */
export const RING_PRESETS: Readonly<Record<string, SubBoxRing>> = {
  default: {
    confirmed: 'border-green-400 shadow-[0_0_0_1px_rgba(34,197,94,0.45)]',
    proposed: 'border-yellow-400 shadow-[0_0_0_1px_rgba(250,204,21,0.45)]',
    rejected: 'border-zinc-600 shadow-none',
  },
  neutral: {
    confirmed: 'border-blue-400 shadow-[0_0_0_1px_rgba(96,165,250,0.45)]',
    proposed: 'border-zinc-500 shadow-none',
    rejected: 'border-zinc-700 shadow-none',
  },
};

/** Every literal ring class string a config may use directly. Derived
 *  from the presets so the two can never drift. */
export const RING_CLASS_ALLOWLIST: ReadonlySet<string> = new Set(
  Object.values(RING_PRESETS).flatMap((r) => [r.confirmed, r.proposed, r.rejected]),
);

/** Key combos a slot's `QueueCapability.keymap` may name. No modifier
 *  combos: `keyboardStore` dispatches on a bare normalized key. */
export const KEY_COMBO_VOCABULARY: ReadonlySet<string> = new Set([
  ...'abcdefghijklmnopqrstuvwxyz0123456789'.split(''),
  'arrowleft',
  'arrowright',
  'arrowup',
  'arrowdown',
  'enter',
  'escape',
  'space',
  'backspace',
]);

/**
 * Combos a DEPLOYMENT slot may never claim (contract §5.5).
 *
 * `n` (skip) and `z` (undo last) are registered by `/review`
 * unconditionally, INCLUDING while a slot tab is active — see
 * `src/routes/review/+page.svelte:1165-1166`, outside the
 * `if (activeSlot)` branch. Claiming one registers a second handler for
 * the same key.
 *
 * `enter` and `arrowright` are emitted unconditionally by
 * `buildSlotKeymap` (`../../review/slotKeymap.ts:65,91`); re-declaring
 * either would double-register. `escape` is edit-mode cancel.
 *
 * `confirm` is handled separately: `buildSlotKeymap` hardcodes it to
 * `enter` and never reads `keymap.confirm`, so a config declaring
 * anything other than exactly `['enter']` is a silent no-op and is
 * rejected rather than accepted-and-ignored.
 */
export const FORBIDDEN_SLOT_COMBOS: ReadonlySet<string> = new Set([
  'n',
  'z',
  'escape',
  'enter',
  'arrowright',
]);

/** `SlotAction` values a config may key its keymap by. Must stay in
 *  sync with `SlotAction` in `../types.ts:195-202`; `slotKeymap.test.ts`
 *  ratchets the pair. */
export const KEYMAP_ACTIONS: readonly string[] = [
  'confirm',
  'reject',
  'markFalsePositive',
  'editBox',
  'back',
  'skip',
  'undo',
];

/** Regex flags a `TextCapability.pattern` may carry. `g`/`y` are
 *  banned: a stateful `lastIndex` makes `.test()` alternate true/false
 *  on identical input. */
export const REGEX_FLAG_ALLOWLIST: ReadonlySet<string> = new Set(['i', 'm', 's', 'u']);

/** Hard caps. Every one of these is a "a broken config must not wedge
 *  the app" bound, not a design limit anyone should hit. */
export const LIMITS = {
  documentBytes: 256 * 1024,
  slots: 32,
  lifecycleStates: 32,
  vocabularyEntries: 256,
  cohorts: 32,
  cohortParams: 16,
  combosPerAction: 4,
  stateAliases: 8,
  identifierChars: 64,
  labelChars: 120,
  pathChars: 200,
  patternSourceChars: 200,
} as const;

/** Slot keys, endpoint ids, url ids, stats keys, wire-field names and
 *  cohort ids. Lands in URLs, storage keys and `slot:${key}` tab ids, so
 *  it is deliberately narrow. */
export const IDENTIFIER_RE = /^[a-z][a-z0-9_]*$/;

/** Wire field names may also be camelCase — the frontend reads whatever
 *  the backend emits (`types.ts:38-47`) and must not dictate casing. */
export const WIRE_FIELD_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

/** Keys that are never legal anywhere in the document. */
export const FORBIDDEN_KEYS: ReadonlySet<string> = new Set([
  '__proto__',
  'constructor',
  'prototype',
]);
