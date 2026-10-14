/**
 * Draft, validation, capture and save state for the "Keyboard shortcuts"
 * card. Every save first calls `POST /keymap/validate` (debounced) and the
 * served report is rendered verbatim; only grammar-level checks (max combos
 * per action, a locked key) run here for immediate feedback while typing.
 */

import {
  apiErrorText,
  keymapClassConflictDetail,
  keymapRevisionConflictDetail,
  keymapValidationFailedDetail,
  putKeymap,
  resetKeymap,
  validateKeymap,
  type KeymapClassConflict,
  type KeymapValidationReport,
} from '$lib/api';
import { formatShortcutKey } from '$lib/keyboardDisplay';
import { normalize, keyboardStore } from '$stores/keyboard.svelte';
import { keymapStore } from '$stores/keymap.svelte';
import { toastStore } from '$stores/toast.svelte';
import { groupMembers } from '$components/settings/keymap/keymapModel';

export function createKeymapCardController() {
  const VALIDATE_DEBOUNCE_MS = 350;

  // Local draft: actionId -> combos, seeded from the current effective
  // document. Only modifiable actions are ever written to this map.
  let draft = $state<Record<string, string[]>>({});
  let report = $state<KeymapValidationReport | null>(null);
  let validating = $state(false);
  let saving = $state(false);
  let capturingActionId = $state<string | null>(null);
  let capturingGroupId = $state<string | null>(null);
  let confirmOpen = $state(false);
  let conflictClasses = $state<KeymapClassConflict[] | null>(null);
  let revisionConflictMessage = $state<string | null>(null);

  function seedDraft(): void {
    const d: Record<string, string[]> = {};
    for (const a of keymapStore.document.actions) {
      if (a.modifiable) d[a.id] = [...a.keys];
    }
    draft = d;
  }
  seedDraft();

  const grammar = $derived(keymapStore.document.grammar);

  /** The most common draft value among a group's members (ties: first). */
  function groupCanonicalKeys(groupId: string): string[] {
    const members = groupMembers(groupId);
    if (members.length === 0) return [];
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local tally map consumed synchronously within this function, never stored in reactive state
    const counts = new Map<string, number>();
    for (const m of members) {
      const json = JSON.stringify(keysOf(m.id));
      counts.set(json, (counts.get(json) ?? 0) + 1);
    }
    let bestJson = JSON.stringify(keysOf(members[0].id));
    let bestCount = 0;
    for (const [json, count] of counts) {
      if (count > bestCount) {
        bestCount = count;
        bestJson = json;
      }
    }
    return JSON.parse(bestJson) as string[];
  }

  /** Computed, not stored: does this member's draft differ from the group? */
  function memberDetached(actionId: string): boolean {
    const a = keymapStore.action(actionId);
    if (!a?.group) return false;
    return (
      JSON.stringify(keysOf(actionId)) !== JSON.stringify(groupCanonicalKeys(a.group))
    );
  }

  function keysOf(actionId: string): string[] {
    return draft[actionId] ?? keymapStore.action(actionId)?.keys ?? [];
  }

  function isChanged(actionId: string): boolean {
    const a = keymapStore.action(actionId);
    if (!a) return false;
    const cur = keysOf(actionId);
    return cur.length !== a.default.length || cur.some((k, i) => k !== a.default[i]);
  }

  function changedActionIds(): string[] {
    return Object.keys(draft).filter((id) => {
      const original = keymapStore.action(id)?.keys ?? [];
      const cur = draft[id];
      return original.length !== cur.length || cur.some((k, i) => k !== original[i]);
    });
  }

  let validateTimer: ReturnType<typeof setTimeout> | null = null;

  function scheduleValidate(): void {
    if (validateTimer) clearTimeout(validateTimer);
    validateTimer = setTimeout(runValidate, VALIDATE_DEBOUNCE_MS);
  }

  async function runValidate(): Promise<void> {
    validating = true;
    try {
      report = await validateKeymap({ overrides: overridesFromDraft() });
    } catch {
      // The validate endpoint is documented to never 4xx/5xx meaningfully
      // (plan §4.3: "never 409/422; a report") — a transient network
      // failure here just means no live feedback until the next edit or
      // Save, which still validates server-side.
      report = null;
    } finally {
      validating = false;
    }
  }

  function overridesFromDraft(): Record<string, string[]> {
    const out: Record<string, string[]> = {};
    for (const [id, keys] of Object.entries(draft)) {
      const a = keymapStore.action(id);
      if (!a) continue;
      const same =
        keys.length === a.default.length && keys.every((k, i) => k === a.default[i]);
      if (!same) out[id] = keys;
    }
    return out;
  }

  function issuesForField(field: string): { errors: string[]; warnings: string[] } {
    const errors = (report?.errors ?? [])
      .filter((i) => i.field === field)
      .map((i) => i.message);
    const warnings = (report?.warnings ?? [])
      .filter((i) => i.field === field)
      .map((i) => i.message);
    return { errors, warnings };
  }

  function removeKey(actionId: string, key: string): void {
    draft[actionId] = keysOf(actionId).filter((k) => k !== key);
    draft = { ...draft };
    scheduleValidate();
  }

  function resetOneLocal(actionId: string): void {
    const a = keymapStore.action(actionId);
    if (!a) return;
    draft[actionId] = [...a.default];
    draft = { ...draft };
    scheduleValidate();
  }

  /** Reset a detached member back to its group's current shared value. */
  function resetMemberToGroup(actionId: string): void {
    const a = keymapStore.action(actionId);
    if (!a?.group) return;
    draft[actionId] = [...groupCanonicalKeys(a.group)];
    draft = { ...draft };
    scheduleValidate();
  }

  /** Remove a key from every member of a verb group at once. */
  function removeGroupKey(groupId: string, key: string): void {
    for (const m of groupMembers(groupId)) {
      draft[m.id] = keysOf(m.id).filter((k) => k !== key);
    }
    draft = { ...draft };
    scheduleValidate();
  }

  function startCapture(actionId: string): void {
    keyboardStore.suspend();
    capturingActionId = actionId;
    capturingGroupId = null;
  }

  function startCaptureGroup(groupId: string): void {
    keyboardStore.suspend();
    capturingGroupId = groupId;
    capturingActionId = null;
  }

  function cancelCapture(): void {
    keyboardStore.resume();
    capturingActionId = null;
    capturingGroupId = null;
  }

  function onCaptureKeydown(e: KeyboardEvent): void {
    e.preventDefault();
    e.stopPropagation();
    if (e.key === 'Escape') {
      cancelCapture();
      return;
    }
    // Ignore a bare modifier press — it isn't a combo by itself.
    if (['Control', 'Meta', 'Alt', 'Shift'].includes(e.key)) return;
    const combo = normalize(e);

    if (capturingGroupId) {
      captureForGroup(capturingGroupId, combo);
      return;
    }

    const actionId = capturingActionId;
    if (!actionId) return;
    if (grammar.locked_keys.includes(combo)) {
      const a = keymapStore.action(actionId);
      if (!a?.locked_keys?.includes(combo)) {
        toastStore.error(
          `'${formatShortcutKey(combo)}' is locked and can't be reassigned.`,
        );
        cancelCapture();
        return;
      }
    }
    const cur = keysOf(actionId);
    if (cur.includes(combo)) {
      cancelCapture();
      return;
    }
    if (cur.length >= grammar.max_combos_per_action) {
      toastStore.error(`Up to ${grammar.max_combos_per_action} keys per action.`);
      cancelCapture();
      return;
    }
    draft[actionId] = [...cur, combo];
    draft = { ...draft };
    cancelCapture();
    scheduleValidate();
  }

  /**
   * Apply a captured combo to every member of a verb group at once.
   * Locked-key combos are always rejected at group level, conservatively
   * — a group can span members whose locked-key sets differ, and this
   * component never leaves a locked key unlocked.
   */
  function captureForGroup(groupId: string, combo: string): void {
    if (grammar.locked_keys.includes(combo)) {
      toastStore.error(
        `'${formatShortcutKey(combo)}' is locked and can't be reassigned.`,
      );
      cancelCapture();
      return;
    }
    const members = groupMembers(groupId);
    for (const m of members) {
      const cur = keysOf(m.id);
      if (cur.includes(combo)) continue;
      if (cur.length >= grammar.max_combos_per_action) continue;
      draft[m.id] = [...cur, combo];
    }
    draft = { ...draft };
    cancelCapture();
    scheduleValidate();
  }

  function openConfirm(): void {
    confirmOpen = true;
  }

  function closeConfirm(): void {
    confirmOpen = false;
  }

  async function doSave(unbind = false): Promise<void> {
    const revision = keymapStore.revision;
    if (revision == null) return;
    saving = true;
    try {
      const doc = await putKeymap({
        expected_revision: revision,
        overrides: overridesFromDraft(),
        unbind_conflicting_class_hotkeys: unbind,
      });
      keymapStore.setDocument(doc, 'served');
      seedDraft();
      report = null;
      confirmOpen = false;
      conflictClasses = null;
      revisionConflictMessage = null;
      toastStore.success('Keyboard shortcuts saved.');
    } catch (e) {
      const revConflict = keymapRevisionConflictDetail(e);
      if (revConflict) {
        revisionConflictMessage = revConflict.message;
        return;
      }
      const classConflict = keymapClassConflictDetail(e);
      if (classConflict) {
        conflictClasses = classConflict.class_conflicts;
        return;
      }
      const validationFailed = keymapValidationFailedDetail(e);
      if (validationFailed) {
        report = validationFailed.report;
        confirmOpen = false;
        toastStore.error(validationFailed.message);
        return;
      }
      const detail = apiErrorText(e);
      toastStore.error(`Save failed: ${detail}`);
    } finally {
      saving = false;
    }
  }

  async function reloadAfterConflict(): Promise<void> {
    const { loadKeymap } = await import('$stores/keymap.svelte');
    await loadKeymap();
    seedDraft();
    revisionConflictMessage = null;
  }

  let resetAllOpen = $state(false);

  async function doResetAll(): Promise<void> {
    const revision = keymapStore.revision;
    if (revision == null) return;
    saving = true;
    try {
      const doc = await resetKeymap({ expected_revision: revision, action_ids: null });
      keymapStore.setDocument(doc, 'served');
      seedDraft();
      report = null;
      resetAllOpen = false;
      toastStore.success('Keyboard shortcuts reset to defaults.');
    } catch (e) {
      const detail = apiErrorText(e);
      toastStore.error(`Reset failed: ${detail}`);
    } finally {
      saving = false;
    }
  }

  const hasChanges = $derived(changedActionIds().length > 0);

  return {
    get validating() {
      return validating;
    },
    get saving() {
      return saving;
    },
    get capturingActionId() {
      return capturingActionId;
    },
    get capturingGroupId() {
      return capturingGroupId;
    },
    get confirmOpen() {
      return confirmOpen;
    },
    get conflictClasses() {
      return conflictClasses;
    },
    set conflictClasses(v: KeymapClassConflict[] | null) {
      conflictClasses = v;
    },
    get revisionConflictMessage() {
      return revisionConflictMessage;
    },
    get resetAllOpen() {
      return resetAllOpen;
    },
    set resetAllOpen(v: boolean) {
      resetAllOpen = v;
    },
    get hasChanges() {
      return hasChanges;
    },
    keysOf,
    isChanged,
    changedActionIds,
    groupCanonicalKeys,
    memberDetached,
    issuesForField,
    removeKey,
    resetOneLocal,
    resetMemberToGroup,
    removeGroupKey,
    startCapture,
    startCaptureGroup,
    cancelCapture,
    onCaptureKeydown,
    openConfirm,
    closeConfirm,
    doSave,
    reloadAfterConflict,
    doResetAll,
  };
}

export type KeymapCardController = ReturnType<typeof createKeymapCardController>;
