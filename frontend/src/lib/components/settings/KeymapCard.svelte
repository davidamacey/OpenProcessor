<script lang="ts">
  /**
   * "Keyboard shortcuts" card — `/settings#keyboard` (K2,
   * docs/design/configurable-keyboard-shortcuts-plan-2026-09-26.md §5.4).
   *
   * Renders one table per keymap context, grouped by the served `group`.
   * Every save first calls `POST /keymap/validate` (debounced) and
   * renders the served report verbatim — this component never computes a
   * collision itself, only the grammar-level checks (max combos per
   * action, a locked key) for immediate feedback while typing.
   *
   * Absent, not disabled, when `keymapAvailability.available !== true` —
   * `/settings/+page.svelte` gates on that, mirroring `ScoresCard`'s own
   * "absent on a pre-route backend" rule.
   *
   * Deviation from the plan's §5.4 sketch, recorded here rather than in a
   * separate doc: verb `group`s rebind together by default (every action
   * sharing a `group` within the SAME context gets the same combos when
   * one member changes), but this pass does not implement the "customize
   * per context" disclosure that lets an operator break a group apart —
   * every group member across every context still updates together. A
   * later pass can add the per-context override without changing this
   * component's save/validate/conflict plumbing.
   */

  import {
    ApiError,
    keymapClassConflictDetail,
    keymapRevisionConflictDetail,
    keymapValidationFailedDetail,
    putKeymap,
    resetKeymap,
    validateKeymap,
    type KeymapClassConflict,
    type KeymapValidationReport,
  } from '$lib/api';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import { formatShortcutKey } from '$lib/keyboardDisplay';
  import { normalize, keyboardStore } from '$stores/keyboard.svelte';
  import { keymapStore } from '$stores/keymap.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import type { KeymapAction } from '$lib/keymapFallback';

  const VALIDATE_DEBOUNCE_MS = 350;

  // Local draft: actionId -> combos, seeded from the current effective
  // document. Only modifiable actions are ever written to this map.
  let draft = $state<Record<string, string[]>>({});
  let report = $state<KeymapValidationReport | null>(null);
  let validating = $state(false);
  let saving = $state(false);
  let capturingActionId = $state<string | null>(null);
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

  const contexts = $derived(keymapStore.document.contexts);
  const grammar = $derived(keymapStore.document.grammar);

  function actionsFor(contextId: string): KeymapAction[] {
    return keymapStore.document.actions.filter((a) => a.context === contextId);
  }

  function groupsFor(contextId: string): (string | null)[] {
    const out: (string | null)[] = [];
    for (const a of actionsFor(contextId)) {
      if (out.includes(a.group)) continue;
      out.push(a.group);
    }
    return out;
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

  function startCapture(actionId: string): void {
    keyboardStore.suspend();
    capturingActionId = actionId;
  }

  function cancelCapture(): void {
    keyboardStore.resume();
    capturingActionId = null;
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
      const detail =
        e instanceof ApiError ? (e.detail ?? e.message) : (e as Error).message;
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
      const detail =
        e instanceof ApiError ? (e.detail ?? e.message) : (e as Error).message;
      toastStore.error(`Reset failed: ${detail}`);
    } finally {
      saving = false;
    }
  }

  const hasChanges = $derived(changedActionIds().length > 0);
</script>

<div id="keyboard" class="rounded-lg border border-zinc-800 bg-zinc-950 p-4">
  <div class="mb-3 flex items-center justify-between">
    <div>
      <h2 class="text-sm font-semibold text-white">Keyboard shortcuts</h2>
      <p class="text-xs text-zinc-500">Applies to every project and every browser.</p>
    </div>
    {#if !keymapStore.isDefault}
      <span
        class="rounded-full border border-amber-800 bg-amber-950/40 px-2 py-0.5 text-[10px] text-amber-300"
      >
        custom keys
      </span>
    {/if}
  </div>

  {#if revisionConflictMessage}
    <div
      class="mb-3 rounded border border-red-800 bg-red-950/40 p-2 text-xs text-red-300"
    >
      {revisionConflictMessage} Shortcuts were changed elsewhere.
      <button class="ml-2 underline" onclick={() => void reloadAfterConflict()}>
        Reload
      </button>
    </div>
  {/if}

  {#if conflictClasses && conflictClasses.length > 0}
    <div
      class="mb-3 rounded border border-amber-800 bg-amber-950/40 p-2 text-xs text-amber-200"
      data-testid="class-conflict-dialog"
    >
      <p class="mb-1">These class hotkeys would collide:</p>
      <ul class="mb-2 list-inside list-disc">
        {#each conflictClasses as c (c.class_id)}
          <li>'{c.combo}' is class '{c.class_name}' ({c.project})</li>
        {/each}
      </ul>
      <div class="flex gap-2">
        <button
          class="rounded bg-amber-700 px-2 py-1 text-white hover:bg-amber-600"
          onclick={() => void doSave(true)}
        >
          Unbind these class keys and save
        </button>
        <button class="text-zinc-400 underline" onclick={() => (conflictClasses = null)}>
          Cancel
        </button>
      </div>
    </div>
  {/if}

  {#each contexts as ctx (ctx.id)}
    <details class="mb-3 rounded border border-zinc-800" open>
      <summary class="cursor-pointer px-3 py-2 text-xs font-semibold text-zinc-300">
        {ctx.label}
        <span class="ml-1 font-normal text-zinc-600">{ctx.description}</span>
      </summary>
      <table class="w-full text-xs">
        <tbody>
          {#each groupsFor(ctx.id) as group (group ?? '__none__')}
            {#each actionsFor(ctx.id).filter((a) => a.group === group) as action (action.id)}
              {@const field = `overrides.${action.id}`}
              {@const fieldIssues = issuesForField(field)}
              <tr class="border-t border-zinc-900">
                <td class="px-3 py-1.5 text-zinc-300">
                  {action.label}
                  {#if !action.available}
                    <span class="ml-1 text-zinc-600"
                      >(no region profile in this project)</span
                    >
                  {/if}
                  {#if fieldIssues.errors.length > 0}
                    <div class="text-red-400">{fieldIssues.errors.join('; ')}</div>
                  {/if}
                  {#if fieldIssues.warnings.length > 0}
                    <div class="text-amber-400">{fieldIssues.warnings.join('; ')}</div>
                  {/if}
                </td>
                <td class="px-3 py-1.5">
                  {#if !action.modifiable}
                    <span
                      class="rounded border border-zinc-700 px-1.5 py-0.5 text-zinc-500"
                    >
                      🔒 {keysOf(action.id).map(formatShortcutKey).join(' / ') || '—'}
                    </span>
                  {:else}
                    <div class="flex flex-wrap items-center gap-1">
                      {#each keysOf(action.id) as k (k)}
                        <span
                          class="flex items-center gap-1 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-blue-300"
                        >
                          {formatShortcutKey(k)}
                          <button
                            class="text-zinc-500 hover:text-red-400"
                            aria-label={`remove ${k}`}
                            onclick={() => removeKey(action.id, k)}
                          >
                            ×
                          </button>
                        </span>
                      {/each}
                      {#if capturingActionId === action.id}
                        <button
                          data-capture
                          use:focusOnMount
                          class="rounded border border-blue-600 bg-blue-950/40 px-1.5 py-0.5 text-blue-300"
                          onkeydown={onCaptureKeydown}
                          onblur={cancelCapture}
                        >
                          press a key…
                        </button>
                      {:else}
                        <button
                          class="rounded border border-zinc-700 px-1.5 py-0.5 text-zinc-400 hover:text-white"
                          onclick={() => startCapture(action.id)}
                        >
                          Change
                        </button>
                      {/if}
                      {#if isChanged(action.id)}
                        <button
                          class="text-zinc-500 underline hover:text-zinc-300"
                          onclick={() => resetOneLocal(action.id)}
                        >
                          reset
                        </button>
                      {/if}
                    </div>
                  {/if}
                </td>
              </tr>
            {/each}
          {/each}
        </tbody>
      </table>
    </details>
  {/each}

  <div class="mt-3 flex items-center gap-3">
    <button
      class="rounded bg-blue-700 px-3 py-1.5 text-sm text-white hover:bg-blue-600 disabled:opacity-50"
      disabled={!hasChanges || saving}
      onclick={openConfirm}
    >
      Save
    </button>
    <button
      class="text-sm text-zinc-400 underline hover:text-zinc-200"
      onclick={() => (resetAllOpen = true)}
    >
      Reset all to defaults
    </button>
    {#if validating}<span class="text-xs text-zinc-500">validating…</span>{/if}
  </div>

  {#if confirmOpen}
    <div
      class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      role="dialog"
      aria-modal="true"
      use:trapFocus={{ onEscape: closeConfirm }}
    >
      <div
        class="w-full max-w-md rounded-lg border border-zinc-700 bg-zinc-950 p-4"
        use:focusOnMount
      >
        <h3 class="mb-2 text-sm font-semibold text-white">Save keyboard shortcuts?</h3>
        <ul class="mb-3 max-h-48 overflow-y-auto text-xs text-zinc-400">
          {#each changedActionIds() as id (id)}
            <li>
              {keymapStore.label(id)}: {keysOf(id).map(formatShortcutKey).join(', ')}
            </li>
          {/each}
        </ul>
        <div class="flex justify-end gap-2">
          <button class="text-sm text-zinc-400" onclick={closeConfirm}>Cancel</button>
          <button
            class="rounded bg-blue-700 px-3 py-1.5 text-sm text-white hover:bg-blue-600"
            onclick={() => void doSave(false)}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  {/if}

  {#if resetAllOpen}
    <div
      class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      role="dialog"
      aria-modal="true"
      use:trapFocus={{ onEscape: () => (resetAllOpen = false) }}
    >
      <div
        class="w-full max-w-md rounded-lg border border-zinc-700 bg-zinc-950 p-4"
        use:focusOnMount
      >
        <h3 class="mb-2 text-sm font-semibold text-white">
          Reset every shortcut to its default?
        </h3>
        <div class="flex justify-end gap-2">
          <button class="text-sm text-zinc-400" onclick={() => (resetAllOpen = false)}>
            Cancel
          </button>
          <button
            class="rounded bg-red-700 px-3 py-1.5 text-sm text-white hover:bg-red-600"
            onclick={() => void doResetAll()}
          >
            Reset all
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>
