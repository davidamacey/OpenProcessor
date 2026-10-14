<script lang="ts">
  /**
   * "Keyboard shortcuts" card — `/settings#keyboard` (K2 + K2b,
   * docs/design/configurable-keyboard-shortcuts-plan-2026-09-26.md §5.4).
   *
   * Two sections:
   *  - **Verb groups** (K2b) — one row per served `group` id shared by
   *    modifiable actions across contexts (e.g. `undo` spans `review`,
   *    `cluster`, `clusters_search`, `region_gallery`). Editing the group
   *    row writes every member action id at once — "rebind Undo" changes
   *    it everywhere. A "Customize per page" disclosure lists each
   *    member's own action under its context label with its own key
   *    chips; editing one there only writes that action id, detaching it
   *    from the group. Detachment is never a stored flag — it's computed
   *    each render as "does this member's draft differ from the other
   *    members' shared value", so it also surfaces a pre-existing
   *    server-side per-context override with no extra bookkeeping.
   *  - **Per-context tables** (K2, unchanged) — every ungrouped action,
   *    plus every locked/non-modifiable action regardless of group (the
   *    `cancel` group is entirely non-modifiable and never appears in the
   *    Verb groups section at all).
   *
   * Every save first calls `POST /keymap/validate` (debounced) and
   * renders the served report verbatim — this component never computes a
   * collision itself, only the grammar-level checks (max combos per
   * action, a locked key) for immediate feedback while typing.
   *
   * Absent, not disabled, when `keymapAvailability.available !== true` —
   * `/settings/+page.svelte` gates on that, mirroring `ScoresCard`'s own
   * "absent on a pre-route backend" rule.
   */

  import { keymapStore } from '$stores/keymap.svelte';
  import KeymapGroupsSection from '$components/settings/keymap/KeymapGroupsSection.svelte';
  import KeymapContextsSection from '$components/settings/keymap/KeymapContextsSection.svelte';
  import KeymapConfirmModals from '$components/settings/keymap/KeymapConfirmModals.svelte';
  import { createKeymapCardController } from '$components/settings/keymap/keymapCardController.svelte';

  const ctl = createKeymapCardController();
</script>

<div id="keyboard" class="rounded-lg border border-zinc-800 bg-zinc-950 p-4">
  <div class="mb-3 flex items-center justify-between">
    <div>
      <h2 class="text-sm font-semibold text-white">Keyboard shortcuts</h2>
      <p class="text-xs text-zinc-500">Applies to this project, in every browser.</p>
    </div>
    {#if !keymapStore.isDefault}
      <span
        class="rounded-full border border-amber-800 bg-amber-950/40 px-2 py-0.5 text-[10px] text-amber-300"
      >
        custom keys
      </span>
    {/if}
  </div>

  {#if ctl.revisionConflictMessage}
    <div
      class="mb-3 rounded border border-red-800 bg-red-950/40 p-2 text-xs text-red-300"
    >
      {ctl.revisionConflictMessage} Shortcuts were changed elsewhere.
      <button class="ml-2 underline" onclick={() => void ctl.reloadAfterConflict()}>
        Reload
      </button>
    </div>
  {/if}

  {#if ctl.conflictClasses && ctl.conflictClasses.length > 0}
    <div
      class="mb-3 rounded border border-amber-800 bg-amber-950/40 p-2 text-xs text-amber-200"
      data-testid="class-conflict-dialog"
    >
      <p class="mb-1">These class hotkeys would collide:</p>
      <ul class="mb-2 list-inside list-disc">
        {#each ctl.conflictClasses as c (c.class_id)}
          <li>'{c.combo}' is class '{c.class_name}' ({c.project})</li>
        {/each}
      </ul>
      <div class="flex gap-2">
        <button
          class="rounded bg-amber-700 px-2 py-1 text-white hover:bg-amber-600"
          onclick={() => void ctl.doSave(true)}
        >
          Unbind these class keys and save
        </button>
        <button
          class="text-zinc-400 underline"
          onclick={() => (ctl.conflictClasses = null)}
        >
          Cancel
        </button>
      </div>
    </div>
  {/if}

  <KeymapGroupsSection {ctl} />

  <KeymapContextsSection {ctl} />

  <div class="mt-3 flex items-center gap-3">
    <button
      class="rounded bg-blue-700 px-3 py-1.5 text-sm text-white hover:bg-blue-600 disabled:opacity-50"
      disabled={!ctl.hasChanges || ctl.saving}
      onclick={ctl.openConfirm}
    >
      Save
    </button>
    <button
      class="text-sm text-zinc-400 underline hover:text-zinc-200"
      onclick={() => (ctl.resetAllOpen = true)}
    >
      Reset all to defaults
    </button>
    {#if ctl.validating}<span class="text-xs text-zinc-500">ctl.validating…</span>{/if}
  </div>

  <KeymapConfirmModals {ctl} />
</div>
