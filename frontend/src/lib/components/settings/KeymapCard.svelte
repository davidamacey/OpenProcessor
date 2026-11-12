<script lang="ts">
  /**
   * "Keyboard shortcuts" card — `/settings/keyboard` (K2 + K2b,
   * configurable keyboard shortcuts plan §5.4).
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
   * the Settings layout gates the section on that, mirroring `ScoresCard`'s own
   * "absent on a pre-route backend" rule.
   */

  import SaveBar from '$components/ui/SaveBar.svelte';
  import { keymapStore } from '$stores/keymap.svelte';
  import KeymapGroupsSection from '$components/settings/keymap/KeymapGroupsSection.svelte';
  import KeymapContextsSection from '$components/settings/keymap/KeymapContextsSection.svelte';
  import KeymapConfirmModals from '$components/settings/keymap/KeymapConfirmModals.svelte';
  import { createKeymapCardController } from '$components/settings/keymap/keymapCardController.svelte';

  const ctl = createKeymapCardController();
</script>

<div id="keyboard">
  {#if !keymapStore.isDefault}
    <p class="mb-3">
      <span
        class="rounded-full border border-warn-line bg-warn-soft px-2 py-0.5 text-caption text-warn-text"
      >
        custom keys
      </span>
    </p>
  {/if}

  {#if ctl.revisionConflictMessage}
    <div
      class="mb-3 rounded border border-danger-line bg-danger-soft p-2 text-xs text-danger-text"
    >
      {ctl.revisionConflictMessage} Shortcuts were changed elsewhere.
      <button class="ml-2 underline" onclick={() => void ctl.reloadAfterConflict()}>
        Reload
      </button>
    </div>
  {/if}

  {#if ctl.conflictClasses && ctl.conflictClasses.length > 0}
    <div
      class="mb-3 rounded border border-warn-line bg-warn-soft p-2 text-xs text-warn-text"
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
          class="rounded bg-warn px-2 py-1 text-white hover:bg-warn"
          onclick={() => void ctl.doSave(true)}
        >
          Unbind these class keys and save
        </button>
        <button class="text-fg-2 underline" onclick={() => (ctl.conflictClasses = null)}>
          Cancel
        </button>
      </div>
    </div>
  {/if}

  <label class="mb-3 block">
    <span class="sr-only">Search shortcuts</span>
    <input
      type="search"
      class="input w-full max-w-sm"
      placeholder="Search shortcuts"
      data-testid="keymap-search"
      bind:value={ctl.filter}
    />
  </label>

  <KeymapGroupsSection {ctl} />

  <KeymapContextsSection {ctl} />

  <div class="mt-3">
    <button
      class="text-sm text-fg-2 underline hover:text-fg-1"
      onclick={() => (ctl.resetAllOpen = true)}
    >
      Reset all to defaults
    </button>
    {#if ctl.validating}<span class="ml-3 text-xs text-fg-3">Checking…</span>{/if}
  </div>

  <SaveBar
    dirty={ctl.hasChanges}
    saving={ctl.saving}
    onsave={ctl.openConfirm}
    ondiscard={ctl.discardChanges}
  />

  <KeymapConfirmModals {ctl} />
</div>
