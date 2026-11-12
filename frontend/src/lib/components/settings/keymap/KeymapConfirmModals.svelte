<script lang="ts">
  import { formatShortcutKey } from '$lib/keyboardDisplay';
  import { keymapStore } from '$stores/keymap.svelte';
  import Modal from '$components/Modal.svelte';
  import type { KeymapCardController } from '$components/settings/keymap/keymapCardController.svelte';

  let { ctl }: { ctl: KeymapCardController } = $props();
</script>

{#if ctl.confirmOpen}
  <Modal
    label="Save keyboard shortcuts"
    onclose={ctl.closeConfirm}
    layer="top"
    scrim="bg-black/70"
    panelClass="w-full max-w-md rounded-lg border border-line-strong bg-surface-0 p-4"
  >
    <h3 class="mb-2 text-sm font-semibold text-white">Save keyboard shortcuts?</h3>
    <ul class="mb-3 max-h-48 overflow-y-auto text-xs text-fg-2">
      {#each ctl.changedActionIds() as id (id)}
        <li>
          {keymapStore.label(id)}: {ctl.keysOf(id).map(formatShortcutKey).join(', ')}
        </li>
      {/each}
    </ul>
    <div class="flex justify-end gap-2">
      <button class="text-sm text-fg-2" onclick={ctl.closeConfirm}>Cancel</button>
      <button
        class="rounded bg-accent-fill px-3 py-1.5 text-sm text-white hover:bg-accent-fill"
        onclick={() => void ctl.doSave(false)}
      >
        Save
      </button>
    </div>
  </Modal>
{/if}

{#if ctl.resetAllOpen}
  <Modal
    label="Reset every shortcut"
    onclose={() => (ctl.resetAllOpen = false)}
    layer="top"
    scrim="bg-black/70"
    panelClass="w-full max-w-md rounded-lg border border-line-strong bg-surface-0 p-4"
  >
    <h3 class="mb-2 text-sm font-semibold text-white">
      Reset every shortcut to its default?
    </h3>
    <div class="flex justify-end gap-2">
      <button class="text-sm text-fg-2" onclick={() => (ctl.resetAllOpen = false)}>
        Cancel
      </button>
      <button
        class="rounded bg-danger-fill px-3 py-1.5 text-sm text-white hover:bg-danger-fill"
        onclick={() => void ctl.doResetAll()}
      >
        Reset all
      </button>
    </div>
  </Modal>
{/if}
