<script lang="ts">
  /**
   * The confirm step before restoring a viewed revision: its body and
   * description are saved as a new revision (§7.6 item 1). It never
   * changes what is active.
   */
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import type { ConfigEditorView } from '$lib/config/configEditor.svelte';

  interface Props {
    ed: ConfigEditorView;
    /** "pack" / "profile". */
    noun: string;
    onclose: () => void;
    /** After a successful restore: the restored and the new revision. */
    onrestored?: (from: number | null, to: number | null) => void;
  }

  let { ed, noun, onclose, onrestored }: Props = $props();

  async function run(): Promise<void> {
    const from = ed.viewing?.revision ?? null;
    if (await ed.restoreViewed()) {
      onclose();
      onrestored?.(from, ed.doc?.revision ?? null);
    }
  }
</script>

{#if ed.viewing}
  <ConfirmDialog
    title="Restore revision {ed.viewing.revision}"
    confirmLabel="Restore"
    busy={ed.saving}
    onconfirm={() => void run()}
    oncancel={onclose}
  >
    <p>
      Saves revision {ed.viewing.revision}'s settings as a new revision. It doesn't change
      the active {noun}.
    </p>
    {#if ed.dirty}
      <p class="text-xs text-amber-300">Your unsaved edits are replaced.</p>
    {/if}
    {#if ed.saveError}<p class="text-red-300">{ed.saveError}</p>{/if}
    {#if ed.conflict}<p class="text-red-300">{ed.conflict.message}</p>{/if}
  </ConfirmDialog>
{/if}
