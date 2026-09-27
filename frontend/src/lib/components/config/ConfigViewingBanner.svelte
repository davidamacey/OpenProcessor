<script lang="ts">
  /**
   * Shown while a past revision is open read-only: back to the latest,
   * Restore (a stored doc only) and Activate that revision.
   */
  import type { ConfigEditorView } from '$lib/config/configEditor.svelte';

  interface Props {
    ed: ConfigEditorView;
    onrestore: () => void;
    onactivate: () => void;
  }

  let { ed, onrestore, onactivate }: Props = $props();
</script>

{#if ed.viewing}
  {@const v = ed.viewing}
  <div
    class="flex flex-wrap items-center gap-2 rounded border border-sky-500/40 bg-sky-500/10 px-3 py-2 text-sm text-sky-100"
    data-testid="viewing-banner"
  >
    <span>Viewing revision {v.revision} (read-only)</span>
    <span class="grow"></span>
    <button type="button" class="btn btn-sm" onclick={() => ed.closeRevision()}
      >Back to latest</button
    >
    {#if ed.doc && !ed.doc.read_only}
      <button
        type="button"
        class="btn btn-sm"
        onclick={onrestore}
        data-testid="restore-revision">Restore as new revision</button
      >
    {/if}
    <button
      type="button"
      class="btn btn-sm"
      onclick={onactivate}
      data-testid="activate-viewed">Activate revision {v.revision}</button
    >
  </div>
{/if}
