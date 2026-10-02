<script lang="ts">
  /**
   * A stored config doc's served revision history (§4.2, §7.2), newest as
   * served; "View" opens one read-only in the editor.
   */
  import type { ConfigEditorView } from '$lib/config/configEditor.svelte';
  import { formatTimestamp } from '$lib/formatDate';

  interface Props {
    ed: ConfigEditorView;
  }

  let { ed }: Props = $props();
</script>

{#if ed.revisions}
  <section class="surface flex flex-col gap-2 p-4 text-sm" aria-label="Revisions">
    <h2 class="text-sm font-semibold">Revisions</h2>
    {#if ed.revisionError}<p class="text-xs text-red-300">{ed.revisionError}</p>{/if}
    <ul class="space-y-1" data-testid="revisions">
      {#each ed.revisions as r (r.revision)}
        <li
          class="flex flex-wrap items-center gap-2 text-xs"
          data-testid="revision-row"
          data-revision={r.revision}
        >
          <span class="font-mono">r{r.revision}</span>
          <span class="text-zinc-500" title={r.saved_at}
            >{formatTimestamp(r.saved_at)}</span
          >
          {#if r.revision === ed.doc?.revision}<span class="text-zinc-400">latest</span
            >{/if}
          {#if ed.doc?.active && r.revision === ed.doc.active_revision}<span
              class="text-emerald-300">active</span
            >{/if}
          <span class="grow"></span>
          <button
            type="button"
            class="btn btn-sm"
            disabled={ed.viewing?.revision === r.revision}
            onclick={() => void ed.viewRevision(r.revision)}>View</button
          >
          {#if r.description}
            <span class="w-full text-zinc-400">{r.description}</span>
          {/if}
          {#if r.cloned_from}
            <span class="w-full font-mono text-zinc-500">cloned from {r.cloned_from}</span
            >
          {/if}
        </li>
      {/each}
    </ul>
  </section>
{/if}
