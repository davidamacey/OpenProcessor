<script lang="ts">
  /**
   * The details of what has focus (see `detailsTarget`). 360 px beside the content from 1280 px; a drawer
   * under that.
   */
  import { DETAILS } from '$lib/workspace/detailsRegistry';
  import { detailsTarget } from '$lib/workspace/detailsTarget';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  const target = $derived(
    detailsTarget(ws.view, ws.url.focus, ws.selection, ws.url.scope),
  );
  const Body = $derived(target ? DETAILS[target.kind] : null);
</script>

{#if ws.detailsOpen}
  <aside
    class="fixed inset-y-0 right-0 z-50 w-[min(360px,100vw)] overflow-y-auto border-l border-line bg-surface-0 p-4 xl:static xl:z-auto xl:w-[360px] xl:shrink-0"
    aria-label="Details"
    data-testid="details-panel"
  >
    <div class="mb-2 flex justify-end xl:hidden">
      <button type="button" class="btn" onclick={() => ws.toggleDetails()}>Close</button>
    </div>
    {#if Body && target}
      <Body {ws} id={target.id} />
    {:else}
      <p class="text-body text-fg-3">Select something to see its details.</p>
    {/if}
  </aside>
{/if}
