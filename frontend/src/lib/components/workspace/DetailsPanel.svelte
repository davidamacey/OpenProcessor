<script lang="ts">
  /**
   * The details of what has focus: a focused detection, else the selection,
   * else the open cluster; a focus the view cannot show is not shown. 360 px beside the content from 1280 px; a drawer
   * under that.
   */
  import { DETAILS } from '$lib/workspace/detailsRegistry';
  import { focusFitsView, type DetailsKind } from '$lib/workspace/detailsViews';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  const target = $derived.by((): { kind: DetailsKind; id: string | null } | null => {
    const focus = ws.url.focus;
    // Beside the Review pane the item shows what the pane does not.
    if (focus?.kind === 'item' && ws.view === 'review')
      return { kind: 'review', id: focus.id };
    if (focus && focus.kind in DETAILS && focusFitsView(focus.kind, ws.view))
      return { kind: focus.kind as DetailsKind, id: focus.id };
    if (ws.selection.kind !== 'none') return { kind: 'selection', id: null };
    if (ws.url.scope?.kind === 'cluster') return { kind: 'cluster', id: ws.url.scope.id };
    return null;
  });
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
      <p class="text-sm text-fg-3">Select something to see its details.</p>
    {/if}
  </aside>
{/if}
