<script lang="ts">
  /**
   * The Workspace's views: the loop's four (Clusters, Gallery, Detections, Review), a divider,
   * then the advanced three (Table, Map, Embedding), as tabs from the registry; a `<select>`
   * under `md`. Switching keeps the filter, the queue, the scope and the selection. Under `lg`
   * the queues and classes rail and the filters are icon buttons in the same row.
   */
  import Icon from '$components/ui/Icon.svelte';
  import VlmMenu from '$components/workspace/VlmMenu.svelte';
  import { availableViews, viewEntry } from '$lib/workspace/viewRegistry';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import type { ViewId } from '$lib/workspace/workspaceUrl';
  import { keymapStore } from '$stores/keymap.svelte';

  let { ws }: { ws: WorkspaceController } = $props();
  const current = $derived(viewEntry(ws.view).id);
  const views = $derived(availableViews(ws));
</script>

<div
  class="flex items-center gap-2 border-b border-line px-4 py-1.5"
  data-testid="view-bar"
>
  <button
    type="button"
    class="btn btn-icon lg:hidden"
    aria-label="Queues and classes"
    aria-expanded={ws.railOpen}
    data-testid="rail-toggle"
    onclick={() => ws.toggleRail()}
  >
    <Icon name="menu" size={16} />
  </button>
  <div role="tablist" aria-label="Views" class="hidden items-center gap-1 md:flex">
    {#each views as v, i (v.id)}
      {#if v.advanced && !views[i - 1]?.advanced}
        <span class="mx-1 h-5 w-px bg-line-strong" aria-hidden="true"></span>
      {/if}
      <button
        type="button"
        role="tab"
        aria-selected={current === v.id}
        class="rounded px-3 py-1 text-body {current === v.id
          ? 'bg-surface-2 text-white'
          : 'text-fg-2 hover:text-white'}"
        data-testid="view-tab-{v.id}"
        onclick={() => ws.setView(v.id)}
      >
        {v.label}
      </button>
    {/each}
  </div>
  <select
    class="select md:hidden"
    aria-label="View"
    value={current}
    onchange={(e) => ws.setView(e.currentTarget.value as ViewId)}
    data-testid="view-select"
  >
    {#each views as v (v.id)}
      <option value={v.id}>{v.label}</option>
    {/each}
  </select>
  <span class="grow"></span>
  <VlmMenu {ws} />
  <button
    type="button"
    class="btn btn-ghost"
    aria-expanded={ws.detailsOpen}
    title="Details ({keymapStore.glyph('workspace.toggle_details')})"
    data-testid="details-toggle"
    onclick={() => ws.toggleDetails()}
  >
    Details
  </button>
</div>
