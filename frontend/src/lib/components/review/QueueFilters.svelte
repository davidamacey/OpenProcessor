<!--
  The filters only the active queue serves (the region queue's status and text search, "include
  the test holdout", ...): one control per served spec the shared item filter does not own, in the
  served order. They live in the URL as plain parameters beside the Workspace filter, so a Review
  link carries them.
-->
<script lang="ts">
  import ServedFilterField from '$lib/components/itemFilter/ServedFilterField.svelte';
  import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws, tabs }: { ws: WorkspaceController; tabs: ReviewTabsController } = $props();

  function set(param: string, value: string | string[]): void {
    ws.patchUrl((params) => {
      params.delete(param);
      for (const v of Array.isArray(value) ? value : value ? [value] : []) {
        params.append(param, v);
      }
    });
  }
</script>

{#if tabs.queueFilterSpecs.length > 0}
  <div
    class="flex min-w-0 flex-wrap items-center gap-3 border-b border-zinc-800 bg-zinc-900/40 px-4 py-2 text-xs"
    data-testid="queue-filters"
  >
    {#each tabs.queueFilterSpecs as spec (spec.param)}
      <ServedFilterField
        {spec}
        value={tabs.queueFilterValues[spec.param]}
        onchange={set}
      />
    {/each}
  </div>
{/if}
