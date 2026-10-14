<script lang="ts">
  import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';

  // Sub-cluster tabs produced by a refine; "all" shows them grouped inline.
  let { view }: { view: ClusterDetailController } = $props();
</script>

<!-- Sub-cluster tabs -->
{#if view.subClusterIds.length > 0}
  <div class="flex items-center gap-2 border-b border-zinc-800 px-4 py-1.5 text-xs">
    <span class="text-zinc-500">sub-clusters:</span>
    <button
      type="button"
      class="chip {view.subTab === null
        ? 'bg-blue-600 text-white'
        : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
      onclick={() => (view.subTab = null)}
    >
      all
    </button>
    {#each view.subClusterIds as sid (sid)}
      <button
        type="button"
        class="chip {view.subTab === sid
          ? 'bg-blue-600 text-white'
          : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
        onclick={() => (view.subTab = sid)}
      >
        #{sid}
      </button>
    {/each}
  </div>
{/if}
