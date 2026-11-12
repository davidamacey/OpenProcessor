<script lang="ts">
  /** What the selection holds, as the server counts it. */
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws }: { ws: WorkspaceController; id: string | null } = $props();

  const s = $derived(ws.summary.summary);
</script>

<section class="space-y-2 text-sm" data-testid="selection-details">
  <h2 class="text-xs font-semibold tracking-wide text-zinc-400 uppercase">Selection</h2>
  {#if s}
    <p class="text-zinc-100">{s.summary_text}</p>
    <dl class="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-zinc-300">
      <dt class="text-zinc-500">clusters</dt>
      <dd>{s.clusters.toLocaleString()}</dd>
      <dt class="text-zinc-500">in test holdout</dt>
      <dd>{s.in_holdout.toLocaleString()}</dd>
      <dt class="text-zinc-500">with a position</dt>
      <dd>{s.located.toLocaleString()}</dd>
    </dl>
    <ul class="space-y-0.5 text-xs text-zinc-300" data-testid="selection-by-class">
      {#each s.by_class as c (c.class_name)}
        <li class="flex justify-between gap-2">
          <span class="truncate">{c.class_name}</span>
          <span class="font-mono">{c.count.toLocaleString()}</span>
        </li>
      {/each}
    </ul>
  {:else}
    <p class="text-zinc-500">Nothing selected.</p>
  {/if}
</section>
