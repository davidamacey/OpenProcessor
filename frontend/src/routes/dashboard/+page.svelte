<script lang="ts">
  /*
   * Pipeline dashboard route.
   *
   * Two panels stacked vertically:
   *   1. `DatasetStats` — live pipeline counts (polls every 10s).
   *   2. `AutoLabelPanel` — the existing "Run Clustering Now" control
   *      with stage-by-stage progress; reused so the manual trigger
   *      and the daemon-fired run share UI.
   */
  import { apiBase } from '$lib/api';
  import { healthStore } from '$stores/health.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import AutoLabelPanel from '$components/AutoLabelPanel.svelte';
  import DatasetStats from '$components/DatasetStats.svelte';

  $effect(() => {
    keyboardStore.setScope('dashboard');
  });
</script>

<div class="mx-auto max-w-7xl space-y-6 p-6">
  <header class="flex items-end justify-between">
    <div>
      <h1 class="text-2xl font-semibold tracking-tight">Dashboard</h1>
      <p class="text-sm text-zinc-500">
        Pipeline stats and manual clustering control. Stats refresh every 10s.
      </p>
    </div>
  </header>

  {#if !healthStore.ok}
    <div
      class="rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200"
    >
      <strong>API unavailable.</strong> Check that openprocessor is running on
      <code class="font-mono"
        >{apiBase || (typeof window !== 'undefined' ? window.location.host : '')}</code
      >.
    </div>
  {/if}

  <AutoLabelPanel />

  <DatasetStats />
</div>
