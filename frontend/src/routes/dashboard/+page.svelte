<script lang="ts">
  /*
   * Pipeline dashboard route — the app's single home page.
   *
   * Merged 2026-09 from the legacy `/` page (recent-crops + quick-actions
   * + a per-class balance chart) into this one, since `/dashboard` was
   * already the live/current pipeline view and having two overlapping
   * "home" pages was confusing. Nothing from the legacy page was
   * dropped — see below for where each piece landed:
   *
   *   1. `AutoLabelPanel` / `DatasetStats` — unchanged, already here.
   *   2. Quick actions (Run Gemma / Export / Snapshot) — brought over
   *      verbatim, same handlers.
   *   3. Class balance bar chart — brought over verbatim; this is a
   *      genuinely different view from DatasetStats' "Labeled by
   *      source" breakdown (per-CLASS, not per-source), so it's kept
   *      as its own section rather than folded into DatasetStats.
   *   4. Recent labels grid — brought over verbatim.
   *
   * The legacy Ingestion panel (Total crops/Validated/Test holdout/
   * Images processed/Pending/Last run) was NOT brought over — every one
   * of those numbers is already shown live (SSE, not a 30s poll) in
   * DatasetStats' "Pipeline stats" and "In-flight pipeline" sections.
   * Duplicating it here would just be two numbers disagreeing during
   * the gap between polls.
   */
  import { apiBase } from '$lib/api';
  import { healthStore } from '$stores/health.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import AutoLabelPanel from '$components/AutoLabelPanel.svelte';
  import DatasetStats from '$components/DatasetStats.svelte';
  import { exportYolo, getCrops, getStats, getThumbUrl, runVlmOnCluster } from '$lib/api';
  import { adequacyLevel } from '$lib/adequacy';
  import type { OpCrop, OpStats } from '$lib/types';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('dashboard');
  });

  // ---- Class balance + recent labels (from the legacy `/` page) --------

  let legacyStats = $state<OpStats | null>(null);
  let recent = $state<OpCrop[]>([]);
  let legacyLoading = $state<boolean>(false);

  let vlmOpen = $state<boolean>(false);
  let vlmClusterId = $state<string>('');
  let vlmBusy = $state<boolean>(false);

  async function refreshLegacy(): Promise<void> {
    legacyLoading = true;
    try {
      const [s, c] = await Promise.allSettled([
        getStats(),
        getCrops({
          label_validated: true,
          sort: 'updated_at:desc',
          limit: 20,
          page: 1,
        }),
      ]);
      legacyStats = s.status === 'fulfilled' ? s.value : null;
      recent = c.status === 'fulfilled' ? c.value.items : [];
    } finally {
      legacyLoading = false;
    }
  }

  $effect(() => {
    void refreshLegacy();
    const id = setInterval(() => void refreshLegacy(), 30_000);
    return () => clearInterval(id);
  });

  async function runVlm(): Promise<void> {
    const id = Number(vlmClusterId);
    if (!Number.isFinite(id) || id < 0) {
      toastStore.error('Enter a valid cluster id');
      return;
    }
    vlmBusy = true;
    try {
      const res = await runVlmOnCluster(id);
      toastStore.success(
        `VLM labeled ${res.predicted ?? 0} crops (${res.updated ?? 0} updated).`,
      );
      vlmOpen = false;
    } catch (e) {
      toastStore.error(`VLM run failed: ${(e as Error).message}`);
    } finally {
      vlmBusy = false;
    }
  }

  async function runExport(): Promise<void> {
    try {
      const res = await exportYolo();
      toastStore.success(`Export job started: ${res.job_id ?? res.status}`);
    } catch (e) {
      toastStore.error(`Export failed: ${(e as Error).message}`);
    }
  }

  const balance = $derived.by(() => {
    if (!legacyStats?.per_class) return [];
    const max = Math.max(1, ...legacyStats.per_class.map((c) => c.validated_count));
    return [...legacyStats.per_class]
      .sort((a, b) => b.validated_count - a.validated_count)
      .slice(0, 30)
      .map((c) => ({
        ...c,
        pct: Math.max(2, Math.round((c.validated_count / max) * 100)),
        tier: adequacyLevel(c.validated_count),
      }));
  });

  function tierBg(t: string): string {
    if (t === 'ok') return 'bg-green-500';
    if (t === 'low') return 'bg-orange-500';
    return 'bg-red-500';
  }
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
      <strong>API unavailable.</strong> Check that the OpenProcessor API is reachable from
      <code class="font-mono"
        >{apiBase || (typeof window !== 'undefined' ? window.location.host : '')}</code
      >.
    </div>
  {/if}

  <!-- Quick actions -->
  <section class="surface flex flex-wrap items-center gap-2 p-4">
    <h2 class="mr-3 text-sm font-semibold text-zinc-400">Quick actions</h2>
    <button class="btn btn-primary" type="button" onclick={() => (vlmOpen = true)}>
      Run VLM Labeling
    </button>
    <button class="btn" type="button" onclick={runExport}>Export Dataset (YOLO)</button>
    <span class="grow"></span>
    <button
      class="btn"
      type="button"
      onclick={() => void refreshLegacy()}
      disabled={legacyLoading}
    >
      {legacyLoading ? 'Refreshing...' : 'Refresh'}
    </button>
  </section>

  <AutoLabelPanel />

  <DatasetStats />

  <!-- Class balance -->
  <section class="surface p-4">
    <header class="mb-3 flex items-center justify-between">
      <h2 class="text-sm font-semibold text-zinc-300">Class balance (validated)</h2>
      <span class="text-xs text-zinc-500">
        green ≥500 · orange 100–499 · red &lt;100
      </span>
    </header>

    {#if !legacyStats}
      <p class="text-sm text-zinc-500">Loading...</p>
    {:else if balance.length === 0}
      <p class="text-sm text-zinc-500">No classes yet.</p>
    {:else}
      <ul class="space-y-1.5">
        {#each balance as row (row.class_id)}
          <li class="flex items-center gap-3 text-xs">
            <span class="w-32 shrink-0 truncate text-zinc-300" title={row.class_name}>
              {row.class_name}
            </span>
            <div class="relative h-4 grow overflow-hidden rounded bg-zinc-900">
              <div class="h-full {tierBg(row.tier)}" style:width="{row.pct}%"></div>
            </div>
            <span class="w-16 shrink-0 text-right font-mono text-zinc-400">
              {row.validated_count}
            </span>
          </li>
        {/each}
      </ul>
    {/if}
  </section>

  <!-- Recent activity -->
  <section class="surface p-4">
    <h2 class="mb-3 text-sm font-semibold text-zinc-300">Recent labels (last 20)</h2>
    {#if recent.length === 0}
      <p class="text-sm text-zinc-500">No recent labels.</p>
    {:else}
      <ul class="grid grid-cols-3 gap-3 sm:grid-cols-5 lg:grid-cols-10">
        {#each recent as crop (crop.id)}
          <li class="rounded-md border border-zinc-800 bg-zinc-900">
            <img
              src={getThumbUrl(crop.id)}
              alt="crop"
              loading="lazy"
              class="aspect-square w-full rounded-t-md object-contain"
            />
            <div
              class="truncate px-1.5 py-1 text-[10px] text-zinc-300"
              title={crop.class_name ?? ''}
            >
              {crop.class_name ?? '—'}
            </div>
          </li>
        {/each}
      </ul>
    {/if}
  </section>
</div>

{#if vlmOpen}
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-3 text-base font-semibold">Run VLM on cluster</h3>
      <label class="mb-3 block text-sm">
        <span class="mb-1 block text-zinc-400">Cluster ID</span>
        <input
          type="number"
          min="0"
          bind:value={vlmClusterId}
          class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
          placeholder="e.g. 42"
        />
      </label>
      <p class="mb-4 text-xs text-zinc-500">
        Only un-validated crops in the cluster will be sent. Test-holdout crops are
        excluded by the API.
      </p>
      <div class="flex justify-end gap-2">
        <button
          type="button"
          class="btn"
          onclick={() => (vlmOpen = false)}
          disabled={vlmBusy}
        >
          Cancel
        </button>
        <button type="button" class="btn btn-primary" onclick={runVlm} disabled={vlmBusy}>
          {vlmBusy ? 'Running...' : 'Run'}
        </button>
      </div>
    </div>
  </div>
{/if}
