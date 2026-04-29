<script lang="ts">
  import { exportYolo, getCrops, getStats, getThumbUrl, runGemmaOnCluster } from '$lib/api';
  import type { OpCrop, OpStats } from '$lib/types';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { healthStore } from '$stores/health.svelte';

  let stats = $state<OpStats | null>(null);
  let recent = $state<OpCrop[]>([]);
  let loading = $state<boolean>(false);
  let error = $state<string | null>(null);

  let gemmaOpen = $state<boolean>(false);
  let gemmaClusterId = $state<string>('');
  let gemmaBusy = $state<boolean>(false);

  async function refresh(): Promise<void> {
    loading = true;
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
      if (s.status === 'fulfilled') stats = s.value;
      else stats = null;
      if (c.status === 'fulfilled') recent = c.value.items;
      else recent = [];
      error = s.status === 'rejected' && c.status === 'rejected' ? 'API unavailable' : null;
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    keyboardStore.setScope('dashboard');
    void refresh();
    const id = setInterval(() => void refresh(), 30_000);
    return () => clearInterval(id);
  });

  async function runGemma(): Promise<void> {
    const id = Number(gemmaClusterId);
    if (!Number.isFinite(id) || id < 0) {
      toastStore.error('Enter a valid cluster id');
      return;
    }
    gemmaBusy = true;
    try {
      const res = await runGemmaOnCluster(id);
      toastStore.success(`Enqueued ${res.enqueued ?? 0} crops for Gemma`);
      gemmaOpen = false;
    } catch (e) {
      toastStore.error(`Gemma run failed: ${(e as Error).message}`);
    } finally {
      gemmaBusy = false;
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

  function snapshot(): void {
    toastStore.info('op_* snapshot is wired to /curation/admin/snapshot in v1.1');
  }

  // Class balance chart data
  const balance = $derived.by(() => {
    if (!stats?.per_class) return [];
    const max = Math.max(1, ...stats.per_class.map((c) => c.validated_count));
    return [...stats.per_class]
      .sort((a, b) => b.validated_count - a.validated_count)
      .slice(0, 30)
      .map((c) => ({
        ...c,
        pct: Math.max(2, Math.round((c.validated_count / max) * 100)),
        tier:
          c.validated_count >= 500
            ? 'green'
            : c.validated_count >= 100
              ? 'orange'
              : 'red',
      }));
  });

  function tierBg(t: string): string {
    if (t === 'green') return 'bg-green-500';
    if (t === 'orange') return 'bg-orange-500';
    return 'bg-red-500';
  }
</script>

<div class="mx-auto max-w-7xl space-y-6 p-6">
  <h1 class="text-2xl font-semibold tracking-tight">Dashboard</h1>

  {#if !healthStore.ok}
    <div
      class="rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200"
    >
      <strong>API unavailable.</strong> Check that openprocessor is running on
      <code class="font-mono">localhost:4603</code>.
    </div>
  {/if}

  <!-- Quick actions -->
  <section class="surface flex flex-wrap items-center gap-2 p-4">
    <h2 class="mr-3 text-sm font-semibold text-zinc-400">Quick actions</h2>
    <button class="btn btn-primary" type="button" onclick={() => (gemmaOpen = true)}>
      Run Gemma Labeling
    </button>
    <button class="btn" type="button" onclick={runExport}>Export Dataset (YOLO)</button>
    <button class="btn" type="button" onclick={snapshot}>Snapshot op_* indexes</button>
    <span class="grow"></span>
    <button class="btn" type="button" onclick={() => void refresh()} disabled={loading}>
      {loading ? 'Refreshing...' : 'Refresh'}
    </button>
  </section>

  <div class="grid grid-cols-1 gap-6 lg:grid-cols-3">
    <!-- Class balance -->
    <section class="surface lg:col-span-2 p-4">
      <header class="mb-3 flex items-center justify-between">
        <h2 class="text-sm font-semibold text-zinc-300">Class balance (validated)</h2>
        <span class="text-xs text-zinc-500">
          green ≥500 · orange 100–499 · red &lt;100
        </span>
      </header>

      {#if !stats}
        <p class="text-sm text-zinc-500">{error ?? 'Loading...'}</p>
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
                <div
                  class="h-full {tierBg(row.tier)}"
                  style:width="{row.pct}%"
                ></div>
              </div>
              <span class="w-16 shrink-0 text-right font-mono text-zinc-400">
                {row.validated_count}
              </span>
            </li>
          {/each}
        </ul>
      {/if}
    </section>

    <!-- Ingestion -->
    <section class="surface p-4">
      <h2 class="mb-3 text-sm font-semibold text-zinc-300">Ingestion</h2>
      {#if !stats}
        <p class="text-sm text-zinc-500">{error ?? 'Loading...'}</p>
      {:else}
        <dl class="space-y-2 text-sm">
          <div class="flex justify-between">
            <dt class="text-zinc-400">Total crops</dt>
            <dd class="font-mono">{stats.total_crops?.toLocaleString() ?? 0}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Validated</dt>
            <dd class="font-mono text-green-300">
              {stats.validated_crops?.toLocaleString() ?? 0}
            </dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Test holdout</dt>
            <dd class="font-mono">{stats.test_holdout_crops?.toLocaleString() ?? 0}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Images processed</dt>
            <dd class="font-mono">
              {stats.ingestion?.images_processed?.toLocaleString() ?? 0}
            </dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Pending</dt>
            <dd class="font-mono text-orange-300">
              {stats.ingestion?.images_pending?.toLocaleString() ?? 0}
            </dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Last run</dt>
            <dd class="text-xs text-zinc-300">
              {stats.ingestion?.last_run_at
                ? new Date(stats.ingestion.last_run_at).toLocaleString()
                : '—'}
            </dd>
          </div>
        </dl>
      {/if}
    </section>
  </div>

  <!-- Recent activity -->
  <section class="surface p-4">
    <h2 class="mb-3 text-sm font-semibold text-zinc-300">Recent labels (last 20)</h2>
    {#if recent.length === 0}
      <p class="text-sm text-zinc-500">No recent labels.</p>
    {:else}
      <ul class="grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-10">
        {#each recent as crop (crop.id)}
          <li class="rounded-md border border-zinc-800 bg-zinc-900">
            <img
              src={getThumbUrl(crop.id)}
              alt="crop"
              loading="lazy"
              class="aspect-square w-full rounded-t-md object-cover"
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

{#if gemmaOpen}
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
  >
    <div class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl">
      <h3 class="mb-3 text-base font-semibold">Run Gemma on cluster</h3>
      <label class="mb-3 block text-sm">
        <span class="mb-1 block text-zinc-400">Cluster ID</span>
        <input
          type="number"
          min="0"
          bind:value={gemmaClusterId}
          class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
          placeholder="e.g. 42"
        />
      </label>
      <p class="mb-4 text-xs text-zinc-500">
        Only un-validated crops in the cluster will be sent. Test-holdout crops are excluded by
        the API.
      </p>
      <div class="flex justify-end gap-2">
        <button
          type="button"
          class="btn"
          onclick={() => (gemmaOpen = false)}
          disabled={gemmaBusy}
        >
          Cancel
        </button>
        <button
          type="button"
          class="btn btn-primary"
          onclick={runGemma}
          disabled={gemmaBusy}
        >
          {gemmaBusy ? 'Running...' : 'Run'}
        </button>
      </div>
    </div>
  </div>
{/if}
