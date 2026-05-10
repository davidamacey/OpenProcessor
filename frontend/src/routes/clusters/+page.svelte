<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { getClusters, getThumbUrl } from '$lib/api';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import type { ClusterFilter, OpCluster } from '$lib/types';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';

  let clusters = $state<OpCluster[]>([]);
  let total = $state<number>(0);
  let loadedPages = $state<number>(0);
  let loading = $state<boolean>(false);
  let loadingMore = $state<boolean>(false);
  let error = $state<string | null>(null);
  const hasMore = $derived(clusters.length < total);

  let sort = $state<NonNullable<ClusterFilter['sort']>>('purity_asc');
  const pageSize = 24;

  const classFilter = $derived.by(() => {
    const v = page.url.searchParams.get('class');
    return v == null ? null : Number.isFinite(+v) ? +v : null;
  });

  // The legacy ensemble stores plates as a *sub-bbox* on each vehicle
  // crop (`plate_bbox_norm`), NOT as standalone docs in the cluster
  // index. So filtering this page by the `license_plate` class always
  // returns 0 / unlabeled clusters — confusing operators who expect to
  // see plate clusters here. Detect that case and route the user to the
  // plates review queue instead, which is the actual home for plate
  // labeling.
  const isLicensePlateFilter = $derived.by<boolean>(() => {
    if (classFilter == null) return false;
    const cls = classesStore.classes.find((c) => c.id === classFilter);
    return (cls?.name ?? '').toLowerCase() === 'license_plate';
  });

  async function loadFirst(): Promise<void> {
    loading = true;
    error = null;
    clusters = [];
    total = 0;
    loadedPages = 0;
    try {
      const res = await getClusters({
        class_id: classFilter ?? undefined,
        sort,
        page: 1,
        page_size: pageSize,
      });
      clusters = res?.items ?? [];
      total = res?.total ?? clusters.length;
      loadedPages = 1;
    } catch (e) {
      error = (e as Error).message;
    } finally {
      loading = false;
    }
  }

  async function loadMore(): Promise<void> {
    if (loadingMore || !hasMore) return;
    loadingMore = true;
    try {
      const next = loadedPages + 1;
      const res = await getClusters({
        class_id: classFilter ?? undefined,
        sort,
        page: next,
        page_size: pageSize,
      });
      const seen = new Set(clusters.map((c) => c.id));
      const fresh = (res?.items ?? []).filter((c) => !seen.has(c.id));
      clusters = [...clusters, ...fresh];
      total = res?.total ?? total;
      loadedPages = next;
    } catch (e) {
      error = (e as Error).message;
    } finally {
      loadingMore = false;
    }
  }

  $effect(() => {
    keyboardStore.setScope('clusters');
  });

  // Re-load on filter / sort change
  $effect(() => {
    void classFilter;
    void sort;
    void loadFirst();
  });

  function borderColor(c: OpCluster): string {
    if (c.has_subclusters) return 'border-blue-500/60';
    const p = c.purity ?? 0;
    if (p >= 0.8) return 'border-green-500/60';
    if (p >= 0.6) return 'border-orange-500/60';
    return 'border-red-500/60';
  }

  function purityBadge(c: OpCluster): { color: string; text: string } {
    const p = c.purity ?? 0;
    if (p >= 0.8) return { color: 'bg-green-500/20 text-green-300', text: 'pure' };
    if (p >= 0.6) return { color: 'bg-orange-500/20 text-orange-200', text: 'mixed' };
    return { color: 'bg-red-500/20 text-red-200', text: 'noisy' };
  }

  function open(c: OpCluster): void {
    void goto(`/clusters/${c.id}`);
  }

  // Infinite scroll owns pagination — totalPages no longer needed.
</script>

<div class="flex h-full flex-col">
  <!-- Toolbar -->
  <div
    class="flex flex-wrap items-center gap-3 border-b border-zinc-800 px-4 py-2.5"
  >
    <h1 class="text-lg font-semibold">Clusters</h1>

    {#if classFilter != null}
      <span
        class="rounded-md border border-blue-500/40 bg-blue-500/10 px-2 py-0.5 text-xs text-blue-200"
      >
        class filter: #{classFilter}
      </span>
    {/if}

    <span class="grow"></span>

    <!-- Color legend for the card border. The cluster grid uses border
         color to encode purity at a glance; without this strip the user
         has to mouse over each card to figure out what the colors mean. -->
    <div class="flex items-center gap-2 text-[10px] text-zinc-500" title="Card border color encodes cluster purity">
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-green-500/60"></span>
        ≥80%
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-orange-500/60"></span>
        ≥60%
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-red-500/60"></span>
        &lt;60%
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-blue-500/60"></span>
        sub-clustered
      </span>
    </div>

    <label class="flex items-center gap-2 text-xs text-zinc-400">
      Sort
      <select
        bind:value={sort}
        class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-100"
      >
        <option value="purity_asc">purity asc</option>
        <option value="purity_desc">purity desc</option>
        <option value="size_desc">size desc</option>
        <option value="size_asc">size asc</option>
        <option value="dominant_class">dominant class</option>
      </select>
    </label>
  </div>

  <!-- Grid -->
  <div class="flex-1 overflow-auto p-4">
    {#if isLicensePlateFilter}
      <!-- Plates aren't standalone crops; clustering them here would
           always show zeros / unlabeled. Route the operator to the
           plates review queue, which IS the home for plate labeling. -->
      <div class="mx-auto max-w-2xl rounded-md border border-blue-500/40 bg-blue-500/5 p-6">
        <h2 class="text-base font-semibold text-blue-200">
          License plates aren't clustered here
        </h2>
        <p class="mt-2 text-sm text-zinc-300">
          Plates live as a <code class="rounded bg-zinc-800 px-1 py-0.5 text-[12px]">plate_bbox_norm</code>
          sub-box on each vehicle crop, not as separate documents in the
          cluster index — so filtering this page by
          <span class="font-mono text-blue-200">license_plate</span> always shows zeros.
        </p>
        <p class="mt-2 text-sm text-zinc-300">
          Review and confirm plate boxes on the dedicated queue:
        </p>
        <div class="mt-4 flex gap-2">
          <a
            href="/review?tab=plates"
            class="btn btn-primary"
          >
            Open Plates review queue
          </a>
          <button class="btn" type="button" onclick={() => goto('/clusters')}>
            Clear class filter
          </button>
        </div>
      </div>
    {:else if loading && clusters.length === 0}
      <p class="text-sm text-zinc-500">Loading...</p>
    {:else if error}
      <p class="text-sm text-red-300">API unavailable: {error}</p>
    {:else if clusters.length === 0}
      <p class="text-sm text-zinc-500">
        No clusters yet — ingest some images and run the auto-label pipeline.
      </p>
    {:else}
      <ul class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {#each clusters as c (c.id)}
          {@const pb = purityBadge(c)}
          <li>
            <button
              type="button"
              class="flex w-full flex-col rounded-md border-2 bg-zinc-900 text-left transition hover:border-zinc-300 {borderColor(
                c,
              )}"
              onclick={() => open(c)}
            >
              <div class="grid grid-cols-2 gap-px overflow-hidden rounded-t bg-zinc-950">
                {#each c.representative_crop_ids?.slice(0, 4) ?? [] as cropId (cropId)}
                  <img
                    src={getThumbUrl(cropId)}
                    alt="thumb"
                    loading="lazy"
                    class="aspect-square w-full bg-zinc-950 object-contain"
                  />
                {/each}
                {#each Array(Math.max(0, 4 - (c.representative_crop_ids?.length ?? 0))) as _, i (i)}
                  <div class="aspect-square w-full bg-zinc-900"></div>
                {/each}
              </div>
              <div class="p-3">
                <div class="mb-1 flex items-center gap-2">
                  <span class="text-sm font-semibold">#{c.id}</span>
                  <span
                    class="rounded px-1.5 py-0.5 text-[10px] font-medium {pb.color}"
                  >
                    {pb.text} {((c.purity ?? 0) * 100).toFixed(0)}
                  </span>
                  {#if c.has_subclusters}
                    <span
                      class="rounded border border-blue-500/40 bg-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-200"
                    >
                      AHC
                    </span>
                  {/if}
                  <span class="grow"></span>
                  <span class="font-mono text-xs text-zinc-400">{c.size}</span>
                </div>
                <div
                  class="truncate text-sm text-zinc-300"
                  title={c.dominant_class_name ?? '—'}
                >
                  {c.dominant_class_name ?? 'unlabeled'}
                  <span class="text-zinc-500">
                    · {((c.dominant_pct ?? 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </button>
          </li>
        {/each}
      </ul>
      <!-- Sentinel MUST live inside the scroll container so the IntersectionObserver
           can root itself on the right element. Outside the overflow-auto div the
           observer falls back to viewport and either never fires or fires forever. -->
      <div
        use:infiniteScroll={{
          onload: loadMore,
          disabled: loadingMore || !hasMore || loading,
        }}
        class="mt-4 h-1"
        aria-hidden="true"
      ></div>
    {/if}
  </div>

  <!-- Status bar (no scroll sentinel here — see above) -->
  <div
    class="flex items-center justify-between gap-3 border-t border-zinc-800 px-4 py-2 text-sm"
  >
    <span class="font-mono text-xs text-zinc-500">
      {clusters.length} / {total}
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if loadingMore}loading more…{:else if hasMore}scroll for more{:else}all loaded{/if}
    </span>
  </div>
</div>
