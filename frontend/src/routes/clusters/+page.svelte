<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { getClusters, getPlates, getThumbUrl, type PlateBrowseItem } from '$lib/api';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import PlateCard from '$lib/components/PlateCard.svelte';
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

  // --- Plate browse (replaces the "License plates aren't clustered" placeholder
  //     when the operator selects the license_plate class filter).
  let plates = $state<PlateBrowseItem[]>([]);
  let platesTotal = $state<number>(0);
  let platesLoading = $state<boolean>(false);
  let platesPage = $state<number>(1);
  let platesError = $state<string | null>(null);
  const platesHasMore = $derived(plates.length < platesTotal);
  const PLATES_PAGE_SIZE = 60;

  // Filter sidebar state — only active on the plates view.
  let plateDetectorFilter = $state<string>('');
  let plateVerifiedOnly = $state<boolean>(false);
  let plateMinScore = $state<number>(0);
  let plateTextQuery = $state<string>('');

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

  // Re-load on filter / sort change — but only for the cluster view.
  // The plate browse view has its own loader keyed on its own params.
  $effect(() => {
    void classFilter;
    void sort;
    if (!isLicensePlateFilter) void loadFirst();
  });

  async function loadPlatesFirst(): Promise<void> {
    platesLoading = true;
    platesError = null;
    plates = [];
    platesTotal = 0;
    platesPage = 1;
    try {
      const res = await getPlates({
        page: 1,
        page_size: PLATES_PAGE_SIZE,
        detector: plateDetectorFilter || undefined,
        verified: plateVerifiedOnly || undefined,
        min_score: plateMinScore > 0 ? plateMinScore : undefined,
        text: plateTextQuery || undefined,
      });
      plates = res.items;
      platesTotal = res.total;
    } catch (e) {
      platesError = (e as Error).message;
    } finally {
      platesLoading = false;
    }
  }

  async function loadPlatesMore(): Promise<void> {
    if (platesLoading || !platesHasMore) return;
    platesLoading = true;
    try {
      const next = platesPage + 1;
      const res = await getPlates({
        page: next,
        page_size: PLATES_PAGE_SIZE,
        detector: plateDetectorFilter || undefined,
        verified: plateVerifiedOnly || undefined,
        min_score: plateMinScore > 0 ? plateMinScore : undefined,
        text: plateTextQuery || undefined,
      });
      const seen = new Set(plates.map((p) => p.crop_id));
      plates = [...plates, ...res.items.filter((p) => !seen.has(p.crop_id))];
      platesTotal = res.total;
      platesPage = next;
    } catch (e) {
      platesError = (e as Error).message;
    } finally {
      platesLoading = false;
    }
  }

  // Re-load plates whenever the filter sidebar values OR the plates-mode flag change.
  $effect(() => {
    void plateDetectorFilter;
    void plateVerifiedOnly;
    void plateMinScore;
    void plateTextQuery;
    if (isLicensePlateFilter) void loadPlatesFirst();
  });

  function openPlateInReview(p: PlateBrowseItem): void {
    void goto(`/review?tab=plates&crop_id=${encodeURIComponent(p.crop_id)}`);
  }

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
      <!-- Plates list view — backed by /curation/plates. Plates live as a
           plate_bbox_norm sub-bbox on each vehicle crop (not as their
           own cluster docs), so this view surfaces them directly with
           detector provenance + OCR text chips. -->
      <div class="flex min-h-0 flex-col gap-3">
        <!-- Filter strip: detector / verified / score / text-search.
             Same layout convention as the cluster sidebar so operators
             flip between modes without re-learning. -->
        <div
          class="flex flex-wrap items-center gap-3 rounded-md border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-xs"
        >
          <label class="flex items-center gap-1.5">
            <span class="text-zinc-400">Detector</span>
            <select
              bind:value={plateDetectorFilter}
              class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
            >
              <option value="">any</option>
              <option value="lpr_nanov11_640">LPR</option>
              <option value="sam3">SAM3</option>
              <option value="paddleocr_det_trt">Paddle det</option>
              <option value="human">Human</option>
            </select>
          </label>
          <label class="flex items-center gap-1.5">
            <input type="checkbox" bind:checked={plateVerifiedOnly} class="accent-blue-500" />
            <span class="text-zinc-400">Verified only</span>
          </label>
          <label class="flex items-center gap-1.5">
            <span class="text-zinc-400">Min score</span>
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              bind:value={plateMinScore}
              class="w-16 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
            />
          </label>
          <label class="flex items-center gap-1.5">
            <span class="text-zinc-400">Text</span>
            <input
              type="text"
              bind:value={plateTextQuery}
              placeholder="e.g. S14"
              class="w-28 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100 focus:border-blue-500 focus:outline-none"
            />
          </label>
          <span class="grow"></span>
          <span class="font-mono text-[11px] text-zinc-500">
            {plates.length.toLocaleString()} / {platesTotal.toLocaleString()} plates
          </span>
        </div>

        {#if platesError}
          <p class="text-sm text-red-300">API unavailable: {platesError}</p>
        {:else if platesLoading && plates.length === 0}
          <p class="text-sm text-zinc-500">Loading plates...</p>
        {:else if plates.length === 0}
          <p class="text-sm text-zinc-500">
            No plates match the current filters. The re-detection drain
            may still be populating provenance — fresh rows appear here
            as the worker processes them.
          </p>
        {:else}
          <div
            class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6"
            use:infiniteScroll={{ onload: loadPlatesMore, disabled: platesLoading || !platesHasMore }}
          >
            {#each plates as p (p.crop_id)}
              <PlateCard crop={p} onclick={openPlateInReview} />
            {/each}
          </div>
          {#if platesLoading}
            <p class="py-2 text-center text-xs text-zinc-500">Loading more…</p>
          {/if}
        {/if}
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
