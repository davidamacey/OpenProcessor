<script lang="ts">
  /**
   * Plates list view for /clusters, backed by {API_PREFIX}/regions — extracted
   * verbatim from clusters/+page.svelte's `{:else if isSlotFilter}`
   * template branch (P2.6, docs/genericization-plan-2026-09-13.md
   * §3.4/§5a). All state/logic lives in the injected `gallery` controller
   * (`slotGalleryController.svelte.ts`); this component is rendering
   * only, unchanged from what the route used to inline.
   *
   * Deliberately NOT parameterized (P2.6 is a verbatim move; P2.7
   * parameterizes by slot). Every class name, string, and behavior below
   * is identical to before the extraction.
   */
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { resolveApiUrl } from '$lib/api';
  import SlotBboxEditor from '$lib/components/SlotBboxEditor.svelte';
  import SlotCard from '$lib/components/SlotCard.svelte';
  import {
    FALSE_POSITIVE_REGION_CLUSTER_ID,
    type SlotGalleryController,
  } from '../../../routes/clusters/slotGalleryController.svelte';
  import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';
  import { regionStatusesStore } from '$stores/regionStatuses.svelte';

  interface Props {
    gallery: SlotGalleryController;
  }

  let { gallery }: Props = $props();

  const label = $derived(gallery.slot.label);
</script>

<!-- Plates list view — backed by {API_PREFIX}/regions. Plates live as a
     region_bbox_norm sub-bbox on each vehicle crop (not as their
     own cluster docs), so this view surfaces them directly with
     detector provenance + OCR text chips. -->
<div class="flex min-h-0 flex-col gap-3">
  <!-- Sticky header: the filter strip + bulk-action toolbar stay pinned
       to the top of the scroll area, so the verify / false-positive /
       no-plate controls remain reachable while scrolling deep into a
       bucket or sub-cluster. -mx-4/-mt-4 cancels the scroll container's
       p-4 so it spans edge-to-edge and pins at the very top. -->
  <div
    class="sticky top-0 z-20 -mx-4 -mt-4 flex flex-col gap-3 border-b border-zinc-800 bg-zinc-950 px-4 pt-4 pb-3"
  >
    <!-- Filter strip: detector / verified / score / text-search. -->
    <div
      class="flex flex-wrap items-center gap-3 rounded-md border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-xs"
    >
      <label class="flex items-center gap-1.5">
        <span class="text-zinc-400">Detector</span>
        <!-- Options are the served vocabulary's filterable detectors
             (`GET {API_PREFIX}/regions/vocabulary`, W0 finding m9) — the exact
             values that can appear in stored region_detector for this
             deployment, rather than a hardcoded model-id list. -->
        <select bind:value={gallery.detectorFilter} class="select-sm">
          <option value="">any</option>
          {#each regionVocabularyStore.filterableDetectors as d (d.id)}
            <option value={d.id}>{d.label}</option>
          {/each}
        </select>
      </label>
      <label class="flex items-center gap-1.5">
        <input
          type="checkbox"
          bind:checked={gallery.verifiedOnly}
          class="accent-blue-500"
        />
        <span class="text-zinc-400">Verified only</span>
      </label>
      <label class="flex items-center gap-1.5">
        <span class="text-zinc-400">Status</span>
        <!-- dq-region (2026-09-24): backed by GET {API_PREFIX}/regions?status=
             (400 on an unknown value) — options are the served status
             vocabulary (GET {API_PREFIX}/regions/statuses), not a hardcoded
             list, so a status this deployment doesn't have never
             appears. Includes verify_rejected (candidate-only rows,
             kept for reversal) alongside detected/no_region_visible/etc. -->
        <select bind:value={gallery.statusFilter} class="select-sm">
          <option value="">any</option>
          {#each regionStatusesStore.list as s (s.value)}
            <option value={s.value}>{s.label}</option>
          {/each}
        </select>
      </label>
      <label class="flex items-center gap-1.5">
        <span class="text-zinc-400">Min score</span>
        <input
          type="number"
          min="0"
          max="1"
          step="0.05"
          bind:value={gallery.minScore}
          class="input-sm w-16"
        />
      </label>
      {#if gallery.slot.capabilities.queue?.textFilter}
        {@const textFilter = gallery.slot.capabilities.queue.textFilter}
        <label class="flex items-center gap-1.5">
          <span class="text-zinc-400">{textFilter.label}</span>
          <input
            type="text"
            bind:value={gallery.textQuery}
            placeholder={textFilter.placeholder}
            class="input-sm w-28"
          />
        </label>
      {/if}

      <!-- Top-N largest-crop gate. The sort runs on the largest 1-3
         crops, so this is the key filter for the plates that matter. -->
      <div class="inline-flex overflow-hidden rounded border border-zinc-700">
        {#each [{ v: null, l: 'All' }, { v: 1, l: 'Largest' }, { v: 2, l: '+2nd' }, { v: 3, l: '+3rd' }] as o (o.l)}
          <button
            type="button"
            class="chip rounded-none border-0 {gallery.maxRank === o.v
              ? 'bg-blue-600 text-white'
              : 'bg-zinc-900 text-zinc-300 hover:bg-zinc-700'}"
            onclick={() => (gallery.maxRank = o.v as number | null)}
          >
            {o.l}
          </button>
        {/each}
      </div>

      {#if gallery.suspectedFpView}
        <button
          type="button"
          class="btn-sm border border-zinc-600 bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
          onclick={gallery.backToClusters}
        >
          ← Clusters
        </button>
        <span class="font-medium text-red-200">Suspected false positives</span>
        <label class="flex items-center gap-1 text-[11px] text-zinc-400">
          ≤
          <input
            type="number"
            step="0.05"
            min="0"
            max="2"
            bind:value={gallery.suspectedFpThreshold}
            class="w-16 rounded border border-zinc-700 bg-zinc-900 px-1 py-0.5 text-zinc-200"
          />
        </label>
        <button
          type="button"
          disabled={gallery.clusterBusy}
          class="btn-sm border border-red-500/50 bg-red-500/20 text-red-100 hover:bg-red-500/30 disabled:opacity-50"
          onclick={gallery.loadSuspectedFp}
        >
          {gallery.clusterBusy ? 'Loading…' : 'Reload'}
        </button>
      {:else if gallery.viewingAll}
        <button
          type="button"
          class="btn-sm border border-zinc-600 bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
          onclick={gallery.backToClusters}
        >
          ← Clusters
        </button>
        <span class="font-medium text-zinc-200">All {label.plural}</span>
      {:else if gallery.selectedCluster == null}
        <button
          type="button"
          disabled={gallery.clusterBusy}
          class="btn-sm border border-purple-500/50 bg-purple-500/20 text-purple-100 hover:bg-purple-500/30 disabled:opacity-50"
          onclick={gallery.runClustering}
          title="Group {label.plural} by visual similarity so outliers/false-positives surface"
        >
          {gallery.clusterBusy ? 'Clustering…' : `⟳ Cluster ${label.plural}`}
        </button>
        <button
          type="button"
          disabled={gallery.clusterBusy}
          class="btn-sm border border-red-500/50 bg-red-500/20 text-red-100 hover:bg-red-500/30 disabled:opacity-50"
          onclick={gallery.loadSuspectedFp}
          title="List {label.singular} crops that look like known false positives (needs FP centroids built)"
        >
          Suspected FPs
        </button>
        <button
          type="button"
          disabled={gallery.clusterBusy}
          class="btn-sm border border-amber-500/50 bg-amber-500/20 text-amber-100 hover:bg-amber-500/30 disabled:opacity-50"
          onclick={gallery.runBuildFpCentroids}
          title="Sub-type the false-positive bucket and (re)build its centroids"
        >
          {gallery.clusterBusy ? 'Building…' : 'Build FP centroids'}
        </button>
        <!-- M3: plate clusters only cover plates that have gone through
             "Cluster plates" — before that (or for plates the run left
             out) the bucket grid below has no card for them at all, so
             this is the only way in. Always shown here, not gated on
             "no non-FP clusters exist", so it stays reachable once
             clustering starts covering only part of the pool too. -->
        <button
          type="button"
          class="btn-sm border border-blue-500/50 bg-blue-500/20 text-blue-100 hover:bg-blue-500/30"
          onclick={gallery.openAll}
          title="Browse every {label.singular}, including ones not in any cluster bucket yet"
        >
          Browse all {label.plural}
        </button>
      {:else}
        <button
          type="button"
          class="btn-sm border border-zinc-600 bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
          onclick={gallery.backToClusters}
        >
          ← Clusters
        </button>
        {#if gallery.selectedCluster === FALSE_POSITIVE_REGION_CLUSTER_ID}
          <span
            class="rounded bg-red-500/25 px-2 py-0.5 text-[11px] font-semibold tracking-wide text-red-200 uppercase"
          >
            ✗ False-positive cluster
          </span>
          <span class="text-[11px] text-zinc-400"
            >not {label.plural} — hard negatives for the detector</span
          >
          <button
            type="button"
            disabled={gallery.clusterBusy}
            class="btn-sm border border-amber-500/50 bg-amber-500/20 text-amber-100 hover:bg-amber-500/30 disabled:opacity-50"
            onclick={gallery.runBuildFpCentroids}
            title="Refine the FP bucket into sub-types and rebuild its centroids"
          >
            {gallery.clusterBusy ? 'Refining…' : 'Refine FP (build centroids)'}
          </button>
        {:else}
          <span class="font-mono text-[11px] text-zinc-300"
            >bucket #{gallery.selectedCluster}</span
          >
          <button
            type="button"
            disabled={gallery.clusterBusy}
            class="btn-sm border border-blue-500/50 bg-blue-500/20 text-blue-100 hover:bg-blue-500/30 disabled:opacity-50"
            onclick={gallery.runRefineCluster}
            title="AHC-refine this bucket into sub-clusters to isolate outliers"
          >
            {gallery.clusterBusy ? 'Refining…' : 'Refine AHC'}
          </button>
          {#if gallery.refineMsg}
            <span class="text-[11px] text-zinc-400">{gallery.refineMsg}</span>
          {/if}
        {/if}
      {/if}
      <span class="grow"></span>
      {#if gallery.pager.items.length > 0}
        <button
          type="button"
          class="btn-sm border border-zinc-700 bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
          onclick={gallery.selectAll}
          title="Select all loaded {label.plural} (shift-click a card for a range, ctrl/cmd-click to toggle)"
        >
          Select all
        </button>
      {/if}
      <span class="font-mono text-[11px] text-zinc-500">
        {gallery.pager.items.length.toLocaleString()} / {gallery.pager.total.toLocaleString()}
      </span>
    </div>

    <!-- Bulk-action toolbar — appears when plates are selected. Triage
       outliers without leaving the gallery (no /review round-trip). -->
    {#if gallery.sel.size > 0}
      <div
        class="flex flex-wrap items-center gap-2 rounded-md border border-blue-500/40 bg-blue-500/10 px-3 py-2 text-xs"
      >
        <span class="font-medium text-blue-200">{gallery.sel.size} selected</span>
        <span class="grow"></span>
        <button
          type="button"
          disabled={gallery.busy}
          class="btn-sm border border-red-500/50 bg-red-500/20 text-red-200 hover:bg-red-500/30 disabled:opacity-50"
          onclick={() =>
            gallery.applyStatus([...gallery.sel.ids], gallery.falsePositiveState())}
        >
          ✗ Mark false positive
        </button>
        <button
          type="button"
          disabled={gallery.busy}
          class="btn-sm border border-zinc-600 bg-zinc-800 text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
          onclick={() => gallery.applyStatus([...gallery.sel.ids], gallery.rejectState())}
        >
          No {label.singular}
        </button>
        <button
          type="button"
          disabled={gallery.busy}
          class="btn-sm border border-green-500/50 bg-green-500/20 text-green-200 hover:bg-green-500/30 disabled:opacity-50"
          onclick={() =>
            gallery.applyStatus([...gallery.sel.ids], gallery.confirmState())}
        >
          ✓ Verify
        </button>
        <button
          type="button"
          class="btn-sm border border-zinc-700 text-zinc-400 hover:bg-zinc-800"
          onclick={() => gallery.sel.clear()}
        >
          Clear
        </button>
      </div>
    {/if}
  </div>
  <!-- /sticky header -->

  {#if gallery.pager.error}
    <p class="text-sm text-red-300">API unavailable: {gallery.pager.error}</p>
  {:else if !gallery.suspectedFpView && !gallery.viewingAll && gallery.selectedCluster == null && gallery.clusters.length > 0}
    <!-- Plate cluster cards. Click one to open its plates (with the
         bulk toolbar + AHC Refine). Buckets with sub-clusters (refined)
         get a blue border so refined buckets are easy to spot. The
         permanent false-positive bucket gets a red border + label. -->
    <ul
      class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6"
    >
      {#each gallery.clusters as c (c.id)}
        <li style="content-visibility:auto;contain-intrinsic-size:auto 200px">
          <button
            type="button"
            class="flex w-full flex-col rounded-md border-2 bg-zinc-900 text-left transition hover:border-zinc-300 {c.cluster_kind ===
            'false_positive'
              ? 'border-red-500/70'
              : c.has_subclusters
                ? 'border-blue-500/60'
                : 'border-zinc-700'}"
            onclick={() => gallery.openCluster(c.id)}
          >
            <div class="grid grid-cols-2 gap-px overflow-hidden rounded-t bg-zinc-950">
              {#each c.representative_thumb_urls?.slice(0, 4) ?? [] as url, i (i)}
                <img
                  src={resolveApiUrl(url)}
                  alt={label.singular}
                  loading="lazy"
                  class="aspect-[2/1] w-full bg-zinc-950 object-contain"
                />
              {/each}
            </div>
            <div class="flex items-center justify-between gap-1 p-2 text-xs">
              {#if c.cluster_kind === 'false_positive'}
                <span
                  class="rounded bg-red-500/25 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-red-200 uppercase"
                  title="Permanent false-positive bucket — these are NOT {label.plural}"
                >
                  ✗ False positives
                </span>
              {:else}
                <span class="font-semibold text-zinc-200">#{c.id}</span>
              {/if}
              <span class="text-zinc-400">{c.size.toLocaleString()}</span>
              {#if c.n_subclusters > 0}
                <span
                  class="rounded bg-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-200"
                  >{c.n_subclusters} sub</span
                >
              {/if}
            </div>
          </button>
        </li>
      {/each}
    </ul>
  {:else if gallery.pager.loading && gallery.pager.items.length === 0}
    <p class="text-sm text-zinc-500">Loading {label.plural}...</p>
  {:else if gallery.pager.items.length === 0}
    <p class="text-sm text-zinc-500">
      No {label.plural} match the current filters. The re-detection drain may still be populating
      provenance — fresh rows appear here as the worker processes them.
    </p>
  {:else}
    <!-- Sub-cluster tabs: appear once a bucket has been AHC-refined.
         "All" shows the grouped view (separators per sub-cluster);
         clicking a chip filters to that one sub-cluster. -->
    {#if gallery.selectedCluster != null && gallery.subclusterIds.length > 0}
      <div class="mb-3 flex flex-wrap items-center gap-1.5">
        <span class="text-[11px] text-zinc-500">sub-clusters:</span>
        <button
          type="button"
          class="chip {gallery.subTab === null
            ? 'bg-blue-500/30 text-blue-100'
            : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
          onclick={() => gallery.selectSubTab(null)}
        >
          all
        </button>
        {#each gallery.subclusterIds as sid (sid)}
          <button
            type="button"
            class="chip font-mono {gallery.subTab === sid
              ? 'bg-blue-500/30 text-blue-100'
              : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
            onclick={() => gallery.selectSubTab(sid)}
          >
            {sid}
            <span class="text-zinc-500">{gallery.subCounts.get(sid) ?? ''}</span>
          </button>
        {/each}
      </div>
    {/if}

    {#each gallery.groups as g (g.key)}
      {#if g.label}
        <div class="mt-3 mb-1.5 flex items-center gap-2">
          <span class="font-mono text-[11px] text-zinc-300">{g.label}</span>
          <span class="text-[11px] text-zinc-500">{g.items.length}</span>
          <span class="h-px grow bg-zinc-800"></span>
        </div>
      {/if}
      <div
        class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6"
      >
        {#each g.items as p (p.crop_id)}
          <SlotCard
            crop={p}
            slot={gallery.slot}
            selected={gallery.sel.has(p.crop_id)}
            onclick={gallery.toggleSelect}
            onedit={gallery.openEditor}
            onmarkfp={(c) =>
              gallery.applyStatus([c.crop_id], gallery.falsePositiveState())}
          />
        {/each}
      </div>
    {/each}
    <!-- Sentinel AFTER the grid (not on it): the observer must root on a
         small element that only intersects once the user scrolls to the
         bottom. Attaching to the tall grid itself keeps it permanently
         intersecting and loads every page at once. -->
    <div
      use:infiniteScroll={{
        onload: gallery.loadMore,
        disabled:
          gallery.pager.loading || gallery.pager.loadingMore || !gallery.pager.hasMore,
      }}
      class="mt-4 h-1"
      aria-hidden="true"
    ></div>
    {#if gallery.pager.loadingMore}
      <p class="py-2 text-center text-xs text-zinc-500">Loading more…</p>
    {/if}
  {/if}
</div>

{#if gallery.editCrop}
  <SlotBboxEditor
    crop={gallery.editCrop}
    slot={gallery.slot}
    onsave={gallery.saveBox}
    onclose={() => (gallery.editCrop = null)}
  />
{/if}
