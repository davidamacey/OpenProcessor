<script lang="ts">
  /**
   * Plates list view for /clusters, backed by /curation/plates — extracted
   * verbatim from clusters/+page.svelte's `{:else if isLicensePlateFilter}`
   * template branch (P2.6, docs/genericization-plan-2026-09-13.md
   * §3.4/§5a). All state/logic lives in the injected `gallery` controller
   * (`plateGalleryController.svelte.ts`); this component is rendering
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
    FP_PLATE_CLUSTER_ID,
    PLATE_CONFIRM_STATE,
    PLATE_REJECT_STATE,
    PLATE_FALSE_POSITIVE_STATE,
    type PlateGalleryController,
  } from '../../../routes/clusters/plateGalleryController.svelte';

  interface Props {
    gallery: PlateGalleryController;
  }

  let { gallery }: Props = $props();
</script>

<!-- Plates list view — backed by /curation/plates. Plates live as a
     plate_bbox_norm sub-bbox on each vehicle crop (not as their
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
        <select bind:value={gallery.plateDetectorFilter} class="select-sm">
          <option value="">any</option>
          <option value="lpr_nanov11_640">LPR</option>
          <option value="sam3">SAM3</option>
          <option value="paddleocr_det_trt">Paddle det</option>
          <option value="human">Human</option>
        </select>
      </label>
      <label class="flex items-center gap-1.5">
        <input
          type="checkbox"
          bind:checked={gallery.plateVerifiedOnly}
          class="accent-blue-500"
        />
        <span class="text-zinc-400">Verified only</span>
      </label>
      <label class="flex items-center gap-1.5">
        <span class="text-zinc-400">Min score</span>
        <input
          type="number"
          min="0"
          max="1"
          step="0.05"
          bind:value={gallery.plateMinScore}
          class="input-sm w-16"
        />
      </label>
      <label class="flex items-center gap-1.5">
        <span class="text-zinc-400">Text</span>
        <input
          type="text"
          bind:value={gallery.plateTextQuery}
          placeholder="e.g. S14"
          class="input-sm w-28"
        />
      </label>

      <!-- Top-N largest-crop gate. The sort runs on the largest 1-3
         crops, so this is the key filter for the plates that matter. -->
      <div class="inline-flex overflow-hidden rounded border border-zinc-700">
        {#each [{ v: null, l: 'All' }, { v: 1, l: 'Largest' }, { v: 2, l: '+2nd' }, { v: 3, l: '+3rd' }] as o (o.l)}
          <button
            type="button"
            class="chip rounded-none border-0 {gallery.plateMaxRank === o.v
              ? 'bg-blue-600 text-white'
              : 'bg-zinc-900 text-zinc-300 hover:bg-zinc-700'}"
            onclick={() => (gallery.plateMaxRank = o.v as number | null)}
          >
            {o.l}
          </button>
        {/each}
      </div>

      {#if gallery.suspectedFpView}
        <button
          type="button"
          class="btn-sm border border-zinc-600 bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
          onclick={gallery.backToPlateClusters}
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
          disabled={gallery.plateClusterBusy}
          class="btn-sm border border-red-500/50 bg-red-500/20 text-red-100 hover:bg-red-500/30 disabled:opacity-50"
          onclick={gallery.loadSuspectedFp}
        >
          {gallery.plateClusterBusy ? 'Loading…' : 'Reload'}
        </button>
      {:else if gallery.selectedPlateCluster == null}
        <button
          type="button"
          disabled={gallery.plateClusterBusy}
          class="btn-sm border border-purple-500/50 bg-purple-500/20 text-purple-100 hover:bg-purple-500/30 disabled:opacity-50"
          onclick={gallery.runClusterPlates}
          title="Group plates by visual similarity so outliers/false-positives surface"
        >
          {gallery.plateClusterBusy ? 'Clustering…' : '⟳ Cluster plates'}
        </button>
        <button
          type="button"
          disabled={gallery.plateClusterBusy}
          class="btn-sm border border-red-500/50 bg-red-500/20 text-red-100 hover:bg-red-500/30 disabled:opacity-50"
          onclick={gallery.loadSuspectedFp}
          title="List plate crops that look like known false positives (needs FP centroids built)"
        >
          Suspected FPs
        </button>
        <button
          type="button"
          disabled={gallery.plateClusterBusy}
          class="btn-sm border border-amber-500/50 bg-amber-500/20 text-amber-100 hover:bg-amber-500/30 disabled:opacity-50"
          onclick={gallery.runBuildFpCentroids}
          title="Sub-type the false-positive bucket and (re)build its centroids"
        >
          {gallery.plateClusterBusy ? 'Building…' : 'Build FP centroids'}
        </button>
      {:else}
        <button
          type="button"
          class="btn-sm border border-zinc-600 bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
          onclick={gallery.backToPlateClusters}
        >
          ← Clusters
        </button>
        {#if gallery.selectedPlateCluster === FP_PLATE_CLUSTER_ID}
          <span
            class="rounded bg-red-500/25 px-2 py-0.5 text-[11px] font-semibold tracking-wide text-red-200 uppercase"
          >
            ✗ False-positive cluster
          </span>
          <span class="text-[11px] text-zinc-400"
            >not plates — hard negatives for LPR</span
          >
          <button
            type="button"
            disabled={gallery.plateClusterBusy}
            class="btn-sm border border-amber-500/50 bg-amber-500/20 text-amber-100 hover:bg-amber-500/30 disabled:opacity-50"
            onclick={gallery.runBuildFpCentroids}
            title="Refine the FP bucket into sub-types and rebuild its centroids"
          >
            {gallery.plateClusterBusy ? 'Refining…' : 'Refine FP (build centroids)'}
          </button>
        {:else}
          <span class="font-mono text-[11px] text-zinc-300"
            >bucket #{gallery.selectedPlateCluster}</span
          >
          <button
            type="button"
            disabled={gallery.plateClusterBusy}
            class="btn-sm border border-blue-500/50 bg-blue-500/20 text-blue-100 hover:bg-blue-500/30 disabled:opacity-50"
            onclick={gallery.runRefinePlateCluster}
            title="AHC-refine this bucket into sub-clusters to isolate outliers"
          >
            {gallery.plateClusterBusy ? 'Refining…' : 'Refine AHC'}
          </button>
          {#if gallery.plateRefineMsg}
            <span class="text-[11px] text-zinc-400">{gallery.plateRefineMsg}</span>
          {/if}
        {/if}
      {/if}
      <span class="grow"></span>
      {#if gallery.platePager.items.length > 0}
        <button
          type="button"
          class="btn-sm border border-zinc-700 bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
          onclick={gallery.selectAllPlates}
          title="Select all loaded plates (shift-click a card for a range, ctrl/cmd-click to toggle)"
        >
          Select all
        </button>
      {/if}
      <span class="font-mono text-[11px] text-zinc-500">
        {gallery.platePager.items.length.toLocaleString()} / {gallery.platePager.total.toLocaleString()}
        gallery.platePager.items
      </span>
    </div>

    <!-- Bulk-action toolbar — appears when plates are selected. Triage
       outliers without leaving the gallery (no /review round-trip). -->
    {#if gallery.plateSel.size > 0}
      <div
        class="flex flex-wrap items-center gap-2 rounded-md border border-blue-500/40 bg-blue-500/10 px-3 py-2 text-xs"
      >
        <span class="font-medium text-blue-200">{gallery.plateSel.size} selected</span>
        <span class="grow"></span>
        <button
          type="button"
          disabled={gallery.plateBusy}
          class="btn-sm border border-red-500/50 bg-red-500/20 text-red-200 hover:bg-red-500/30 disabled:opacity-50"
          onclick={() =>
            gallery.applyPlateStatus(
              [...gallery.plateSel.ids],
              PLATE_FALSE_POSITIVE_STATE,
            )}
        >
          ✗ Mark false positive
        </button>
        <button
          type="button"
          disabled={gallery.plateBusy}
          class="btn-sm border border-zinc-600 bg-zinc-800 text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
          onclick={() =>
            gallery.applyPlateStatus([...gallery.plateSel.ids], PLATE_REJECT_STATE)}
        >
          No plate
        </button>
        <button
          type="button"
          disabled={gallery.plateBusy}
          class="btn-sm border border-green-500/50 bg-green-500/20 text-green-200 hover:bg-green-500/30 disabled:opacity-50"
          onclick={() =>
            gallery.applyPlateStatus([...gallery.plateSel.ids], PLATE_CONFIRM_STATE)}
        >
          ✓ Verify
        </button>
        <button
          type="button"
          class="btn-sm border border-zinc-700 text-zinc-400 hover:bg-zinc-800"
          onclick={() => gallery.plateSel.clear()}
        >
          Clear
        </button>
      </div>
    {/if}
  </div>
  <!-- /sticky header -->

  {#if gallery.platePager.error}
    <p class="text-sm text-red-300">API unavailable: {gallery.platePager.error}</p>
  {:else if !gallery.suspectedFpView && gallery.selectedPlateCluster == null && gallery.plateClusters.length > 0}
    <!-- Plate cluster cards. Click one to open its plates (with the
         bulk toolbar + AHC Refine). Buckets with sub-clusters (refined)
         get a blue border so refined buckets are easy to spot. The
         permanent false-positive bucket gets a red border + label. -->
    <ul
      class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6"
    >
      {#each gallery.plateClusters as c (c.id)}
        <li style="content-visibility:auto;contain-intrinsic-size:auto 200px">
          <button
            type="button"
            class="flex w-full flex-col rounded-md border-2 bg-zinc-900 text-left transition hover:border-zinc-300 {c.cluster_kind ===
            'false_positive'
              ? 'border-red-500/70'
              : c.has_subclusters
                ? 'border-blue-500/60'
                : 'border-zinc-700'}"
            onclick={() => gallery.openPlateCluster(c.id)}
          >
            <div class="grid grid-cols-2 gap-px overflow-hidden rounded-t bg-zinc-950">
              {#each c.representative_thumb_urls?.slice(0, 4) ?? [] as url, i (i)}
                <img
                  src={resolveApiUrl(url)}
                  alt="plate"
                  loading="lazy"
                  class="aspect-[2/1] w-full bg-zinc-950 object-contain"
                />
              {/each}
            </div>
            <div class="flex items-center justify-between gap-1 p-2 text-xs">
              {#if c.cluster_kind === 'false_positive'}
                <span
                  class="rounded bg-red-500/25 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-red-200 uppercase"
                  title="Permanent false-positive bucket — these are NOT plates"
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
  {:else if gallery.platePager.loading && gallery.platePager.items.length === 0}
    <p class="text-sm text-zinc-500">Loading plates...</p>
  {:else if gallery.platePager.items.length === 0}
    <p class="text-sm text-zinc-500">
      No plates match the current filters. The re-detection drain may still be populating
      provenance — fresh rows appear here as the worker processes them.
    </p>
  {:else}
    <!-- Sub-cluster tabs: appear once a bucket has been AHC-refined.
         "All" shows the grouped view (separators per sub-cluster);
         clicking a chip filters to that one sub-cluster. -->
    {#if gallery.selectedPlateCluster != null && gallery.plateSubclusterIds.length > 0}
      <div class="mb-3 flex flex-wrap items-center gap-1.5">
        <span class="text-[11px] text-zinc-500">sub-clusters:</span>
        <button
          type="button"
          class="chip {gallery.plateSubTab === null
            ? 'bg-blue-500/30 text-blue-100'
            : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
          onclick={() => gallery.selectPlateSubTab(null)}
        >
          all
        </button>
        {#each gallery.plateSubclusterIds as sid (sid)}
          <button
            type="button"
            class="chip font-mono {gallery.plateSubTab === sid
              ? 'bg-blue-500/30 text-blue-100'
              : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
            onclick={() => gallery.selectPlateSubTab(sid)}
          >
            {sid}
            <span class="text-zinc-500">{gallery.plateSubCounts.get(sid) ?? ''}</span>
          </button>
        {/each}
      </div>
    {/if}

    {#each gallery.plateGroups as g (g.key)}
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
            selected={gallery.plateSel.has(p.crop_id)}
            onclick={gallery.togglePlateSelect}
            onedit={gallery.openPlateEditor}
            onmarkfp={(c) =>
              gallery.applyPlateStatus([c.crop_id], PLATE_FALSE_POSITIVE_STATE)}
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
        onload: gallery.loadPlatesMore,
        disabled:
          gallery.platePager.loading ||
          gallery.platePager.loadingMore ||
          !gallery.platePager.hasMore,
      }}
      class="mt-4 h-1"
      aria-hidden="true"
    ></div>
    {#if gallery.platePager.loadingMore}
      <p class="py-2 text-center text-xs text-zinc-500">Loading more…</p>
    {/if}
  {/if}
</div>

{#if gallery.editPlateCrop}
  <SlotBboxEditor
    crop={gallery.editPlateCrop}
    onsave={gallery.savePlateBbox}
    onclose={() => (gallery.editPlateCrop = null)}
  />
{/if}
