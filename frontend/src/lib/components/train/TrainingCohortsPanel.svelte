<script lang="ts">
  /** /train's training-cohort chips per class with the inline preview grid. */
  import SlotCard from '$components/SlotCard.svelte';
  import CropCard from '$components/CropCard.svelte';
  import type { RegionBrowseItem } from '$lib/api_regions';
  import type { Crop } from '$lib/types';
  import type { TrainCohortsController } from '$lib/train/trainCohortsController.svelte';

  interface Props {
    ctl: TrainCohortsController;
  }

  let { ctl }: Props = $props();
</script>

<!-- Training cohorts —
     GET {API_PREFIX}/training_cohorts?class_id=
     serves both the region profile's cohorts AND every other
     class's 4 class-agnostic core cohorts; a tier-2 slot's own
     declared cohort fills in only if the server didn't already send
     that id (see loadGroupCohorts). Grouped by class so "pick the
     classes you want to train and it goes" reads directly off the
     screen. Selecting a chip
     loads a 24-card sanity-preview grid. -->
<section class="rounded-md border border-zinc-800 bg-zinc-900">
  <header
    class="flex items-center justify-between gap-3 border-b border-zinc-800 px-3 py-2"
  >
    <div class="flex flex-col">
      <h2 class="text-sm font-semibold text-zinc-100">Training cohorts</h2>
      <p class="text-[11px] text-zinc-500">
        Curation-preview slices for the next training cycle, grouped by class. Pick a chip
        to preview a sanity grid before committing.
      </p>
    </div>
    <button
      type="button"
      class="btn"
      onclick={() => void ctl.refreshCohortCounts()}
      title="Refresh cohort counts"
    >
      Refresh
    </button>
  </header>
  {#each ctl.visibleCohortGroups as group (group.classId)}
    <div
      class="border-b border-zinc-800 p-3 last:border-b-0"
      use:ctl.lazyLoadGroupCounts={group}
    >
      <h3 class="mb-2 text-xs font-semibold text-zinc-300">{group.className}</h3>
      <div class="flex flex-wrap gap-1.5">
        {#each group.cohorts as cohort (cohort.id)}
          {@const key = ctl.cohortKey(group.classId, cohort)}
          {@const count = ctl.cohortCounts[key]}
          {@const selected = ctl.selectedCohortKey === key}
          <button
            type="button"
            class="flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors
                   {selected
              ? 'border-blue-500 bg-blue-500/10 text-blue-100'
              : 'border-zinc-700 bg-zinc-950 text-zinc-300 hover:border-blue-500/50'}"
            onclick={() => void ctl.loadCohortPreview(group, cohort)}
            title={cohort.description}
          >
            <span class="font-medium">{cohort.label}</span>
            <span
              class="font-mono text-[10px] {selected ? 'text-blue-200' : 'text-zinc-500'}"
            >
              {count == null ? '…' : count.toLocaleString()}
            </span>
          </button>
        {/each}
      </div>
      <!-- M14 (2026-09-24 interactive pass): the preview used to render
           once, after every one of the ~85 class groups — clicking a
           chip near the top left the preview grid roughly 4,700px
           below, off-screen with no scroll-into-view, so the click
           looked like a no-op. Rendering it inline, right under the
           group whose chip was clicked, means the preview always
           appears exactly where the operator is already looking. -->
      {#if ctl.selectedCohortKey}
        {@const activeCohort = group.cohorts.find(
          (c) => ctl.cohortKey(group.classId, c) === ctl.selectedCohortKey,
        )}
        {#if activeCohort}
          <div class="mt-3 border-t border-zinc-800 pt-3">
            {#if ctl.cohortPreviewError}
              <p class="text-xs text-red-300">Preview failed: {ctl.cohortPreviewError}</p>
            {:else if ctl.cohortPreviewLoading && ctl.cohortPreview.length === 0}
              <p class="text-xs text-zinc-500">Loading preview…</p>
            {:else if ctl.cohortPreview.length === 0}
              <p class="text-xs text-zinc-500">
                No rows match this cohort yet — the re-detection drain may still be
                populating provenance. Check back as the queue drains.
              </p>
            {:else}
              <div
                class="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8"
              >
                {#each ctl.cohortPreview as item ('row_key' in item ? item.row_key : item.id)}
                  {#if activeCohort.rowKind === 'slot'}
                    {@const cohortSlot = ctl.regionSlotForGroup(group)}
                    {#if cohortSlot}
                      <SlotCard
                        crop={item as RegionBrowseItem}
                        slot={cohortSlot}
                        onclick={(p) => ctl.openCohortItem(group, activeCohort, p)}
                        onreprocessed={() =>
                          void ctl.loadCohortPreview(group, activeCohort)}
                        compact
                      />
                    {/if}
                  {:else}
                    <CropCard
                      crop={item as Crop}
                      onclick={(c) => ctl.openCohortItem(group, activeCohort, c)}
                      onreprocessed={() =>
                        void ctl.loadCohortPreview(group, activeCohort)}
                    />
                  {/if}
                {/each}
              </div>
            {/if}
          </div>
        {/if}
      {/if}
    </div>
  {/each}
  {#if ctl.zeroCandidateGroups.length > 0}
    <!-- Collapsed by default: a class only lands here once every one
         of its cohorts SERVED a count of 0 (see isAllZeroLoaded) — a
         class still loading (or one this session hasn't scrolled to
         yet) always renders in the normal list above, never here. -->
    <details class="border-b border-zinc-800 p-3 last:border-b-0">
      <summary class="cursor-pointer text-xs text-zinc-500 hover:text-zinc-300">
        {ctl.zeroCandidateGroups.length} class{ctl.zeroCandidateGroups.length === 1
          ? ''
          : 'es'} with no candidates{#if ctl.cohortGroupsPending > 0}
          <span class="text-zinc-600" data-testid="zero-cohorts-so-far"
            >{` so far (${ctl.cohortGroupsPending} still loading)`}</span
          >{/if}
      </summary>
      <div class="mt-2 space-y-2">
        {#each ctl.zeroCandidateGroups as group (group.classId)}
          <div class="flex flex-wrap items-baseline gap-1.5">
            <h4 class="mr-1 text-[11px] font-semibold text-zinc-500">
              {group.className}
            </h4>
            {#each group.cohorts as cohort (cohort.id)}
              <span
                class="rounded-full border border-zinc-800 bg-zinc-950 px-2 py-0.5 text-[10px] text-zinc-600"
                title={cohort.description}
              >
                {cohort.label} <span class="font-mono">0</span>
              </span>
            {/each}
          </div>
        {/each}
      </div>
    </details>
  {/if}
</section>
