<script lang="ts">
  /**
   * /train's training cohorts: one compact table, a row per class and a column per cohort, every
   * count served by `GET /training_cohorts/summary` in a single request. A count opens that
   * class's preview of the cohort in a Drawer (never inline), so a hundred classes stay one
   * screen. Empty rows are hidden by default.
   */
  import { onMount } from 'svelte';
  import SlotCard from '$components/SlotCard.svelte';
  import CropCard from '$components/CropCard.svelte';
  import Button from '$components/ui/Button.svelte';
  import Card from '$components/ui/Card.svelte';
  import DataTable, { type Column } from '$components/ui/DataTable.svelte';
  import Drawer from '$components/ui/Drawer.svelte';
  import EmptyState from '$components/ui/EmptyState.svelte';
  import ErrorState from '$components/ui/ErrorState.svelte';
  import Skeleton from '$components/ui/Skeleton.svelte';
  import type { CohortRow } from '$lib/api_train';
  import type { RegionBrowseItem } from '$lib/api_regions';
  import type { Crop } from '$lib/types';
  import type { TrainCohortsController } from '$lib/train/trainCohortsController.svelte';

  interface Props {
    ctl: TrainCohortsController;
  }
  let { ctl }: Props = $props();

  onMount(() => {
    const abort = new AbortController();
    void ctl.load(abort.signal);
    return () => abort.abort();
  });

  const columns = $derived<Column<CohortRow>[]>([
    { id: 'class', label: 'Class', value: (r) => r.class_name },
    ...(ctl.summary?.cohorts ?? []).map((c) => ({
      id: c.id,
      label: c.label,
      title: c.description,
      align: 'right' as const,
      value: (r: CohortRow) => r.counts[c.id] ?? 0,
    })),
  ]);
  const fmt = (n: number) => n.toLocaleString();
  const hiddenText = $derived(
    `${ctl.hiddenEmpty} ${ctl.hiddenEmpty === 1 ? 'class' : 'classes'} with no candidates hidden.`,
  );
  const picked = $derived(ctl.selected);
</script>

<Card padded={false} data-testid="training-cohorts">
  <header class="flex flex-wrap items-center gap-3 border-b border-line px-4 py-3">
    <div class="min-w-0 grow">
      <h2 class="text-title font-semibold text-fg-1">Training cohorts</h2>
      <p class="text-meta text-fg-3">
        Slices of the dataset for the next training cycle, per class. Open a count to
        preview it.
      </p>
    </div>
    <input
      type="search"
      class="input w-48"
      placeholder="Filter classes"
      aria-label="Filter classes"
      bind:value={ctl.filter}
      data-testid="cohorts-filter"
    />
    <label class="flex items-center gap-1.5 text-control text-fg-2">
      <input
        type="checkbox"
        bind:checked={ctl.hideEmpty}
        data-testid="cohorts-hide-empty"
      />
      Hide empty
    </label>
    <Button
      size="sm"
      icon="refresh-cw"
      onclick={() => void ctl.load()}
      loading={ctl.loading}>Refresh</Button
    >
  </header>

  <div class="p-4">
    {#if ctl.failure}
      <ErrorState
        title="Training cohorts unavailable"
        error={ctl.failure}
        onretry={() => void ctl.load()}
      />
    {:else if !ctl.summary}
      <Skeleton lines={4} />
    {:else if ctl.summary.rows.length === 0}
      <EmptyState
        title="No classes yet"
        description="Cohorts are counted per class. Create classes first."
      />
    {:else}
      <DataTable
        {columns}
        rows={ctl.rows}
        rowKey={(r) => `${r.kind}:${r.class_id ?? r.class_name}`}
        label="Training cohorts by class"
        initialSort={{ column: 'class', dir: 'asc' }}
      >
        {#snippet cell(row, col)}
          {#if col.id === 'class'}
            <span class="text-fg-1">{row.class_name}</span>
            {#if row.kind === 'detector'}
              <span
                class="ml-1 text-caption text-fg-3"
                title="A label the detector named that the registry has no class for"
                >detector label</span
              >
            {/if}
          {:else}
            {@const column = ctl.summary!.cohorts.find((c) => c.id === col.id)!}
            {@const n = row.counts[col.id] ?? 0}
            {#if ctl.previewable(row, column)}
              <button
                type="button"
                class="text-accent-text underline-offset-2 hover:underline"
                onclick={() => void ctl.openPreview(row, column)}
                aria-label="Preview {fmt(n)} {column.label} for {row.class_name}"
                >{fmt(n)}</button
              >
            {:else}
              <span class={n === 0 ? 'text-fg-3' : 'text-fg-1'}>{fmt(n)}</span>
            {/if}
          {/if}
        {/snippet}
      </DataTable>
      {#if ctl.rows.length === 0}
        <p class="mt-3 text-meta text-fg-3" data-testid="cohorts-no-match">
          No class matches the filter.
        </p>
      {/if}
      {#if ctl.hiddenEmpty > 0}
        <p class="mt-2 text-meta text-fg-3" data-testid="cohorts-hidden">
          {hiddenText}
        </p>
      {/if}
    {/if}
  </div>
</Card>

{#if picked}
  <Drawer
    label="{picked.column.label}: {picked.row.class_name}"
    side="right"
    widthClass="w-[40rem]"
    onclose={ctl.closePreview}
  >
    <div class="p-4" data-testid="cohort-preview">
      <p class="mb-3 text-meta text-fg-3">{picked.column.description}</p>
      {#if ctl.cohortPreviewError}
        <ErrorState
          title="Preview failed"
          message={ctl.cohortPreviewError}
          onretry={() => void ctl.openPreview(picked.row, picked.column)}
        />
      {:else if ctl.cohortPreviewLoading && ctl.cohortPreview.length === 0}
        <Skeleton lines={3} />
      {:else if ctl.cohortPreview.length === 0}
        <p class="text-body text-fg-3">
          No rows match this cohort yet. The re-detection drain may still be populating
          provenance.
        </p>
      {:else}
        <div class="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {#each ctl.cohortPreview as item ('row_key' in item ? item.row_key : item.id)}
            {#if picked.column.row_kind === 'region'}
              {@const cohortSlot = ctl.regionSlotForGroup(picked.row)}
              {#if cohortSlot}
                <SlotCard
                  crop={item as RegionBrowseItem}
                  slot={cohortSlot}
                  onclick={(p) => ctl.openCohortItem(picked.row, picked.column, p)}
                  onreprocessed={() => void ctl.openPreview(picked.row, picked.column)}
                  compact
                />
              {/if}
            {:else}
              <CropCard
                crop={item as Crop}
                onclick={(c) => ctl.openCohortItem(picked.row, picked.column, c)}
                onreprocessed={() => void ctl.openPreview(picked.row, picked.column)}
              />
            {/if}
          {/each}
        </div>
      {/if}
    </div>
  </Drawer>
{/if}
