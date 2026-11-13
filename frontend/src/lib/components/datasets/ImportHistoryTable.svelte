<script lang="ts">
  /**
   * "Past imports" under the wizard in the Import dataset tab: this project's dataset imports
   * (`GET /datasets/imports`), status filter chips from the served `status_labels`, each row
   * opening its job page under Ingest.
   */
  import { resolve } from '$app/paths';
  import DataTable, { type Column } from '$components/ui/DataTable.svelte';
  import Section from '$components/ui/Section.svelte';
  import { listDatasetImports } from '$lib/api_datasetImport';
  import { apiErrorText } from '$lib/api_errors';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { DatasetFormatsResponse, DatasetImportJob } from '$lib/types_import';

  interface Props {
    formats: DatasetFormatsResponse;
  }
  let { formats }: Props = $props();

  const PAGE_SIZE = 50;

  let status = $state('');
  let pageNo = $state(1);
  let items = $state<DatasetImportJob[] | null>(null);
  let total = $state(0);
  let error = $state<string | null>(null);

  async function load(): Promise<void> {
    try {
      const list = await listDatasetImports({
        page: pageNo,
        page_size: PAGE_SIZE,
        status: status || null,
      });
      items = list.items;
      total = list.total;
      error = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      error = apiErrorText(e);
    }
  }

  $effect(() => {
    if (datasetsAvailability.available !== true) return;
    void status;
    void pageNo;
    void load();
  });

  const pages = $derived(Math.max(1, Math.ceil(total / PAGE_SIZE)));

  const columns: Column<DatasetImportJob>[] = [
    { id: 'name', label: 'Name', value: (j) => j.name ?? j.import_id },
    { id: 'format', label: 'Format', value: (j) => j.source.format ?? '' },
    {
      id: 'images',
      label: 'Images',
      value: (j) => j.progress.images_done,
      align: 'right',
    },
    { id: 'status', label: 'Status', value: (j) => j.status },
    { id: 'started', label: 'Started', value: (j) => j.started_at ?? '' },
  ];
  const formatLabel = (id: string | null | undefined): string =>
    formats.formats.find((f) => f.format === id)?.label ?? id ?? '';
</script>

<Section title="Past imports">
  <div
    class="mb-3 flex flex-wrap gap-1 text-meta"
    role="group"
    aria-label="Filter by status"
  >
    <button
      type="button"
      class="chip {status === '' ? 'border-accent text-accent-text' : ''}"
      aria-pressed={status === ''}
      onclick={() => {
        status = '';
        pageNo = 1;
      }}>All</button
    >
    {#each Object.entries(formats.status_labels) as [id, label] (id)}
      <button
        type="button"
        class="chip {status === id ? 'border-accent text-accent-text' : ''}"
        aria-pressed={status === id}
        onclick={() => {
          status = id;
          pageNo = 1;
        }}>{label}</button
      >
    {/each}
  </div>

  {#if error}
    <p class="text-body text-danger-text">{error}</p>
  {:else if !items}
    <p class="text-body text-fg-3">Loading…</p>
  {:else if items.length === 0}
    <p class="text-body text-fg-3" data-testid="imports-empty">No imports.</p>
  {:else}
    <div data-testid="imports-list">
      <DataTable
        {columns}
        rows={items}
        rowKey={(j) => j.import_id}
        label="Past imports"
        maxHeightClass="max-h-96"
      >
        {#snippet cell(j, col)}
          {#if col.id === 'name'}
            <a
              class="text-accent-text hover:underline"
              href={resolve(
                projectHref(`/ingest/imports/${encodeURIComponent(j.import_id)}`),
              )}>{j.name ?? j.import_id}</a
            >
          {:else if col.id === 'format'}
            {formatLabel(j.source.format)}
          {:else if col.id === 'images'}
            {j.progress.images_done.toLocaleString()} / {j.progress.images_total.toLocaleString()}
          {:else if col.id === 'status'}
            {formats.status_labels[j.status] ?? j.status}
          {:else}
            {j.started_at ?? '—'}
          {/if}
        {/snippet}
      </DataTable>
    </div>
    {#if pages > 1}
      <div class="mt-2 flex items-center gap-2 text-meta text-fg-2">
        <button
          type="button"
          class="btn btn-sm"
          disabled={pageNo <= 1}
          onclick={() => pageNo--}>Prev</button
        >
        <span>page {pageNo} of {pages}</span>
        <button
          type="button"
          class="btn btn-sm"
          disabled={pageNo >= pages}
          onclick={() => pageNo++}>Next</button
        >
      </div>
    {/if}
  {/if}
</Section>
