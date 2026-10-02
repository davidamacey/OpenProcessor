<script lang="ts">
  /**
   * `/datasets/imports` — this project's dataset imports (any_domain_plan.md
   * §7.12 item 3): `GET /datasets/imports`, status chips from the served
   * `status_labels`, each row linking to its job view.
   */
  import { resolve } from '$app/paths';
  import DatasetsGate from '$components/datasets/DatasetsGate.svelte';
  import { datasetErrorText, listDatasetImports } from '$lib/api';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { DatasetImportList } from '$lib/types_import';

  const PAGE_SIZE = 50;

  let status = $state('');
  let pageNo = $state(1);
  let list = $state<DatasetImportList | null>(null);
  let error = $state<string | null>(null);

  async function load(): Promise<void> {
    try {
      list = await listDatasetImports({
        page: pageNo,
        page_size: PAGE_SIZE,
        status: status || null,
      });
      error = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      error = datasetErrorText(e);
    }
  }

  $effect(() => {
    if (datasetsAvailability.available !== true) return;
    void status;
    void pageNo;
    void load();
  });

  const pages = $derived(list ? Math.max(1, Math.ceil(list.total / PAGE_SIZE)) : 1);
</script>

<div class="mx-auto max-w-6xl space-y-6 p-6">
  <div class="flex flex-wrap items-baseline justify-between gap-2">
    <h1 class="text-lg font-semibold text-zinc-100">Dataset imports</h1>
    {#if datasetsAvailability.available === true}
      <a class="btn btn-primary btn-sm" href={resolve(projectHref('/datasets/import'))}
        >Import a labeled dataset</a
      >
    {/if}
  </div>

  <DatasetsGate>
    {#snippet children(formats)}
      <div
        class="flex flex-wrap gap-1 text-xs"
        role="group"
        aria-label="Filter by status"
      >
        <button
          type="button"
          class="chip {status === '' ? 'border-blue-500 text-blue-200' : ''}"
          aria-pressed={status === ''}
          onclick={() => {
            status = '';
            pageNo = 1;
          }}>All</button
        >
        {#each Object.entries(formats.status_labels) as [id, label] (id)}
          <button
            type="button"
            class="chip {status === id ? 'border-blue-500 text-blue-200' : ''}"
            aria-pressed={status === id}
            onclick={() => {
              status = id;
              pageNo = 1;
            }}>{label}</button
          >
        {/each}
      </div>

      {#if error}
        <p class="text-sm text-red-300">{error}</p>
      {:else if !list}
        <p class="text-sm text-zinc-500">Loading…</p>
      {:else if list.items.length === 0}
        <p class="text-sm text-zinc-500" data-testid="imports-empty">No imports.</p>
      {:else}
        <table class="w-full text-left text-xs" data-testid="imports-list">
          <thead class="text-zinc-500">
            <tr>
              <th class="py-1 pr-3">Import</th>
              <th class="py-1 pr-3">Status</th>
              <th class="py-1 pr-3 text-right">Images</th>
              <th class="py-1 pr-3">Source</th>
              <th class="py-1">Started</th>
            </tr>
          </thead>
          <tbody>
            {#each list.items as j (j.import_id)}
              <tr class="border-t border-zinc-800">
                <td class="py-1.5 pr-3">
                  <a
                    class="text-blue-300 hover:underline"
                    href={resolve(
                      projectHref(`/datasets/imports/${encodeURIComponent(j.import_id)}`),
                    )}>{j.name ?? j.import_id}</a
                  >
                </td>
                <td class="py-1.5 pr-3">{formats.status_labels[j.status] ?? j.status}</td>
                <td class="py-1.5 pr-3 text-right font-mono">
                  {j.progress.images_done.toLocaleString()} / {j.progress.images_total.toLocaleString()}
                </td>
                <td class="py-1.5 pr-3 font-mono text-zinc-400">{j.source.root}</td>
                <td class="py-1.5 text-zinc-400">{j.started_at ?? '—'}</td>
              </tr>
            {/each}
          </tbody>
        </table>
        {#if pages > 1}
          <div class="flex items-center gap-2 text-xs text-zinc-400">
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
    {/snippet}
  </DatasetsGate>
</div>
