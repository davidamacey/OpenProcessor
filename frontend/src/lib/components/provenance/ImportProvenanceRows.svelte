<!--
  W10 import and lock provenance rows for `CropMetaPanel`'s main <dl>.
  Renders <dt>/<dd> pairs, or nothing. Every value is the served one; a
  row appears only when it carries a value. `proposal_chain` strings are
  shown verbatim (question C-4: their format is not documented).
-->
<script lang="ts">
  import { resolve } from '$app/paths';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import { formatTimestamp } from '$lib/formatDate';
  import { projectHref } from '$lib/projectPaths';
  import type { Crop } from '$lib/types';

  let { crop }: { crop: Crop } = $props();

  const importIds = $derived(crop.import_ids ?? []);
  const chain = $derived(crop.proposal_chain ?? []);

  // Only an item that names an import needs to know whether the import
  // pages exist.
  $effect(() => {
    if (importIds.length > 0) void datasetsAvailability.init();
  });
  const linkable = $derived(datasetsAvailability.available === true);
</script>

{#if crop.label_locked}
  <dt class="text-zinc-500">Label</dt>
  <dd class="text-zinc-200" data-testid="import-label-locked">Locked</dd>
{/if}
{#if crop.dataset_split}
  <dt class="text-zinc-500">Dataset split</dt>
  <dd class="text-zinc-200" data-testid="import-split">{crop.dataset_split}</dd>
{/if}
{#if importIds.length > 0}
  <dt class="text-zinc-500">Imports</dt>
  <dd class="flex flex-wrap gap-x-2 font-mono text-zinc-200" data-testid="import-ids">
    {#each importIds as id (id)}
      {#if linkable}
        <a
          class="text-blue-300 underline hover:text-blue-200"
          href={resolve(projectHref(`/datasets/imports/${encodeURIComponent(id)}`))}
          >{id}</a
        >
      {:else}
        <span>{id}</span>
      {/if}
    {/each}
  </dd>
{/if}
{#if crop.imported_at}
  <dt class="text-zinc-500">Imported</dt>
  <dd class="font-mono text-zinc-300" title={crop.imported_at} data-testid="import-at">
    {formatTimestamp(crop.imported_at)}
  </dd>
{/if}
{#if crop.proposed_by_import}
  <dt class="text-zinc-500">Proposed by import</dt>
  <dd class="font-mono text-zinc-200" data-testid="import-proposed-by">
    {crop.proposed_by_import}
  </dd>
{/if}
{#if crop.on_negative_frame}
  <dt class="text-zinc-500">Negative frame</dt>
  <dd class="text-zinc-200" data-testid="import-negative-frame">on a negative frame</dd>
{/if}
{#if crop.import_standalone_region}
  <dt class="text-zinc-500">Standalone region</dt>
  <dd class="text-zinc-200" data-testid="import-standalone-region">
    imported without a parent item
  </dd>
{/if}
{#if chain.length > 0}
  <dt class="text-zinc-500">Proposal chain</dt>
  <dd class="flex flex-wrap gap-1" data-testid="import-proposal-chain">
    {#each chain as step, i (i)}
      <span
        class="rounded border border-zinc-700 bg-zinc-800/60 px-1 font-mono text-[10px] text-zinc-300"
        >{step}</span
      >
    {/each}
  </dd>
{/if}
