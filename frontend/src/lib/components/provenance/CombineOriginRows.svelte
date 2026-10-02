<!--
  P4 combine origin rows for `CropMetaPanel`'s main <dl> (a <dt>/<dd> pair
  per fact, or nothing). Rendered only when the server says the item was
  copied by a combine (`origin_project` non-null). Every value is the
  served one, verbatim.
-->
<script lang="ts">
  import type { Crop } from '$lib/types';

  let { crop }: { crop: Crop } = $props();

  const merged = $derived(crop.combine_merged_origins ?? []);
  const conflictOrigins = $derived(crop.combine_conflict_origins ?? []);
</script>

{#if crop.origin_project != null}
  <dt class="text-zinc-500">Combined from</dt>
  <dd class="text-zinc-200" data-testid="combine-origin-project">
    <span class="font-mono">{crop.origin_project}</span>
    {#if crop.combine_conflict}
      <span
        class="ml-1 rounded border border-amber-500/40 bg-amber-500/15 px-1 text-[10px] text-amber-200"
        data-testid="combine-origin-conflict">Conflict between sources</span
      >
    {/if}
  </dd>
  {#if crop.origin_item_id != null}
    <dt class="text-zinc-500">Origin item</dt>
    <dd class="break-all font-mono text-zinc-200" data-testid="combine-origin-item">
      {crop.origin_item_id}
    </dd>
  {/if}
  {#if crop.origin_image_id != null}
    <dt class="text-zinc-500">Origin image</dt>
    <dd class="break-all font-mono text-zinc-200" data-testid="combine-origin-image">
      {crop.origin_image_id}
    </dd>
  {/if}
  {#if crop.origin_split != null}
    <dt class="text-zinc-500">Origin split</dt>
    <dd class="text-zinc-200" data-testid="combine-origin-split">{crop.origin_split}</dd>
  {/if}
  {#if crop.combine_conflict && conflictOrigins.length > 0}
    <dt class="text-zinc-500">Conflicting origins</dt>
    <dd class="font-mono text-zinc-200" data-testid="combine-origin-conflicts">
      {conflictOrigins.join(', ')}
    </dd>
  {/if}
  {#if merged.length > 0}
    <dt class="text-zinc-500">Merged origins</dt>
    <dd class="font-mono text-zinc-200" data-testid="combine-origin-merged">
      {merged.join(', ')}
    </dd>
  {/if}
{/if}
