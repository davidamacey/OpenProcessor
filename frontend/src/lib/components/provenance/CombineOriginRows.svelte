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
  <dt class="text-fg-3">Combined from</dt>
  <dd class="text-fg-1" data-testid="combine-origin-project">
    <code>{crop.origin_project}</code>
    {#if crop.combine_conflict}
      <span
        class="ml-1 rounded border border-warn/40 bg-warn/15 px-1 text-caption text-warn-text"
        data-testid="combine-origin-conflict">Conflict between sources</span
      >
    {/if}
  </dd>
  {#if crop.origin_item_id != null}
    <dt class="text-fg-3">Origin item</dt>
    <dd class="break-all text-fg-1" data-testid="combine-origin-item">
      <code>{crop.origin_item_id}</code>
    </dd>
  {/if}
  {#if crop.origin_image_id != null}
    <dt class="text-fg-3">Origin image</dt>
    <dd class="break-all text-fg-1" data-testid="combine-origin-image">
      <code>{crop.origin_image_id}</code>
    </dd>
  {/if}
  {#if crop.origin_split != null}
    <dt class="text-fg-3">Origin split</dt>
    <dd class="text-fg-1" data-testid="combine-origin-split">{crop.origin_split}</dd>
  {/if}
  {#if crop.combine_conflict && conflictOrigins.length > 0}
    <dt class="text-fg-3">Conflicting origins</dt>
    <dd class="text-fg-1" data-testid="combine-origin-conflicts">
      <code>{conflictOrigins.join(', ')}</code>
    </dd>
  {/if}
  {#if merged.length > 0}
    <dt class="text-fg-3">Merged origins</dt>
    <dd class="text-fg-1" data-testid="combine-origin-merged">
      <code>{merged.join(', ')}</code>
    </dd>
  {/if}
{/if}
