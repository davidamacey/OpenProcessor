<!--
  W9 provenance rows for `CropMetaPanel`'s main <dl>: which VLM endpoint
  (`name@revision`), model and prompt pack produced this item's VLM
  reading. Each row renders only when the item carries the value, verbatim
  (nothing is looked up or relabeled). Renders <dt>/<dd> pairs, or nothing.
-->
<script lang="ts">
  import type { Crop } from '$lib/types';

  let { crop }: { crop: Crop } = $props();

  const rows = $derived(
    [
      { id: 'endpoint', label: 'VLM endpoint', value: crop.vlm_endpoint },
      { id: 'model', label: 'VLM model', value: crop.vlm_model },
      { id: 'prompt-pack', label: 'VLM prompt pack', value: crop.vlm_prompt_pack },
    ].filter((r): r is { id: string; label: string; value: string } => r.value != null),
  );
</script>

{#each rows as r (r.id)}
  <dt class="text-zinc-500">{r.label}</dt>
  <dd
    class="break-all font-mono text-xs text-zinc-200"
    data-testid="vlm-provenance-{r.id}"
  >
    {r.value}
  </dd>
{/each}
