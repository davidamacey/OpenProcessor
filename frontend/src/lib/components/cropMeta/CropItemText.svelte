<script lang="ts">
  import type { Crop } from '$lib/types';
  import { getThumbUrl } from '$lib/api_cropRead';
  import { pct } from '$lib/cropMeta/pct';

  let { crop }: { crop: Crop } = $props();

  let showTextBoxes = $state(false);
</script>

{#if crop.item_text_lines && crop.item_text_lines.length > 0}
  <div class="mt-4 border-t border-line pt-3">
    <div
      class="mb-1.5 flex items-center justify-between text-caption uppercase tracking-wider text-fg-3"
    >
      <span>Item text</span>
      <button
        type="button"
        class="lowercase tracking-normal text-fg-2 hover:text-fg-1"
        onclick={() => (showTextBoxes = !showTextBoxes)}
      >
        {showTextBoxes ? 'hide boxes' : 'show boxes'}
      </button>
    </div>
    {#if showTextBoxes}
      <div
        class="relative mb-2 aspect-square w-full overflow-hidden rounded bg-surface-1"
      >
        <img
          src={getThumbUrl(crop.id, 320)}
          alt="item text overlay"
          class="h-full w-full object-contain"
        />
        {#each crop.item_text_lines as line, i (i)}
          {#if line.box_norm && line.box_norm.length === 4}
            {@const [x1, y1, x2, y2] = line.box_norm}
            <div
              class="pointer-events-none absolute border border-info/80"
              style="left:{x1 * 100}%; top:{y1 * 100}%; width:{(x2 - x1) *
                100}%; height:{(y2 - y1) * 100}%;"
            ></div>
          {/if}
        {/each}
      </div>
    {/if}
    <ul class="space-y-0.5 text-meta">
      {#each crop.item_text_lines as line, i (i)}
        <li class="flex items-center gap-1.5">
          <span
            class="rounded border border-line-strong bg-surface-1 px-1.5 py-0.5 font-mono text-fg-1"
          >
            {line.text || '∅'}
          </span>
          <span class="font-mono text-caption text-fg-3">{pct(line.confidence)}</span>
        </li>
      {/each}
    </ul>
  </div>
{/if}
