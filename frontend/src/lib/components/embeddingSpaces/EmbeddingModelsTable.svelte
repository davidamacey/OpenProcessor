<script lang="ts">
  /** The encoders this deployment can serve, as the catalog serves them; an unavailable one shows why. */
  import type { EmbeddingModelEntry } from '$lib/types_embeddingSpaces';

  interface Props {
    models: EmbeddingModelEntry[];
  }

  let { models }: Props = $props();
</script>

<section class="surface relative min-w-0 overflow-x-auto p-4" aria-label="Encoders">
  <h2 class="mb-2 text-base font-semibold">Encoders this deployment can serve</h2>
  <table class="w-full text-left text-sm" data-testid="model-table">
    <thead class="text-xs text-zinc-500">
      <tr>
        <th class="py-1 pr-3 font-normal">Encoder</th>
        <th class="py-1 pr-3 font-normal">Modalities</th>
        <th class="py-1 pr-3 font-normal">Native size</th>
        <th class="py-1 pr-3 font-normal">Smaller sizes</th>
        <th class="py-1 pr-3 font-normal">Licence</th>
        <th class="py-1 font-normal">Available</th>
      </tr>
    </thead>
    <tbody>
      {#each models as m (`${m.model_key}@${m.version}`)}
        <tr
          class="border-t border-zinc-800 align-top"
          data-testid="model-row"
          data-model={m.model_key}
        >
          <td class="py-1.5 pr-3 font-mono text-xs">{m.model_key} v{m.version}</td>
          <td class="py-1.5 pr-3 text-xs">{m.modalities.join(', ')}</td>
          <td class="py-1.5 pr-3 font-mono text-xs">{m.native_dim}</td>
          <td class="py-1.5 pr-3 font-mono text-xs"
            >{m.matryoshka_dims.join(', ') || '—'}</td
          >
          <td class="py-1.5 pr-3 text-xs">{m.licence}</td>
          <td class="py-1.5 text-xs">
            {#if m.available}yes{:else}<span class="text-amber-300"
                >no{m.unavailable_reason ? `: ${m.unavailable_reason}` : ''}</span
              >{/if}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</section>
