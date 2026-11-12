<script lang="ts">
  /** The encoders this deployment can serve, as the catalog serves them; an unavailable one shows why. */
  import type { EmbeddingModelEntry } from '$lib/types_embeddingSpaces';

  interface Props {
    models: EmbeddingModelEntry[];
  }

  let { models }: Props = $props();
</script>

<section class="surface relative min-w-0 overflow-x-auto p-4" aria-label="Encoders">
  <h2 class="mb-2 text-title font-semibold">Encoders this deployment can serve</h2>
  <table class="w-full text-left text-body" data-testid="model-table">
    <thead class="text-meta text-fg-3">
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
          class="border-t border-line align-top"
          data-testid="model-row"
          data-model={m.model_key}
        >
          <td class="py-1.5 pr-3 font-mono text-meta">{m.model_key} v{m.version}</td>
          <td class="py-1.5 pr-3 text-meta">{m.modalities.join(', ')}</td>
          <td class="py-1.5 pr-3 font-mono text-meta">{m.native_dim}</td>
          <td class="py-1.5 pr-3 font-mono text-meta"
            >{m.matryoshka_dims.join(', ') || '—'}</td
          >
          <td class="py-1.5 pr-3 text-meta">{m.licence}</td>
          <td class="py-1.5 text-meta">
            {#if m.available}yes{:else}<span class="text-warn-text"
                >no{m.unavailable_reason ? `: ${m.unavailable_reason}` : ''}</span
              >{/if}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</section>
