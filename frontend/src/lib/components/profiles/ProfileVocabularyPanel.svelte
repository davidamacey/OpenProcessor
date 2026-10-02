<script lang="ts">
  /**
   * The project's config vocabulary (`GET /config/vocabulary`, §7.4), read
   * only: the models, segmenters, OCR models, text-reading modes and VLM
   * endpoints a region profile can name. The vocabulary has no write
   * route; a profile picks from it. Ids without a served label print raw
   * (W4-Q11).
   */
  import type { ConfigVocabulary, VocabModel } from '$lib/types_profiles';
  import SegmenterStatus from './SegmenterStatus.svelte';

  interface Props {
    vocab: ConfigVocabulary;
  }

  let { vocab }: Props = $props();

  const ocrLists = $derived([
    { id: 'pipeline', label: 'OCR pipeline', models: vocab.ocr.pipeline_models },
    { id: 'det', label: 'OCR text detector', models: vocab.ocr.det_models },
    { id: 'rec', label: 'OCR text recognizer', models: vocab.ocr.rec_models },
  ]);
</script>

{#snippet models(list: VocabModel[], testid: string)}
  {#if list.length === 0}
    <p class="text-xs text-zinc-500">None served.</p>
  {:else}
    <div class="overflow-x-auto">
      <table class="w-full text-left text-xs" data-testid={testid}>
        <thead class="text-zinc-500">
          <tr>
            <th class="py-0.5 pr-3 font-normal">Model</th>
            <th class="py-0.5 pr-3 font-normal">Source</th>
            <th class="py-0.5 pr-3 font-normal">State</th>
            <th class="py-0.5 font-normal">Notes</th>
          </tr>
        </thead>
        <tbody>
          {#each list as m (m.name + (m.project ?? ''))}
            <tr class="border-t border-zinc-800">
              <td class="py-0.5 pr-3 font-mono">{m.choice.label}</td>
              <td class="py-0.5 pr-3 font-mono">{m.source}</td>
              <td class="py-0.5 pr-3 font-mono">
                {m.state ?? '—'}{m.ready === false ? ' (not ready)' : ''}
              </td>
              <td class="py-0.5 text-zinc-400">
                {#if m.configured}<span class="mr-2 text-emerald-300"
                    >configured for this role</span
                  >{/if}
                {#if m.project}<span class="mr-2"
                    >project <span class="font-mono">{m.project}</span>{m.shared
                      ? ' · shared'
                      : ''}</span
                  >{/if}
                {#if m.class_mapping}
                  <span
                    >{m.class_mapping.mapped} classes map{m.class_mapping.unmapped
                      .length > 0
                      ? `; unmapped: ${m.class_mapping.unmapped.join(', ')}`
                      : ''}</span
                  >
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
{/snippet}

<div class="flex flex-col gap-4 text-sm" data-testid="vocabulary-panel">
  <section class="flex flex-col gap-1">
    <h3 class="text-sm font-semibold text-zinc-200">Detectors</h3>
    {@render models(vocab.detectors, 'vocab-detectors')}
  </section>

  <section class="flex flex-col gap-1">
    <h3 class="text-sm font-semibold text-zinc-200">Segmenters</h3>
    <SegmenterStatus segmenters={vocab.segmenters} />
  </section>

  <section class="flex flex-col gap-1">
    <h3 class="text-sm font-semibold text-zinc-200">
      OCR <span class="text-xs font-normal text-zinc-400"
        >({vocab.ocr.available ? 'available' : 'not available'})</span
      >
    </h3>
    {#each ocrLists as l (l.id)}
      <p class="text-xs text-zinc-400">{l.label}</p>
      {@render models(l.models, `vocab-ocr-${l.id}`)}
    {/each}
  </section>

  <section class="flex flex-col gap-1">
    <h3 class="text-sm font-semibold text-zinc-200">Text-reading modes</h3>
    <ul class="space-y-0.5 text-xs" data-testid="vocab-text-modes">
      {#each vocab.text_reader_modes as m (m.id)}
        <li>
          <span class="text-zinc-200">{m.label}</span>
          <code class="ml-1 font-mono text-zinc-500">{m.id}</code>
          <span class="ml-1 text-zinc-400"
            >{m.reads_text ? 'reads text' : 'no text'}{m.needs_vlm
              ? ' · needs the VLM'
              : ''}{m.needs_ocr ? ' · needs OCR' : ''}</span
          >
        </li>
      {/each}
    </ul>
  </section>

  <section class="flex flex-col gap-1">
    <h3 class="text-sm font-semibold text-zinc-200">VLM endpoints</h3>
    {#if vocab.vlm.endpoints.length === 0}
      <p class="text-xs text-zinc-500">No VLM is configured.</p>
    {:else}
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs" data-testid="vocab-vlm">
          <thead class="text-zinc-500">
            <tr>
              <th class="py-0.5 pr-3 font-normal">Endpoint</th>
              <th class="py-0.5 pr-3 font-normal">Model</th>
              <th class="py-0.5 pr-3 font-normal">Where</th>
              <th class="py-0.5 font-normal">Status</th>
            </tr>
          </thead>
          <tbody>
            {#each vocab.vlm.endpoints as v (v.name)}
              <tr class="border-t border-zinc-800">
                <td class="py-0.5 pr-3 font-mono"
                  >{v.name}{#if v.active}<span class="ml-1 font-sans text-emerald-300"
                      >active</span
                    >{/if}</td
                >
                <td class="py-0.5 pr-3 font-mono"
                  >{v.model ?? '—'}{v.resolved_model && v.resolved_model !== v.model
                    ? ` (${v.resolved_model})`
                    : ''}</td
                >
                <td class="py-0.5 pr-3"
                  >{v.locality ?? '—'}{v.sends_images_externally
                    ? ' · sends images externally'
                    : ''}</td
                >
                <td class="py-0.5 font-mono">{v.status ?? '—'}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>
</div>
