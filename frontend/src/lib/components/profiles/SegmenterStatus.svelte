<script lang="ts">
  /**
   * The deployment's segmenters as served by `GET /config/vocabulary`
   * `segmenters[]`: status, endpoint, masks, the server's
   * candidate cap and the segmenter's own score floor. Shown next to the
   * profile's segmenter fields so the author sees the served limits; no
   * client cap or floor exists. Status ids have no served labels.
   */
  import type { VocabSegmenter } from '$lib/types_profiles';

  interface Props {
    segmenters: VocabSegmenter[];
  }

  let { segmenters }: Props = $props();
</script>

<div
  class="rounded border border-line bg-surface-1/40 p-2 text-meta"
  data-testid="segmenter-status"
>
  {#if segmenters.length === 0}
    <p class="text-fg-2">No segmenter is configured on this deployment.</p>
  {:else}
    <table class="w-full text-left">
      <thead class="text-fg-3">
        <tr>
          <th class="py-0.5 pr-3 font-normal">Segmenter</th>
          <th class="py-0.5 pr-3 font-normal">Status</th>
          <th class="py-0.5 pr-3 font-normal">Masks</th>
          <th class="py-0.5 pr-3 text-right font-normal">Candidate cap</th>
          <th class="py-0.5 text-right font-normal">Default floor</th>
        </tr>
      </thead>
      <tbody>
        {#each segmenters as s (s.name)}
          <tr class="border-t border-line" data-testid="segmenter-row">
            <td class="py-0.5 pr-3">
              <span>{s.choice.label}</span>
              {#if s.endpoint}<code class="block text-caption text-fg-3"
                  >{s.endpoint}</code
                >{/if}
            </td>
            <td class="py-0.5 pr-3">{s.status}</td>
            <td class="py-0.5 pr-3">{s.masks ? 'yes' : 'no'}</td>
            <td class="py-0.5 pr-3 text-right tabular-nums">{s.max_candidates ?? '—'}</td>
            <td class="py-0.5 text-right tabular-nums">{s.default_min_score ?? '—'}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</div>
