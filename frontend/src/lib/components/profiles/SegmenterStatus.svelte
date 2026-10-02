<script lang="ts">
  /**
   * The deployment's segmenters as served by `GET /config/vocabulary`
   * `segmenters[]` (§7.4, W8.4): status, endpoint, masks, the server's
   * candidate cap and the segmenter's own score floor. Shown next to the
   * profile's segmenter fields so the author sees the served limits; no
   * client cap or floor exists. Status ids have no served labels (W4-Q11).
   */
  import type { VocabSegmenter } from '$lib/types_profiles';

  interface Props {
    segmenters: VocabSegmenter[];
  }

  let { segmenters }: Props = $props();
</script>

<div
  class="rounded border border-zinc-800 bg-zinc-900/40 p-2 text-xs"
  data-testid="segmenter-status"
>
  {#if segmenters.length === 0}
    <p class="text-zinc-400">No segmenter is configured on this deployment.</p>
  {:else}
    <table class="w-full text-left">
      <thead class="text-zinc-500">
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
          <tr class="border-t border-zinc-800" data-testid="segmenter-row">
            <td class="py-0.5 pr-3">
              <span class="font-mono">{s.choice.label}</span>
              {#if s.endpoint}<span class="block font-mono text-[10px] text-zinc-500"
                  >{s.endpoint}</span
                >{/if}
            </td>
            <td class="py-0.5 pr-3 font-mono">{s.status}</td>
            <td class="py-0.5 pr-3">{s.masks ? 'yes' : 'no'}</td>
            <td class="py-0.5 pr-3 text-right font-mono">{s.max_candidates ?? '—'}</td>
            <td class="py-0.5 text-right font-mono">{s.default_min_score ?? '—'}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</div>
