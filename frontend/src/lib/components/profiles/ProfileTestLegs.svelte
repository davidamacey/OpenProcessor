<script lang="ts">
  /**
   * The legs of a region-profile test (W5): per leg its served status,
   * reason and elapsed time, and a candidate table. Dropped candidates
   * are greyed with the served `drop_reason` in the tooltip. Nothing is
   * judged here; the server's `selected` / `drop_reason` decide.
   */
  import { candidateLabel } from '$lib/configTest/overlayShapes';
  import { humanizeId } from '$lib/humanizeId';
  import type { RegionTestLeg } from '$lib/types_configTest';

  interface Props {
    legs: RegionTestLeg[];
  }

  let { legs }: Props = $props();

  const num = (n: number | null, digits = 2): string =>
    n == null ? '—' : n.toFixed(digits);
</script>

<div class="flex flex-col gap-3" data-testid="profile-test-legs">
  {#each legs as leg (leg.leg)}
    <section
      class="rounded border border-zinc-800 p-2"
      data-testid="test-leg"
      data-leg={leg.leg}
    >
      <p class="flex flex-wrap items-baseline gap-2 text-xs">
        <span class="font-semibold text-zinc-200">{humanizeId(leg.leg)}</span>
        <span
          class="rounded border px-1.5 {leg.status === 'error'
            ? 'border-red-500/40 text-red-300'
            : leg.status === 'ok'
              ? 'border-emerald-500/40 text-emerald-300'
              : 'border-zinc-600 text-zinc-400'}"
          data-testid="test-leg-status">{humanizeId(leg.status)}</span
        >
        {#if leg.elapsed_ms != null}
          <span class="font-mono text-zinc-500">{leg.elapsed_ms} ms</span>
        {/if}
        {#if leg.reason}
          <span class="text-zinc-300" data-testid="test-leg-reason">{leg.reason}</span>
        {/if}
      </p>
      {#if (leg.candidates ?? []).length > 0}
        <table class="mt-1 w-full text-left text-xs">
          <thead class="text-zinc-500">
            <tr>
              <th class="py-0.5 pr-3">Candidate</th>
              <th class="py-0.5 pr-3 text-right">Score</th>
              <th class="py-0.5 pr-3">Selected</th>
              <th class="py-0.5 pr-3">Dropped because</th>
              <th class="py-0.5 pr-3 text-right">Mask IoU</th>
              <th class="py-0.5">Detector</th>
            </tr>
          </thead>
          <tbody>
            {#each leg.candidates ?? [] as c (c.candidate_index)}
              <tr
                class="border-t border-zinc-800 {c.selected ? '' : 'opacity-50'}"
                data-testid="test-candidate"
                data-leg={leg.leg}
                data-index={c.candidate_index}
                data-selected={c.selected}
                title={c.drop_reason
                  ? `Dropped: ${humanizeId(c.drop_reason)}`
                  : undefined}
              >
                <td class="py-0.5 pr-3 font-mono"
                  >{candidateLabel(leg.leg, c.candidate_index)}</td
                >
                <td class="py-0.5 pr-3 text-right font-mono">{num(c.score)}</td>
                <td class="py-0.5 pr-3">{c.selected ? 'yes' : 'no'}</td>
                <td class="py-0.5 pr-3"
                  >{c.drop_reason ? humanizeId(c.drop_reason) : ''}</td
                >
                <td class="py-0.5 pr-3 text-right font-mono">{num(c.mask_iou)}</td>
                <td class="py-0.5 font-mono text-zinc-400">{c.detector ?? '—'}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </section>
  {/each}
</div>
