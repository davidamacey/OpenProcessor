<!--
  Served per-class precision of the detector or the VLM from the accuracy
  audit: n, correct, precision with its Wilson 95% interval. A class the
  server flags `insufficient_sample` is dimmed and says so; its precision
  is not a number to trust yet.
-->
<script lang="ts">
  import { percentText } from '$lib/labelConfirmation';
  import type { AuditClassStat } from '$lib/types_labelConfirmation';

  interface Props {
    title: string;
    blurb: string;
    stats: AuditClassStat[];
    minPerClass: number;
    testId: string;
  }
  let { title, blurb, stats, minPerClass, testId }: Props = $props();
</script>

<section class="surface flex flex-col gap-2 p-4" data-testid={testId}>
  <div>
    <h3 class="text-sm font-semibold text-fg-1">{title}</h3>
    <p class="text-xs text-fg-3">{blurb}</p>
  </div>
  {#if stats.length === 0}
    <p class="text-sm text-fg-3">No audited crops yet.</p>
  {:else}
    <div class="overflow-x-auto">
      <table class="w-full text-left text-xs">
        <thead class="text-fg-3">
          <tr>
            <th class="py-1 pr-3 font-medium">Class</th>
            <th class="py-1 pr-3 text-right font-medium">Audited</th>
            <th class="py-1 pr-3 text-right font-medium">Correct</th>
            <th class="py-1 text-right font-medium">Precision (95% interval)</th>
          </tr>
        </thead>
        <tbody>
          {#each stats as row (row.name)}
            <tr
              class="border-t border-line {row.insufficient_sample
                ? 'text-fg-3'
                : 'text-fg-1'}"
              data-testid="audit-class-row"
              data-insufficient={row.insufficient_sample}
            >
              <td class="py-1.5 pr-3">
                <span class="break-all">{row.name}</span>
                {#if row.insufficient_sample}
                  <span
                    class="ml-1 whitespace-nowrap rounded border border-warn/40 bg-warn/10 px-1 text-caption text-warn"
                    title="Fewer than {minPerClass} audited crops: not enough to trust this precision."
                    data-testid="insufficient-sample">insufficient sample</span
                  >
                {/if}
              </td>
              <td class="py-1.5 pr-3 text-right font-mono">{row.n}</td>
              <td class="py-1.5 pr-3 text-right font-mono">{row.correct}</td>
              <td class="py-1.5 text-right font-mono">
                <span data-testid="audit-precision">{percentText(row.precision, 1)}</span>
                <span class="block text-caption whitespace-nowrap text-fg-3"
                  >{percentText(row.ci_low, 1)} to {percentText(row.ci_high, 1)}</span
                >
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</section>
