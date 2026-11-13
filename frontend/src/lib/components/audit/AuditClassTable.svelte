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

  // Worst precision first; a class with no precision yet goes last.
  const sorted = $derived(
    [...stats].sort((a, b) => (a.precision ?? 2) - (b.precision ?? 2)),
  );
  const pct = (v: number | null | undefined): string =>
    `${Math.max(0, Math.min(1, v ?? 0)) * 100}%`;
</script>

<section class="surface flex flex-col gap-2 p-4" data-testid={testId}>
  <div>
    <h3 class="text-body font-semibold text-fg-1">{title}</h3>
    <p class="text-meta text-fg-3">{blurb}</p>
  </div>
  {#if stats.length === 0}
    <p class="text-body text-fg-3">No audited crops yet.</p>
  {:else}
    <div class="overflow-x-auto">
      <table class="w-full text-left text-meta">
        <thead class="text-fg-3">
          <tr>
            <th class="py-1 pr-3 font-medium">Class</th>
            <th class="py-1 pr-3 text-right font-medium">Audited</th>
            <th class="py-1 pr-3 text-right font-medium">Correct</th>
            <th class="py-1 text-right font-medium">Precision (95% interval)</th>
          </tr>
        </thead>
        <tbody>
          {#each sorted as row (row.name)}
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
                    class="ml-1 whitespace-nowrap rounded border border-warn/40 bg-warn/10 px-1 text-caption text-warn-text"
                    title="Fewer than {minPerClass} audited crops: not enough to trust this precision."
                    data-testid="insufficient-sample">insufficient sample</span
                  >
                {/if}
              </td>
              <td class="py-1.5 pr-3 text-right tabular-nums">{row.n}</td>
              <td class="py-1.5 pr-3 text-right tabular-nums">{row.correct}</td>
              <td class="py-1.5 text-right tabular-nums">
                <span data-testid="audit-precision">{percentText(row.precision, 1)}</span>
                <span class="block text-caption whitespace-nowrap text-fg-3"
                  >{percentText(row.ci_low, 1)} to {percentText(row.ci_high, 1)}</span
                >
                <span
                  class="relative ml-auto mt-1 block h-1.5 w-28 rounded-full bg-surface-3"
                  data-testid="audit-interval"
                  aria-hidden="true"
                >
                  <span
                    class="absolute inset-y-0 rounded-full {row.insufficient_sample
                      ? 'bg-fg-3'
                      : 'bg-accent'}"
                    style:left={pct(row.ci_low)}
                    style:width="{Math.max(
                      0,
                      Math.min(1, row.ci_high ?? 0) - Math.min(1, row.ci_low ?? 0),
                    ) * 100}%"
                  ></span>
                </span>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</section>
