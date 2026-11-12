<!--
  The served confusion matrix of the accuracy audit: one row per detector
  class, one column per class a human chose, each cell a served crop count.
  The agreeing cell of a row (same class name) is outlined. Rows and
  columns are the keys of the served object, only sorted for reading.
-->
<script lang="ts">
  interface Props {
    confusion: Record<string, Record<string, number>>;
  }
  let { confusion }: Props = $props();

  const rows = $derived(Object.keys(confusion).sort((a, b) => a.localeCompare(b)));
  const cols = $derived(
    [...new Set(rows.flatMap((r) => Object.keys(confusion[r] ?? {})))].sort((a, b) =>
      a.localeCompare(b),
    ),
  );
</script>

<section class="surface flex flex-col gap-2 p-4" data-testid="audit-confusion">
  <div>
    <h3 class="text-body font-semibold text-fg-1">Detector class vs human class</h3>
    <p class="text-meta text-fg-3">
      Rows are what the detector said, columns are what a human chose. Crops on the
      outlined cells agree.
    </p>
  </div>
  {#if rows.length === 0}
    <p class="text-body text-fg-3">No audited crops yet.</p>
  {:else}
    <div class="overflow-x-auto">
      <table class="text-left text-meta">
        <thead class="text-fg-3">
          <tr>
            <th class="sticky left-0 bg-surface-1 py-1 pr-3 font-medium">Detector</th>
            {#each cols as c (c)}
              <th class="px-2 py-1 text-right font-medium whitespace-nowrap">{c}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each rows as r (r)}
            <tr class="border-t border-line">
              <th
                class="sticky left-0 bg-surface-1 py-1.5 pr-3 font-medium whitespace-nowrap text-fg-1"
                >{r}</th
              >
              {#each cols as c (c)}
                {@const n = confusion[r]?.[c] ?? 0}
                <td
                  class="px-2 py-1.5 text-right font-mono {r === c
                    ? 'outline outline-1 -outline-offset-1 outline-success/50'
                    : ''} {n === 0 ? 'text-fg-3' : 'text-fg-1'}"
                  data-testid="confusion-cell"
                  data-detector={r}
                  data-human={c}>{n}</td
                >
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</section>
