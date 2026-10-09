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
    <h3 class="text-sm font-semibold text-zinc-200">Detector class vs human class</h3>
    <p class="text-xs text-zinc-500">
      Rows are what the detector said, columns are what a human chose. Crops on the
      outlined cells agree.
    </p>
  </div>
  {#if rows.length === 0}
    <p class="text-sm text-zinc-500">No audited crops yet.</p>
  {:else}
    <div class="overflow-x-auto">
      <table class="text-left text-xs">
        <thead class="text-zinc-500">
          <tr>
            <th class="sticky left-0 bg-zinc-900 py-1 pr-3 font-medium">Detector</th>
            {#each cols as c (c)}
              <th class="px-2 py-1 text-right font-medium whitespace-nowrap">{c}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each rows as r (r)}
            <tr class="border-t border-zinc-800">
              <th
                class="sticky left-0 bg-zinc-900 py-1.5 pr-3 font-medium whitespace-nowrap text-zinc-200"
                >{r}</th
              >
              {#each cols as c (c)}
                {@const n = confusion[r]?.[c] ?? 0}
                <td
                  class="px-2 py-1.5 text-right font-mono {r === c
                    ? 'outline outline-1 -outline-offset-1 outline-green-500/50'
                    : ''} {n === 0 ? 'text-zinc-600' : 'text-zinc-200'}"
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
