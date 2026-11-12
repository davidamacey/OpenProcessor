<!--
  Chooses which of the served columns the table shows. The choice goes up as column ids in the
  served vocabulary's order; the page writes it to the URL.
-->
<script lang="ts">
  import type { TableColumn } from '$lib/types_table';

  interface Props {
    vocabulary: TableColumn[];
    chosen: string[];
    onchange: (ids: string[]) => void;
  }

  let { vocabulary, chosen, onchange }: Props = $props();

  function toggle(id: string, on: boolean): void {
    const next = vocabulary.filter((c) => (c.id === id ? on : chosen.includes(c.id)));
    if (next.length > 0) onchange(next.map((c) => c.id));
  }
</script>

<details class="relative" data-testid="column-picker">
  <summary
    class="cursor-pointer rounded border border-line-strong px-2 py-1 text-xs text-fg-2 hover:text-fg-1"
  >
    Columns
  </summary>
  <fieldset
    class="absolute right-0 z-20 mt-1 flex min-w-44 flex-col gap-1 rounded border border-line-strong bg-surface-1 p-2 shadow-lg"
  >
    <legend class="sr-only">Columns shown</legend>
    {#each vocabulary as c (c.id)}
      <label class="flex items-center gap-2 text-xs text-fg-1">
        <input
          type="checkbox"
          checked={chosen.includes(c.id)}
          disabled={chosen.length === 1 && chosen[0] === c.id}
          onchange={(e) => toggle(c.id, e.currentTarget.checked)}
        />
        {c.label || c.id}
      </label>
    {/each}
  </fieldset>
</details>
