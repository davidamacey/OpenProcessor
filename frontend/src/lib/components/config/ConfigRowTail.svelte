<script
  lang="ts"
  generics="Row extends { name: string; read_only?: boolean; updated_at?: string | null }"
>
  /** The last two cells of a config list row: when it was updated, and Clone / Delete. */
  import type { CloneDeleteFlow } from '$lib/config/cloneDeleteFlow.svelte';
  import type { Snippet } from 'svelte';
  import { formatTimestamp } from '$lib/format/date';

  interface Props {
    row: Row;
    flow: CloneDeleteFlow<Row>;
    /** Rendered before the Clone button. */
    lead?: Snippet;
  }

  let { row, flow, lead }: Props = $props();
</script>

<td class="py-1.5 pr-3 text-xs text-zinc-500">
  {row.updated_at ? formatTimestamp(row.updated_at) : '—'}
</td>
<td class="py-1.5 text-right whitespace-nowrap">
  {@render lead?.()}
  <button
    type="button"
    class="btn btn-sm"
    onclick={() => flow.openClone({ name: row.name, source: null })}>Clone</button
  >
  {#if !row.read_only}
    <button type="button" class="btn btn-sm" onclick={() => flow.askDelete(row)}
      >Delete</button
    >
  {/if}
</td>
