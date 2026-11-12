<script lang="ts">
  /** The per-class object counts of one export, a class missing from train or val flagged. */
  import type { ExportClassSplitCounts } from '$lib/types_export';

  interface Props {
    rows: readonly ExportClassSplitCounts[];
    /** Wrapper spacing and height cap. */
    class?: string;
  }

  let { rows, class: className = '' }: Props = $props();
</script>

<div class="overflow-auto rounded border border-line {className}">
  <table class="w-full text-xs">
    <thead
      class="sticky top-0 border-b border-line bg-surface-0 text-left uppercase text-fg-3"
    >
      <tr>
        <th class="px-2 py-1 font-medium">Class</th>
        <th class="px-2 py-1 text-right font-medium">Train (objects)</th>
        <th class="px-2 py-1 text-right font-medium">Val (objects)</th>
        <th class="px-2 py-1 text-right font-medium">Test (objects)</th>
      </tr>
    </thead>
    <tbody>
      {#each rows as c (c.class_id)}
        {@const missing = c.train === 0 || c.val === 0}
        <tr
          class="border-b border-line {missing
            ? 'bg-danger/10 text-danger-text'
            : 'text-fg-2'}"
        >
          <td class="px-2 py-1">{c.class_name}</td>
          <td class="px-2 py-1 text-right font-mono">{c.train.toLocaleString()}</td>
          <td class="px-2 py-1 text-right font-mono">{c.val.toLocaleString()}</td>
          <td class="px-2 py-1 text-right font-mono">{c.test.toLocaleString()}</td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>
