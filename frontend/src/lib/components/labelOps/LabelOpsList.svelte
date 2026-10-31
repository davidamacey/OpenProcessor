<!--
  The recorded operations, newest first, as served. One "Details" button per
  row (no nested interactive controls); the row is a plain list item.
-->
<script lang="ts">
  import { formatCount } from '$lib/format/count';
  import { formatTimestamp } from '$lib/format/date';
  import { humanizeId } from '$lib/humanizeId';
  import type { LabelOp } from '$lib/types_labelOps';

  interface Props {
    items: LabelOp[];
    onopen: (opId: string) => void;
  }
  let { items, onopen }: Props = $props();

  function title(op: LabelOp): string {
    const verb = op.kind === 'label' ? 'Label' : 'Move';
    return op.class_name ? `${verb} to ${op.class_name}` : verb;
  }
</script>

<ul class="flex flex-col gap-2" data-testid="label-ops-list">
  {#each items as op (op.op_id)}
    <li
      class="surface flex flex-wrap items-center gap-x-4 gap-y-2 p-3 text-sm"
      data-testid="label-op-row"
    >
      <div class="flex min-w-0 grow flex-col gap-0.5">
        <span class="truncate font-medium" data-testid="label-op-title">{title(op)}</span>
        <span class="text-xs text-zinc-400" title={op.at}>
          {formatTimestamp(op.at)}{#if op.label_source}&nbsp; · by {humanizeId(
              op.label_source,
            )}{/if}
        </span>
      </div>
      <span class="text-xs text-zinc-300" data-testid="label-op-count">
        <span class="font-mono">{formatCount(op.count)}</span> items,
        <span class="font-mono">{formatCount(op.current)}</span> still labeled
      </span>
      <span
        class="rounded border border-zinc-700 px-1.5 py-0.5 text-xs text-zinc-300"
        data-testid="label-op-status">{humanizeId(op.status)}</span
      >
      <button
        type="button"
        class="btn"
        aria-label={`Details for ${title(op)} at ${formatTimestamp(op.at)}`}
        onclick={() => onopen(op.op_id)}>Details</button
      >
    </li>
  {/each}
</ul>
