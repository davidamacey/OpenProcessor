<script lang="ts">
  import { SORTER_GROUP_HOTKEYS, displayKey } from '$lib/data/sorterHotkeys';
  import type { OpClass } from '$lib/types';

  interface Props {
    /** Top-N classes for number-hotkey assignment (1..9, 0). */
    topClasses: OpClass[];
    /** Optional click handler for class buttons (assigns the class). */
    onclickClass?: (classId: number) => void;
    /** Optional click handler for group buttons (filters / jumps). */
    onclickGroup?: (group: string) => void;
    /** Compact mode: smaller padding, used in tight grids. */
    compact?: boolean;
  }

  let {
    topClasses,
    onclickClass,
    onclickGroup,
    compact = false,
  }: Props = $props();

  const padCls = $derived(compact ? 'px-1.5 py-0.5 text-[11px]' : 'px-2 py-1 text-xs');
  const kbdCls = $derived(
    compact
      ? 'mr-1 rounded bg-zinc-800 px-1 py-0.5 font-mono text-[9px] text-zinc-400'
      : 'mr-1.5 rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] text-zinc-400',
  );
</script>

<div class="flex flex-col gap-1.5">
  <!-- Group hotkeys (legacy_sorter parity) -->
  <div class="flex flex-wrap gap-1">
    <span class="self-center pr-1 text-[10px] font-semibold tracking-wide text-zinc-500
                 uppercase">
      Groups
    </span>
    {#each SORTER_GROUP_HOTKEYS as g (g.key)}
      <button
        type="button"
        class="rounded border border-zinc-700 bg-zinc-900 {padCls} text-zinc-200
               hover:border-blue-500/60 hover:bg-blue-500/10 hover:text-white
               focus:outline-none focus:ring-2 focus:ring-blue-500/40"
        title="Filter to {g.label} (hotkey {displayKey(g.key)})"
        onclick={() => onclickGroup?.(g.group)}
        disabled={!onclickGroup}
      >
        <kbd class={kbdCls}>{displayKey(g.key)}</kbd>
        {g.label}
      </button>
    {/each}
  </div>

  <!-- Class hotkeys (top-N from current scope) -->
  {#if topClasses.length > 0}
    <div class="flex flex-wrap gap-1">
      <span class="self-center pr-1 text-[10px] font-semibold tracking-wide text-zinc-500
                   uppercase">
        Classes
      </span>
      {#each topClasses as cls, i (cls.id)}
        <button
          type="button"
          class="rounded border border-zinc-700 bg-zinc-900 {padCls} text-zinc-200
                 hover:border-blue-500/60 hover:bg-blue-500/10 hover:text-white
                 focus:outline-none focus:ring-2 focus:ring-blue-500/40"
          title="Assign {cls.name} (hotkey {i === 9 ? '0' : i + 1})"
          onclick={() => onclickClass?.(cls.id)}
          disabled={!onclickClass}
        >
          <kbd class={kbdCls}>{i === 9 ? '0' : i + 1}</kbd>
          {cls.name}
        </button>
      {/each}
    </div>
  {/if}
</div>
