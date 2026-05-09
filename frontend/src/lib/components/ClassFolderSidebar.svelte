<script lang="ts">
  import { dndzone, SOURCES } from 'svelte-dnd-action';
  import { SORTER_GROUP_HOTKEYS, displayKey } from '$lib/data/sorterHotkeys';
  import type { OpClass } from '$lib/types';

  interface Props {
    /** All registry classes (deprecated already filtered upstream). */
    classes: OpClass[];
    /** Number of selected crops shown as a badge. */
    selectedCount: number;
    /** Called when crops are dropped onto a class folder — assigns them. */
    onDrop: (classId: number, className: string) => void | Promise<void>;
    /** Called when user clicks a class folder name (filter / focus). */
    onClick?: (classId: number, className: string) => void;
    /** Optional currently-active class filter for highlight state. */
    activeClassId?: number | null;
  }

  let {
    classes,
    selectedCount,
    onDrop,
    onClick,
    activeClassId = null,
  }: Props = $props();

  // Group classes by their ``group`` field. Order matches sorter conventions
  // (cars, sportycars, exotics, class_a, class_bs, ...). Anything without
  // a recognized group falls into 'other'.
  type GroupBucket = { group: string; key: string | null; classes: OpClass[] };

  const grouped = $derived.by<GroupBucket[]>(() => {
    const byGroup = new Map<string, OpClass[]>();
    for (const c of classes) {
      const g = c.group ?? 'other';
      const arr = byGroup.get(g) ?? [];
      arr.push(c);
      byGroup.set(g, arr);
    }
    // Sort each group's classes alphabetically.
    for (const arr of byGroup.values()) {
      arr.sort((a, b) => a.name.localeCompare(b.name));
    }
    // Order groups: SORTER_GROUP_HOTKEYS first, others appended.
    const keyByGroup = new Map(SORTER_GROUP_HOTKEYS.map((g) => [g.group, g.key]));
    const ordered: GroupBucket[] = [];
    for (const g of SORTER_GROUP_HOTKEYS) {
      const arr = byGroup.get(g.group);
      if (arr && arr.length > 0) {
        ordered.push({ group: g.group, key: g.key, classes: arr });
        byGroup.delete(g.group);
      }
    }
    for (const [group, arr] of [...byGroup.entries()].sort((a, b) =>
      a[0].localeCompare(b[0]),
    )) {
      ordered.push({ group, key: keyByGroup.get(group) ?? null, classes: arr });
    }
    return ordered;
  });

  // Collapsed state per group (default expanded).
  let collapsed = $state<Set<string>>(new Set());
  function toggle(group: string): void {
    const next = new Set(collapsed);
    if (next.has(group)) next.delete(group);
    else next.add(group);
    collapsed = next;
  }

  // svelte-dnd-action requires a per-zone items array we mutate-not. We use
  // empty arrays per drop target and read the dragged ids from the consider/
  // finalize event payloads.
  function onConsider(_classId: number) {
    return (_e: CustomEvent): void => {
      // Visual feedback only — no state change. The drop target's outline
      // styling is controlled by dropTargetStyle.
    };
  }

  function onFinalize(cls: OpClass) {
    return (e: CustomEvent): void => {
      // svelte-dnd-action puts the dropped items in detail.items; we don't
      // actually mount them in this dropzone (it stays empty), but we need
      // the source to know not to add them anywhere. The actual assignment
      // happens via the parent's onDrop callback using the current selection
      // — which is what the user actually meant by "drag the selected".
      const { items, info } = e.detail as {
        items: Array<{ id: string }>;
        info: { source?: string; trigger?: string };
      };
      // Only fire when the drop is from another zone (the grid).
      if (info.source !== SOURCES.KEYBOARD && info.source !== SOURCES.POINTER) return;
      if (items.length === 0) return;
      void onDrop(cls.id, cls.name);
    };
  }
</script>

<aside
  class="flex w-64 shrink-0 flex-col border-r border-zinc-800 bg-zinc-950"
  aria-label="Class folders — drop selected crops here to label"
>
  <div class="border-b border-zinc-800 px-3 py-2">
    <h2 class="text-xs font-semibold tracking-wide text-zinc-400 uppercase">
      Class folders
    </h2>
    <p class="mt-1 text-[11px] leading-tight text-zinc-500">
      {#if selectedCount > 0}
        Drag the {selectedCount} selected crop{selectedCount === 1 ? '' : 's'} onto a class
        to label.
      {:else}
        Select crops in the grid, then drag them onto a class folder.
      {/if}
    </p>
  </div>

  <div class="flex-1 overflow-y-auto py-2">
    {#each grouped as bucket (bucket.group)}
      <div class="mb-2">
        <button
          type="button"
          class="flex w-full items-center gap-1.5 px-3 py-1 text-left text-[11px]
                 font-semibold tracking-wide text-zinc-400 uppercase
                 hover:bg-zinc-900 hover:text-zinc-200"
          onclick={() => toggle(bucket.group)}
        >
          <span class="font-mono text-[9px] text-zinc-600">
            {collapsed.has(bucket.group) ? '▶' : '▼'}
          </span>
          {#if bucket.key}
            <kbd
              class="rounded bg-zinc-800 px-1 py-0.5 font-mono text-[9px] text-zinc-400"
            >
              {displayKey(bucket.key)}
            </kbd>
          {/if}
          <span class="grow truncate" title={bucket.group}>{bucket.group}</span>
          <span class="font-mono text-[10px] text-zinc-600">
            {bucket.classes.length}
          </span>
        </button>

        {#if !collapsed.has(bucket.group)}
          <ul class="mt-0.5 space-y-0.5 px-2">
            {#each bucket.classes as cls (cls.id)}
              <li>
                <div
                  class="group flex items-center gap-1 rounded border px-2 py-1 text-xs
                         transition {activeClassId === cls.id
                    ? 'border-blue-500/60 bg-blue-500/10 text-white'
                    : 'border-transparent bg-zinc-900/40 text-zinc-300 hover:border-blue-500/40 hover:bg-blue-500/5'}"
                  use:dndzone={{
                    items: [],
                    type: 'op-crop',
                    flipDurationMs: 150,
                    dropTargetStyle: { outline: '2px dashed rgb(59 130 246 / 0.8)' },
                    dropFromOthersDisabled: false,
                    dragDisabled: true,
                  }}
                  onconsider={onConsider(cls.id)}
                  onfinalize={onFinalize(cls)}
                >
                  <button
                    type="button"
                    class="grow truncate text-left"
                    title="Filter to {cls.name}"
                    onclick={() => onClick?.(cls.id, cls.name)}
                  >
                    {cls.name}
                  </button>
                  {#if cls.validated_count}
                    <span class="font-mono text-[10px] text-zinc-500">
                      {cls.validated_count}
                    </span>
                  {/if}
                </div>
              </li>
            {/each}
          </ul>
        {/if}
      </div>
    {/each}
  </div>
</aside>
