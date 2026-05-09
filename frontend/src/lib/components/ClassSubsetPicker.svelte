<script lang="ts">
  /**
   * Class multi-select with preset chips and live counters.
   *
   * Owns its UI state (search query, expanded/collapsed); the parent
   * binds `selected` (set of class_ids) and `singleCls`.
   */
  import type { OpClass } from '$lib/types';
  import type { ClassSubsetPreset } from '$lib/types_train';

  interface Props {
    classes: OpClass[];
    /** Selected class_ids. `null`/empty = "all classes". */
    selected: number[] | null;
    setSelected: (ids: number[] | null) => void;
    singleCls: boolean;
    setSingleCls: (v: boolean) => void;
    presets?: ClassSubsetPreset[];
  }

  let {
    classes,
    selected,
    setSelected,
    singleCls,
    setSingleCls,
    presets = [],
  }: Props = $props();

  let expanded = $state<boolean>(false);
  let query = $state<string>('');

  // null/empty selection means "every non-deprecated class". Show that
  // as "All N classes" rather than an empty checkbox list.
  const allClasses = $derived(classes.filter((c) => !c.deprecated));
  const allMode = $derived(selected === null);
  const selectedSet = $derived(new Set(selected ?? []));

  const filtered = $derived.by(() => {
    const q = query.trim().toLowerCase();
    const list = q
      ? allClasses.filter(
          (c) =>
            c.name.toLowerCase().includes(q) ||
            (c.group ?? '').toLowerCase().includes(q),
        )
      : allClasses;
    return [...list].sort((a, b) => (b.validated_count ?? 0) - (a.validated_count ?? 0));
  });

  // Counters use the resolved class list — when `allMode`, that's
  // every non-deprecated class.
  const effectiveIds = $derived.by(() => {
    if (allMode) return allClasses.map((c) => c.id);
    return selected ?? [];
  });
  const totalValidated = $derived.by(() => {
    let n = 0;
    const sel = new Set(effectiveIds);
    for (const c of allClasses) {
      if (sel.has(c.id)) n += c.validated_count ?? 0;
    }
    return n;
  });

  function toggle(id: number): void {
    // First click out of "all" mode materialises the explicit list.
    if (allMode) {
      const next = allClasses.map((c) => c.id).filter((x) => x !== id);
      setSelected(next);
      return;
    }
    const cur = new Set(selected ?? []);
    if (cur.has(id)) cur.delete(id);
    else cur.add(id);
    setSelected([...cur].sort((a, b) => a - b));
  }

  function selectAll(): void {
    setSelected(null);
  }

  function clearAll(): void {
    setSelected([]);
  }

  function applyPreset(p: ClassSubsetPreset): void {
    const sel = p.selector;
    let ids: number[] | null = null;
    if (sel.kind === 'all') {
      ids = null;
    } else if (sel.kind === 'all_except') {
      const exclude = new Set((sel.names ?? []).map((s) => s.toLowerCase()));
      ids = allClasses
        .filter((c) => !exclude.has(c.name.toLowerCase()))
        .map((c) => c.id)
        .sort((a, b) => a - b);
    } else if (sel.kind === 'names') {
      const include = new Set((sel.names ?? []).map((s) => s.toLowerCase()));
      ids = allClasses
        .filter((c) => include.has(c.name.toLowerCase()))
        .map((c) => c.id)
        .sort((a, b) => a - b);
    } else if (sel.kind === 'groups') {
      const groups = new Set((sel.groups ?? []).map((s) => s.toLowerCase()));
      ids = allClasses
        .filter((c) => groups.has((c.group ?? '').toLowerCase()))
        .map((c) => c.id)
        .sort((a, b) => a - b);
    }
    setSelected(ids);
    if (p.single_cls_default != null) {
      setSingleCls(!!p.single_cls_default);
    }
  }

  // Compact summary for the header. Shown whether expanded or not so
  // the user knows their selection without unfolding the picker.
  const summary = $derived.by(() => {
    if (allMode) {
      return `All ${allClasses.length} classes · ${totalValidated.toLocaleString()} validated crops`;
    }
    const n = selected?.length ?? 0;
    return `${n} class${n === 1 ? '' : 'es'} selected · ${totalValidated.toLocaleString()} validated crops`;
  });
</script>

<section class="rounded-md border border-zinc-800 bg-zinc-900">
  <button
    type="button"
    class="flex w-full items-center gap-3 px-3 py-2 text-left text-sm hover:bg-zinc-800"
    onclick={() => (expanded = !expanded)}
    aria-expanded={expanded}
  >
    <span class="font-mono text-xs text-zinc-500">{expanded ? '▼' : '▶'}</span>
    <span class="font-semibold text-zinc-100">Classes to train</span>
    <span class="grow text-xs text-zinc-400">{summary}</span>
  </button>

  {#if expanded}
    <div class="border-t border-zinc-800 p-3">
      <!-- Top row: search + select / clear -->
      <div class="mb-2 flex flex-wrap items-center gap-2">
        <input
          type="search"
          bind:value={query}
          placeholder="Search classes…"
          class="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
        <button type="button" class="btn" onclick={selectAll}>Select all</button>
        <button type="button" class="btn" onclick={clearAll}>Clear</button>
      </div>

      <!-- Preset chips -->
      {#if presets.length > 0}
        <div class="mb-3 flex flex-wrap gap-1.5">
          <span class="text-[11px] uppercase tracking-wide text-zinc-500">Presets:</span>
          {#each presets as p (p.name)}
            <button
              type="button"
              title={p.description}
              class="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-0.5 text-xs text-zinc-200 hover:border-blue-500 hover:bg-blue-500/10"
              onclick={() => applyPreset(p)}
            >
              {p.label}
            </button>
          {/each}
        </div>
      {/if}

      <!-- Class list -->
      <div class="max-h-72 overflow-auto rounded border border-zinc-800 bg-zinc-950">
        {#if filtered.length === 0}
          <p class="px-3 py-3 text-xs text-zinc-500">No classes match.</p>
        {:else}
          <ul class="divide-y divide-zinc-900">
            {#each filtered as cls (cls.id)}
              {@const checked = allMode || selectedSet.has(cls.id)}
              <li>
                <label
                  class="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm hover:bg-zinc-900"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onchange={() => toggle(cls.id)}
                    class="h-4 w-4 cursor-pointer accent-blue-500"
                  />
                  <span class="grow truncate text-zinc-200">{cls.name}</span>
                  {#if cls.group}
                    <span class="font-mono text-[10px] text-zinc-500">{cls.group}</span>
                  {/if}
                  <span class="font-mono text-xs text-zinc-400">
                    {(cls.validated_count ?? 0).toLocaleString()}
                  </span>
                </label>
              </li>
            {/each}
          </ul>
        {/if}
      </div>

      <div class="mt-3 flex flex-wrap items-center justify-between gap-3">
        <p class="text-xs text-zinc-400">{summary}</p>
        <label class="flex cursor-pointer items-center gap-2 text-xs text-zinc-300">
          <input
            type="checkbox"
            checked={singleCls}
            onchange={(e) => setSingleCls((e.currentTarget as HTMLInputElement).checked)}
            class="h-4 w-4 cursor-pointer accent-blue-500"
          />
          Single-class detector (collapse all to "object")
        </label>
      </div>
    </div>
  {/if}
</section>
