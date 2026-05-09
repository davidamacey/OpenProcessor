<script lang="ts">
  import {
    deleteCropLabel,
    getReviewQueue,
    getSourceImageWithBbox,
    getThumbUrl,
    putCropLabel,
  } from '$lib/api';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import type { ReviewItem, ReviewTab, UndoEntry } from '$lib/types';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { undoStore } from '$stores/undo.svelte';

  const TABS: Array<{ id: ReviewTab; label: string }> = [
    { id: 'mismatches', label: 'Mismatches' },
    { id: 'gemma_low_conf', label: 'Gemma Low-Conf' },
    { id: 'outliers', label: 'Outliers' },
    { id: 'uncertainty', label: 'Uncertainty' },
  ];

  let tab = $state<ReviewTab>('mismatches');
  const pageSize = 30;
  let cursor = $state<number>(0); // index within accumulated items
  let items = $state<ReviewItem[]>([]);
  let total = $state<number>(0);
  let loadedPages = $state<number>(0);
  let loading = $state<boolean>(false);
  let loadingMore = $state<boolean>(false);
  let error = $state<string | null>(null);
  const hasMore = $derived(items.length < total);

  // Filter bar
  let hddSource = $state<string>('');
  let classFilter = $state<number | null>(null);
  let confMin = $state<number>(0);
  let confMax = $state<number>(1);

  function _filter(): Record<string, unknown> {
    const f: Record<string, unknown> = {};
    if (hddSource) f.hdd_source = hddSource;
    if (classFilter != null) f.class_id = classFilter;
    if (confMin > 0) f.conf_min = confMin;
    if (confMax < 1) f.conf_max = confMax;
    return f;
  }

  async function loadFirst(): Promise<void> {
    loading = true;
    error = null;
    items = [];
    total = 0;
    loadedPages = 0;
    cursor = 0;
    try {
      const data = await getReviewQueue(tab, 1, pageSize, _filter());
      items = data?.items ?? [];
      total = data?.total ?? items.length;
      loadedPages = 1;
    } catch (e) {
      error = (e as Error).message;
    } finally {
      loading = false;
    }
  }

  async function loadMore(): Promise<void> {
    if (loadingMore || !hasMore) return;
    loadingMore = true;
    try {
      const next = loadedPages + 1;
      const data = await getReviewQueue(tab, next, pageSize, _filter());
      const seen = new Set(items.map((i) => i.id));
      const fresh = (data?.items ?? []).filter((i) => !seen.has(i.id));
      items = [...items, ...fresh];
      total = data?.total ?? total;
      loadedPages = next;
    } catch (e) {
      error = (e as Error).message;
    } finally {
      loadingMore = false;
    }
  }

  $effect(() => {
    keyboardStore.setScope('review');
  });

  $effect(() => {
    void tab;
    void hddSource;
    void classFilter;
    void confMin;
    void confMax;
    void loadFirst();
  });

  const current = $derived<ReviewItem | null>(items[cursor] ?? null);

  const topClasses = $derived(classesStore.topNForCluster(0, 10));

  function snap(it: ReviewItem): UndoEntry {
    return {
      crop_id: it.id,
      prior_class_id: it.class_id,
      prior_label_source: it.label_source,
      prior_validated: it.label_validated,
      at: Date.now(),
    };
  }

  async function assign(classId: number): Promise<void> {
    if (!current) return;
    undoStore.push(snap(current));
    const cls = classesStore.byId(classId);
    // Optimistic: drop from list and advance.
    const id = current.id;
    items = items.filter((x) => x.id !== id);
    total = Math.max(0, total - 1);
    cursor = Math.min(cursor, Math.max(0, items.length - 1));
    if (cursor >= items.length - 1 && hasMore) void loadMore();
    try {
      await putCropLabel(id, classId);
      toastStore.success(`Labeled "${cls?.name ?? classId}".`);
    } catch (e) {
      toastStore.error(`Label failed: ${(e as Error).message}`);
    }
  }

  async function confirmAndAdvance(): Promise<void> {
    if (!current) return;
    const proposed = current.proposed_class_id ?? current.class_id;
    if (proposed == null) {
      toastStore.warn('No proposed class on this item.');
      return;
    }
    await assign(proposed);
  }

  function skip(): void {
    cursor = Math.min(items.length - 1, cursor + 1);
    if (cursor >= items.length - 1 && hasMore) void loadMore();
  }

  async function discard(): Promise<void> {
    if (!current) return;
    const ok = window.confirm('Discard this crop?');
    if (!ok) return;
    try {
      await deleteCropLabel(current.id);
      const id = current.id;
      items = items.filter((x) => x.id !== id);
      total = Math.max(0, total - 1);
    } catch (e) {
      toastStore.error(`Discard failed: ${(e as Error).message}`);
    }
  }

  async function undoLast(): Promise<void> {
    const entry = undoStore.pop();
    if (!entry) {
      toastStore.info('Nothing to undo.');
      return;
    }
    try {
      await deleteCropLabel(entry.crop_id);
      toastStore.success('Reverted.');
    } catch (e) {
      toastStore.error(`Undo failed: ${(e as Error).message}`);
    }
  }

  // Keyboard shortcuts
  $effect(() => {
    const offs: Array<() => void> = [];
    const reg = (combo: string, fn: () => void | Promise<void>, desc: string) =>
      offs.push(keyboardStore.register(combo, () => void fn(), 'review', desc));

    for (let i = 0; i < 10; i++) {
      const key = i === 9 ? '0' : String(i + 1);
      reg(
        key,
        async () => {
          const cls = topClasses[i];
          if (!cls) return;
          await assign(cls.id);
        },
        `Assign top-class #${i + 1}`,
      );
    }
    reg('enter', confirmAndAdvance, 'Confirm proposed & advance');
    reg('n', skip, 'Skip');
    reg('d', discard, 'Discard');
    reg('z', undoLast, 'Undo last');
    // Arrow keys navigate within the loaded queue; auto-load next page when
    // approaching the end so the cursor never starves.
    reg(
      'arrowleft',
      () => {
        cursor = Math.max(0, cursor - 1);
      },
      'Previous item',
    );
    reg(
      'arrowright',
      () => {
        cursor = Math.min(items.length - 1, cursor + 1);
        if (cursor >= items.length - 1 && hasMore) void loadMore();
      },
      'Next item',
    );

    return () => offs.forEach((off) => off());
  });
</script>

<div class="flex h-full flex-col">
  <!-- Tabs -->
  <div class="flex items-center gap-1 border-b border-zinc-800 px-4">
    {#each TABS as t (t.id)}
      <button
        type="button"
        class="px-3 py-2.5 text-sm border-b-2 {tab === t.id
          ? 'border-blue-500 text-white'
          : 'border-transparent text-zinc-400 hover:text-zinc-200'}"
        onclick={() => {
          tab = t.id;
        }}
      >
        {t.label}
      </button>
    {/each}
    <span class="grow"></span>
    <span class="font-mono text-xs text-zinc-500">
      {items.length > 0 ? `${cursor + 1} / ${items.length}` : '—'} loaded · {total} total
    </span>
  </div>

  <!-- Filter bar -->
  <div
    class="flex flex-wrap items-center gap-3 border-b border-zinc-800 bg-zinc-900/40 px-4 py-2 text-xs"
  >
    <label class="flex items-center gap-1.5">
      <span class="text-zinc-400">HDD source</span>
      <input
        type="text"
        bind:value={hddSource}
        placeholder="any"
        class="w-32 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100 focus:border-blue-500 focus:outline-none"
      />
    </label>

    <label class="flex items-center gap-1.5">
      <span class="text-zinc-400">Class</span>
      <select
        bind:value={classFilter}
        class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
      >
        <option value={null}>any</option>
        {#each classesStore.classes as cls (cls.id)}
          <option value={cls.id}>{cls.name}</option>
        {/each}
      </select>
    </label>

    <label class="flex items-center gap-1.5">
      <span class="text-zinc-400">Conf</span>
      <input
        type="number"
        min="0"
        max="1"
        step="0.05"
        bind:value={confMin}
        class="w-16 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
      />
      <span class="text-zinc-500">..</span>
      <input
        type="number"
        min="0"
        max="1"
        step="0.05"
        bind:value={confMax}
        class="w-16 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
      />
    </label>

    <span class="grow"></span>

    <span class="text-[11px] text-zinc-500">
      <kbd>1–9</kbd> assign · <kbd>Enter</kbd> confirm · <kbd>N</kbd> skip · <kbd>D</kbd> discard ·
      <kbd>Z</kbd> undo
    </span>
  </div>

  <!-- Body -->
  <div class="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-hidden p-4 lg:grid-cols-2">
    {#if loading && items.length === 0}
      <p class="col-span-full text-sm text-zinc-500">Loading...</p>
    {:else if error}
      <p class="col-span-full text-sm text-red-300">API unavailable: {error}</p>
    {:else if !current}
      <p class="col-span-full text-sm text-zinc-500">Queue empty.</p>
    {:else}
      <!-- Source image with bbox -->
      <div class="flex min-h-0 flex-col surface p-2">
        <div class="mb-2 flex items-center gap-2 px-1 text-xs text-zinc-400">
          <span>source</span>
          <span class="grow"></span>
          <span class="font-mono">{current.hdd_source ?? ''}</span>
        </div>
        <div class="flex min-h-0 flex-1 items-center justify-center bg-zinc-950">
          <img
            src={getSourceImageWithBbox(current.id)}
            alt="source"
            class="max-h-full max-w-full object-contain"
          />
        </div>
      </div>

      <!-- Crop + meta -->
      <div class="flex min-h-0 flex-col surface p-2">
        <div class="mb-2 flex items-center gap-2 px-1 text-xs text-zinc-400">
          <span>crop</span>
          <span class="grow"></span>
          <span class="font-mono">{current.id.slice(0, 12)}…</span>
        </div>
        <div class="flex min-h-0 flex-1 items-center justify-center bg-zinc-950">
          <img
            src={getThumbUrl(current.id, 384)}
            alt="crop"
            class="max-h-full max-w-full object-contain"
          />
        </div>

        <dl class="mt-3 grid grid-cols-2 gap-y-1 text-xs">
          <dt class="text-zinc-500">Reason</dt>
          <dd class="text-zinc-200">{current.reason}</dd>

          <dt class="text-zinc-500">Current label</dt>
          <dd class="text-zinc-200">
            {current.class_name ?? '—'}
            <span class="ml-1 text-zinc-500">({current.label_source})</span>
          </dd>

          <dt class="text-zinc-500">Proposed</dt>
          <dd class="text-yellow-200">{current.proposed_class_name ?? '—'}</dd>

          <dt class="text-zinc-500">Confidence</dt>
          <dd class="font-mono">
            {current.label_confidence != null
              ? `${(current.label_confidence * 100).toFixed(1)}%`
              : '—'}
          </dd>
        </dl>

        <div class="mt-3 flex flex-wrap gap-2">
          <button class="btn btn-primary" type="button" onclick={confirmAndAdvance}>
            Confirm
          </button>
          <button class="btn" type="button" onclick={skip}>Skip</button>
          <button class="btn btn-danger" type="button" onclick={discard}>Discard</button>
          <button class="btn" type="button" onclick={undoLast}>Undo</button>
        </div>

        <!-- Top-class label buttons (click OR press the hotkey) -->
        <div class="mt-3 flex flex-wrap gap-1.5">
          {#each topClasses as cls, i (cls.id)}
            <button
              type="button"
              class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200
                     hover:border-blue-500/60 hover:bg-blue-500/10 hover:text-white
                     focus:outline-none focus:ring-2 focus:ring-blue-500/40"
              title="Assign {cls.name} (hotkey {i === 9 ? '0' : i + 1})"
              onclick={() => assign(cls.id)}
            >
              <kbd
                class="mr-1.5 rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] text-zinc-400"
              >
                {i === 9 ? '0' : i + 1}
              </kbd>
              {cls.name}
            </button>
          {/each}
        </div>
        <p class="mt-1.5 text-[10px] text-zinc-500">
          Click a class or press its hotkey to assign + advance.
        </p>
      </div>
    {/if}
  </div>

  <!-- Infinite-scroll status bar (no Next/Prev buttons) -->
  <div
    class="flex items-center justify-between gap-3 border-t border-zinc-800 px-4 py-2 text-sm"
  >
    <span class="text-xs text-zinc-500">
      item {Math.min(cursor + 1, items.length)} of {items.length} loaded · {total} total
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if loadingMore}loading more…{:else if hasMore}{total - items.length} more available{:else}all loaded{/if}
    </span>
  </div>

  <!-- Sentinel: when this scrolls into view, load the next page -->
  <div
    use:infiniteScroll={{ onload: loadMore, disabled: loadingMore || !hasMore || loading }}
    class="h-1"
    aria-hidden="true"
  ></div>
</div>
