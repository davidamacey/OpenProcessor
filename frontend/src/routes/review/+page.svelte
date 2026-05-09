<script lang="ts">
  import {
    deleteCropLabel,
    getReviewQueue,
    getSourceImageWithBbox,
    getThumbUrl,
    putCropLabel,
  } from '$lib/api';
  import type { PaginatedResponse, ReviewItem, ReviewTab, UndoEntry } from '$lib/types';
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
  let pageNum = $state<number>(1);
  const pageSize = 30;
  let cursor = $state<number>(0); // index within current page
  let data = $state<PaginatedResponse<ReviewItem> | null>(null);
  let loading = $state<boolean>(false);
  let error = $state<string | null>(null);

  // Filter bar
  let hddSource = $state<string>('');
  let classFilter = $state<number | null>(null);
  let confMin = $state<number>(0);
  let confMax = $state<number>(1);

  async function load(): Promise<void> {
    loading = true;
    error = null;
    try {
      const filter: Record<string, unknown> = {};
      if (hddSource) filter.hdd_source = hddSource;
      if (classFilter != null) filter.class_id = classFilter;
      if (confMin > 0) filter.conf_min = confMin;
      if (confMax < 1) filter.conf_max = confMax;
      data = await getReviewQueue(tab, pageNum, pageSize, filter);
      cursor = 0;
    } catch (e) {
      error = (e as Error).message;
      data = null;
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    keyboardStore.setScope('review');
  });

  $effect(() => {
    void tab;
    void pageNum;
    void hddSource;
    void classFilter;
    void confMin;
    void confMax;
    void load();
  });

  const items = $derived(data?.items ?? []);
  const current = $derived<ReviewItem | null>(items[cursor] ?? null);
  const total = $derived(data?.total ?? 0);
  const totalPages = $derived(Math.max(1, Math.ceil(total / pageSize)));

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
    if (data) {
      data = { ...data, items: data.items.filter((x) => x.id !== id) };
    }
    cursor = Math.min(cursor, Math.max(0, (data?.items.length ?? 1) - 1));
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
    if (cursor >= items.length - 1 && pageNum < totalPages) {
      pageNum += 1;
    }
  }

  async function discard(): Promise<void> {
    if (!current) return;
    const ok = window.confirm('Discard this crop?');
    if (!ok) return;
    try {
      await deleteCropLabel(current.id);
      const id = current.id;
      if (data) data = { ...data, items: data.items.filter((x) => x.id !== id) };
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
    reg(
      'arrowleft',
      () => {
        if (pageNum > 1) pageNum -= 1;
      },
      'Previous page',
    );
    reg(
      'arrowright',
      () => {
        if (pageNum < totalPages) pageNum += 1;
      },
      'Next page',
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
          pageNum = 1;
        }}
      >
        {t.label}
      </button>
    {/each}
    <span class="grow"></span>
    <span class="font-mono text-xs text-zinc-500">
      {data ? `${cursor + 1} / ${items.length}` : '—'} on page · {total} total
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

        <!-- Top-class hotkey legend -->
        <div class="mt-3 flex flex-wrap gap-2 text-[11px] text-zinc-500">
          {#each topClasses as cls, i (cls.id)}
            <span>
              <kbd>{i === 9 ? '0' : i + 1}</kbd>
              <span class="ml-0.5 text-zinc-300">{cls.name}</span>
            </span>
          {/each}
        </div>
      </div>
    {/if}
  </div>

  <!-- Pagination -->
  <div
    class="flex items-center justify-end gap-3 border-t border-zinc-800 px-4 py-2 text-sm"
  >
    <button
      class="btn"
      type="button"
      disabled={pageNum <= 1}
      onclick={() => (pageNum = Math.max(1, pageNum - 1))}
    >
      ← Prev
    </button>
    <span class="font-mono text-xs text-zinc-400">page {pageNum} / {totalPages}</span>
    <button
      class="btn"
      type="button"
      disabled={pageNum >= totalPages}
      onclick={() => (pageNum = Math.min(totalPages, pageNum + 1))}
    >
      Next →
    </button>
  </div>
</div>
