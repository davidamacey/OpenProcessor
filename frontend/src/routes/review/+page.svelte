<script lang="ts">
  import {
    deleteCropLabel,
    getReviewQueue,
    getSourceImageWithBbox,
    getThumbUrl,
    putCropLabel,
    setCropPlate,
  } from '$lib/api';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import PlateEditor from '$lib/components/PlateEditor.svelte';
  import { bboxNormToXYXY } from '$lib/plate_geometry';
  import type { BBoxNorm, OpClass, ReviewItem, ReviewTab, UndoEntry } from '$lib/types';
  import { classesStore } from '$stores/classes.svelte';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { undoStore } from '$stores/undo.svelte';

  // Unified review by default — one continuous queue of every crop that
  // needs a human, sorted most-uncertain first. The narrower tabs stay
  // available for diagnosing where uncertainty came from.
  const TABS: Array<{ id: ReviewTab; label: string }> = [
    { id: 'all', label: 'All' },
    { id: 'mismatches', label: 'Mismatches' },
    { id: 'gemma_low_conf', label: 'Gemma Low-Conf' },
    { id: 'outliers', label: 'Outliers' },
    { id: 'uncertainty', label: 'Uncertainty' },
    // Phase 5 active-learning loop: validated crops where the newly
    // promoted model disagrees with the human label.
    { id: 'model_disagreements', label: 'Model Disagreements' },
    // Plate-detection review: crops with an LPR/SAM3+Gemma-verified
    // plate bbox waiting for human confirmation in PlateEditor.
    { id: 'plates', label: 'Plates' },
  ];

  let tab = $state<ReviewTab>('all');
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

  // Review is one-at-a-time — only `current` is rendered. We fetch one
  // page on tab/filter change and the cursor-arrow handler pulls the
  // next page as the user nears the end. The earlier eager prefetch
  // drained every page upfront, which on the busy 'all' tab fired ~4
  // chained network calls before first paint and made the page feel
  // frozen on slow connections. Lazy paging keeps first-paint snappy.
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

  // Per-class hotkeys defined on /classes are routed here through the
  // layout-level global keydown listener via dropOnClassStore. Pressing
  // a class's bound letter assigns the current crop and advances —
  // matching the cluster page's bulk-label dispatch shape so the same
  // hotkey works everywhere it makes sense. The plates tab is a
  // different flow (confirming a bbox, not a class) so we no-op there
  // and leave the letters free for plate actions.
  $effect(() => {
    if (tab === 'plates') return;
    const off = dropOnClassStore.register(async (cls: OpClass) => {
      if (!current) {
        toastStore.info('No item to label.');
        return;
      }
      await assign(cls.id);
    });
    return () => off();
  });

  // Tab + class filter fire loadFirst() immediately (single-click changes
  // are intentional). Text + slider filters debounce by 250ms so typing
  // hddSource or dragging the confidence sliders doesn't cause a refetch
  // per keystroke.
  $effect(() => {
    void tab;
    void classFilter;
    void loadFirst();
  });
  let filterDebounce: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    void hddSource;
    void confMin;
    void confMax;
    if (filterDebounce) clearTimeout(filterDebounce);
    filterDebounce = setTimeout(() => {
      filterDebounce = null;
      void loadFirst();
    }, 250);
    return () => {
      if (filterDebounce) {
        clearTimeout(filterDebounce);
        filterDebounce = null;
      }
    };
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
    // Snapshot before discard so undo (Z) brings the crop back into
    // the queue. No nag-confirm — the user presses D dozens of times
    // per session.
    undoStore.push(snap(current));
    const id = current.id;
    items = items.filter((x) => x.id !== id);
    total = Math.max(0, total - 1);
    cursor = Math.min(cursor, Math.max(0, items.length - 1));
    if (cursor >= items.length - 1 && hasMore) void loadMore();
    try {
      await deleteCropLabel(id);
      toastStore.success('Discarded. Press Z to undo.');
    } catch (e) {
      toastStore.error(`Discard failed: ${(e as Error).message}`);
    }
  }

  // -- plate-tab actions ----------------------------------------------
  // The plates tab shows crops where the LPR / SAM3 / Gemma chain
  // proposed a plate bbox but it didn't clear the auto-confirm bar.
  // The operator's job is to either accept the proposal as-is, redraw
  // the bbox in the editor, or mark "no plate visible".
  let plateEditorOpen = $state<boolean>(false);

  function _advancePastPlate(id: string): void {
    items = items.filter((x) => x.id !== id);
    total = Math.max(0, total - 1);
    cursor = Math.min(cursor, Math.max(0, items.length - 1));
    if (cursor >= items.length - 1 && hasMore) void loadMore();
  }

  async function confirmPlate(): Promise<void> {
    if (!current) return;
    if (!current.plate_bbox_norm) {
      toastStore.warn('No plate proposal to confirm — open the editor.');
      return;
    }
    const id = current.id;
    const bbox = bboxNormToXYXY(current.plate_bbox_norm) as [
      number,
      number,
      number,
      number,
    ];
    _advancePastPlate(id);
    try {
      await setCropPlate(id, bbox);
      toastStore.success('Plate confirmed.');
    } catch (e) {
      toastStore.error(`Confirm failed: ${(e as Error).message}`);
    }
  }

  async function rejectPlate(): Promise<void> {
    if (!current) return;
    const id = current.id;
    _advancePastPlate(id);
    try {
      // null bbox = "no plate visible" per setCropPlate contract.
      await setCropPlate(id, null);
      toastStore.success('Plate rejected (no plate visible).');
    } catch (e) {
      toastStore.error(`Reject failed: ${(e as Error).message}`);
    }
  }

  function openPlateEditor(): void {
    if (!current) return;
    plateEditorOpen = true;
  }

  function onPlateEditorSave(_newBbox: BBoxNorm | null): void {
    plateEditorOpen = false;
    if (!current) return;
    // PlateEditor PUTs through setCropPlate itself, so just advance.
    _advancePastPlate(current.id);
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

  // Keyboard shortcuts. Per-class letter hotkeys (configured on /classes)
  // are routed through dropOnClassStore by the layout-level keydown
  // listener and work on every tab. The shortcuts below are the
  // tab-action shortcuts; on the plates tab Enter/D get rebound to plate
  // confirm/reject so the same finger pattern works for both flows.
  $effect(() => {
    const offs: Array<() => void> = [];
    const reg = (combo: string, fn: () => void | Promise<void>, desc: string) =>
      offs.push(keyboardStore.register(combo, () => void fn(), 'review', desc));

    if (tab === 'plates') {
      reg('enter', confirmPlate, 'Confirm plate & advance');
      reg('e', openPlateEditor, 'Edit plate bbox');
      reg('d', rejectPlate, 'Reject (no plate visible)');
    } else {
      reg('enter', confirmAndAdvance, 'Confirm proposed & advance');
      reg('d', discard, 'Discard');
    }
    reg('n', skip, 'Skip');
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
      {#if tab === 'plates'}
        <kbd>Enter</kbd> confirm plate · <kbd>E</kbd> edit · <kbd>D</kbd> reject ·
        <kbd>N</kbd> skip · <kbd>Z</kbd> undo
      {:else}
        per-class letter assigns · <kbd>Enter</kbd> confirm · <kbd>N</kbd> skip ·
        <kbd>D</kbd> discard · <kbd>Z</kbd> undo
      {/if}
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
            loading="lazy"
            decoding="async"
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
            loading="lazy"
            decoding="async"
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

        {#if tab === 'plates'}
          <!-- Plate-detection review row. The proposal came from
               LPR/SAM3 + Gemma but didn't clear the auto-confirm bar.
               Confirm = accept the proposed bbox as-is, Edit = open the
               drag-and-resize editor, Reject = mark "no plate visible"
               so the export doesn't emit a bogus plate label row. -->
          <div class="mt-3 grid grid-cols-2 gap-y-1 text-xs">
            <span class="text-zinc-500">Plate score</span>
            <span class="font-mono text-zinc-200">
              {current.plate_score != null
                ? `${(current.plate_score * 100).toFixed(1)}%`
                : '—'}
            </span>
            <span class="text-zinc-500">Plate status</span>
            <span class="text-zinc-200">{current.plate_status ?? '—'}</span>
          </div>
          <div class="mt-3 flex flex-wrap gap-2">
            <button class="btn btn-primary" type="button" onclick={confirmPlate}>
              Confirm Plate
            </button>
            <button class="btn" type="button" onclick={openPlateEditor}>Edit</button>
            <button class="btn btn-danger" type="button" onclick={rejectPlate}>
              Reject (no plate)
            </button>
            <button class="btn" type="button" onclick={skip}>Skip</button>
            <button class="btn" type="button" onclick={undoLast}>Undo</button>
          </div>
        {:else}
          <div class="mt-3 flex flex-wrap gap-2">
            <button class="btn btn-primary" type="button" onclick={confirmAndAdvance}>
              Confirm
            </button>
            <button class="btn" type="button" onclick={skip}>Skip</button>
            <button class="btn btn-danger" type="button" onclick={discard}>Discard</button>
            <button class="btn" type="button" onclick={undoLast}>Undo</button>
          </div>
        {/if}

        <!-- Most-validated classes — click to label OR press the per-class
             hotkey configured on /classes. Hotkey badges only show for
             classes the user has explicitly bound (otherwise the strip is
             still clickable, just no kbd hint). The class strip is hidden
             on the plates tab; class assignment isn't relevant there. -->
        {#if tab !== 'plates'}
        <div class="mt-3 flex flex-wrap gap-1.5">
          {#each topClasses as cls (cls.id)}
            <button
              type="button"
              class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200
                     hover:border-blue-500/60 hover:bg-blue-500/10 hover:text-white
                     focus:outline-none focus:ring-2 focus:ring-blue-500/40"
              title={cls.hotkey_letter
                ? `Assign ${cls.name} (press ${cls.hotkey_letter})`
                : `Assign ${cls.name}`}
              onclick={() => assign(cls.id)}
            >
              {#if cls.hotkey_letter}
                <kbd
                  class="mr-1.5 rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] uppercase text-blue-300"
                >
                  {cls.hotkey_letter}
                </kbd>
              {/if}
              {cls.name}
            </button>
          {/each}
        </div>
        <p class="mt-1.5 text-[10px] text-zinc-500">
          Click a class or press its bound letter (set hotkeys on /classes).
        </p>
        {/if}
      </div>
    {/if}
  </div>

  {#if plateEditorOpen && current}
    <PlateEditor
      crop={current}
      onsave={onPlateEditorSave}
      onclose={() => {
        plateEditorOpen = false;
      }}
    />
  {/if}

  <!-- Status bar — review is one-at-a-time so there's no "scroll to load more"
       affordance; the queue auto-fetches the next page in the background as
       the cursor advances toward the end of the loaded items. -->
  <div
    class="flex items-center justify-between gap-3 border-t border-zinc-800 px-4 py-2 text-sm"
  >
    <span class="font-mono text-xs text-zinc-500">
      {Math.min(cursor + 1, items.length)} / {total}
      {#if items.length < total}
        <span class="ml-1 text-zinc-600">(loaded {items.length})</span>
      {/if}
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if loadingMore}loading more…{:else if !hasMore && items.length > 0}all loaded{:else if hasMore}auto-fetching{/if}
    </span>
  </div>
</div>
