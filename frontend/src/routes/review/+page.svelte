<script lang="ts">
  import {
    deleteCropLabel,
    reviewDismissCrop,
    getCrop,
    getReviewQueue,
    getSourceImageWithBbox,
    getThumbUrl,
    putCropLabel,
    setCropPlate,
    updateCropPlateMeta,
    type PlateMetaPatch,
  } from '$lib/api';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import DetectorChip from '$lib/components/DetectorChip.svelte';
  import PlateBboxCanvas from '$lib/components/PlateBboxCanvas.svelte';
  import {
    bboxNormToXYXY,
    cropToSourceFrame,
    sourceToCropFrame,
  } from '$lib/plate_geometry';
  import type { BBoxNorm, OpClass, ReviewItem, ReviewTab, UndoEntry } from '$lib/types';
  import { subscribeKbEvents, type OpEventSubscription } from '$lib/sse';
  import { untrack } from 'svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { undoStore } from '$stores/undo.svelte';
  import { onMount } from 'svelte';

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
    // Primary-subject active-learning queues — the highest-value crops to
    // label for the next v6 pass (largest subjects v6 was unsure on, and
    // COCO-confirmed vehicles v6 missed entirely).
    { id: 'primary_low_conf', label: 'Primary · Low-Conf' },
    { id: 'coco_blind_spots', label: 'COCO Blind Spots' },
  ];

  // Which tabs honor the primary-subject controls (rank toggle + clarity).
  const PRIMARY_TABS: ReviewTab[] = ['primary_low_conf', 'coco_blind_spots'];

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

  // Eagerly prefetch the next page when the cursor is within this many
  // items of the end of the loaded buffer. Without this, the user sees
  // 'no more' for one render-frame whenever they confirm the last item
  // we've loaded — refreshing the page then shows there were more all
  // along.
  const PREFETCH_AHEAD = 5;
  function maybePrefetch(): void {
    if (loadingMore || !hasMore) return;
    if (items.length - cursor <= PREFETCH_AHEAD) {
      void loadMore();
    }
  }

  // Filter bar
  let hddSource = $state<string>('');
  let classFilter = $state<number | null>(null);
  let confMin = $state<number>(0);
  let confMax = $state<number>(1);
  // Plate-text search — only meaningful on tab=plates; ignored elsewhere
  // server-side. Surface in the filter strip when the operator is on
  // the plates tab.
  let plateTextQuery = $state<string>('');

  // Primary-subject controls (primary_low_conf / coco_blind_spots tabs).
  // subjectScope: 1 = largest only, 2 = largest + 2nd (the tabs default to 2
  // server-side when unset). Clarity slider commits on release.
  let subjectScope = $state<0 | 1 | 2>(0);
  const BLUR_MAX = 2;
  let blurSlider = $state<number>(0);
  let minBlurRatio = $state<number | null>(null);
  function commitBlur(): void {
    minBlurRatio = blurSlider > 0 ? blurSlider : null;
  }

  function _filter(): Record<string, unknown> {
    const f: Record<string, unknown> = {};
    if (hddSource) f.hdd_source = hddSource;
    if (classFilter != null) f.class_id = classFilter;
    if (confMin > 0) f.conf_min = confMin;
    if (confMax < 1) f.conf_max = confMax;
    if (tab === 'plates' && plateTextQuery) f.text = plateTextQuery;
    if (PRIMARY_TABS.includes(tab)) {
      if (subjectScope !== 0) f.max_rank = subjectScope;
      if (minBlurRatio != null) f.min_blur_ratio = minBlurRatio;
    }
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

  // -- SSE: live updates as ingest + workers classify new crops -------
  // We don't fetch each new crop individually (no single-crop endpoint
  // is exposed); instead we count incoming crop.* events and surface a
  // "X new crops" pill the user can click to refresh the queue. Auto-
  // refreshing while the operator is mid-keystroke would be jarring —
  // they decide when to pull in the new batch.
  let liveNewCount = $state<number>(0);
  let liveSub: OpEventSubscription | null = null;
  onMount(() => {
    liveSub = subscribeKbEvents({
      // Both classification + plate-verify changes are interesting on
      // the review page — the operator may be on any tab.
      onEvent: (ev) => {
        if (
          ev.type === 'crop.classified' ||
          ev.type === 'crop.created' ||
          ev.type === 'crop.plate_verified'
        ) {
          liveNewCount += 1;
        }
      },
    });
    return () => {
      liveSub?.close();
      liveSub = null;
    };
  });

  function refreshFromLive(): void {
    liveNewCount = 0;
    void loadFirst();
  }

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
    void subjectScope;
    void minBlurRatio;
    void loadFirst();
  });
  let filterDebounce: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    void hddSource;
    void plateTextQuery;
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
    maybePrefetch();
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
    maybePrefetch();
  }

  async function discard(): Promise<void> {
    if (!current) return;
    // Discard = "permanently dismiss this crop from every review queue."
    // Stamps review_dismissed_at on the backend; the review queue's
    // must_not filter excludes any crop with that field set. The
    // original class / plate state is preserved (this is NOT an
    // unlabel — use Z to undo if dismissed by mistake).
    undoStore.push(snap(current));
    const id = current.id;
    items = items.filter((x) => x.id !== id);
    total = Math.max(0, total - 1);
    cursor = Math.min(cursor, Math.max(0, items.length - 1));
    maybePrefetch();
    try {
      await reviewDismissCrop(id);
      toastStore.success('Dismissed from review. Press Z to undo.');
    } catch (e) {
      toastStore.error(`Discard failed: ${(e as Error).message}`);
    }
  }

  // -- plate-tab actions ----------------------------------------------
  // Inline editor — no modal. The canvas is always live; if the user
  // tweaks the proposed bbox, Confirm saves the edited version. If they
  // leave it alone, Confirm saves the proposal as-is. The goal is one
  // keystroke (Enter) per plate when scanning thousands of crops.
  //
  // editedPlateLocal lives in the *crop-local* frame (the same space the
  // PlateBboxCanvas operates in). We seed it from current.plate_bbox_norm
  // (source-frame) by projecting through the parent vehicle bbox; the
  // seeding effect re-runs whenever the cursor advances to a new crop.
  let editedPlateLocal = $state<BBoxNorm | null>(null);
  let plateCanvas = $state<{ handleKey: (e: KeyboardEvent) => boolean } | null>(
    null,
  );
  // Read-only by default: the canvas only becomes interactive when the
  // operator presses E (or clicks Edit bbox). Most cascade-detected
  // plates are already correct — forcing the heavy drag-handle UI on
  // every crop is what made the tab feel "weird" vs. the other review
  // tabs. Edit mode resets to false on every cursor advance so the
  // operator always lands on the next plate in scan-and-confirm mode.
  let editMode = $state<boolean>(false);
  let plateSaving = $state<boolean>(false);

  // Inline editors for the plate metadata fields. Seeded from the
  // current crop on every cursor advance; saved on blur / Enter via
  // PATCH /curation/crops/{id}/plate_meta. Each field saves independently
  // with optimistic-UI + revert-on-error, matching the assign() pattern.
  let editedPlateText = $state<string>('');
  let editedPlateStatus = $state<string>('');
  let editedRejectionReason = $state<string>('');
  // Status values an operator is allowed to write; mirrors
  // HUMAN_PLATE_STATUS_VALUES in openprocessor legacy.py. Kept inline
  // since it's a tiny set and adding a $lib/constants file for three
  // strings is overkill.
  const PLATE_STATUS_OPTIONS: Array<{ value: string; label: string }> = [
    { value: 'detected', label: 'detected (plate visible)' },
    { value: 'no_plate_visible', label: 'no plate visible' },
    { value: 'verify_rejected', label: 'rejected (bad detection)' },
    { value: 'false_positive', label: 'false positive (keep box)' },
  ];

  // Undo stack for plate confirm/reject. Each entry holds the previously
  // confirmed plate so "Back" can re-insert the crop into the queue and
  // restore the bbox the user just saved (allowing them to fix a mistake
  // without re-finding the crop). Bounded to 20 entries — enough for
  // half a session of confusion, small enough to keep memory tiny.
  interface PlateUndoEntry {
    item: ReviewItem;
    insertAt: number;
    /**
     * The plate bbox in source-frame that was sent to the server for
     * this confirm — null means "rejected" (no plate visible).
     */
    saved: [number, number, number, number] | null;
  }
  let plateUndoStack = $state<PlateUndoEntry[]>([]);
  const PLATE_UNDO_MAX = 20;
  function _pushPlateUndo(entry: PlateUndoEntry): void {
    plateUndoStack = [...plateUndoStack, entry].slice(-PLATE_UNDO_MAX);
  }

  async function plateBack(): Promise<void> {
    const last = plateUndoStack[plateUndoStack.length - 1];
    if (!last) {
      toastStore.info('Nothing to go back to.');
      return;
    }
    plateUndoStack = plateUndoStack.slice(0, -1);
    // Refetch the crop so the operator sees what the database actually
    // holds — the local snapshot can lag (e.g. another worker re-ran
    // OCR or another curator edited concurrently). This is the
    // "confidence in changes" guarantee the user asked for.
    let fresh: ReviewItem;
    try {
      const c = await getCrop(last.item.id);
      // The /curation/crops/{id} endpoint returns a OpCrop, but the review
      // queue carries extra fields (reason, proposed_*). Keep the
      // snapshot's queue-only metadata and overlay the authoritative
      // store fields on top.
      fresh = { ...last.item, ...c } as ReviewItem;
    } catch (e) {
      toastStore.warn(
        `Re-fetch failed; restoring local snapshot: ${(e as Error).message}`,
      );
      fresh = last.item;
    }
    const insertAt = Math.min(last.insertAt, items.length);
    const next = [...items];
    next.splice(insertAt, 0, fresh);
    items = next;
    total = total + 1;
    cursor = insertAt;
    toastStore.info('Stepped back. Press E to re-edit, Enter to re-confirm.');
  }

  function _seedPlateFromCurrent(): void {
    if (!current || !current.plate_bbox_norm || !current.bbox_norm) {
      editedPlateLocal = null;
      return;
    }
    editedPlateLocal = sourceToCropFrame(current.plate_bbox_norm, current.bbox_norm);
  }

  // Plate-centered viewport for the right-side canvas. **Frozen** —
  // computed once when the crop loads and held steady during edits.
  // If we derived it from `editedPlateLocal` instead, every drag tick
  // would recompute the zoom and the IMG transform would pan/scale
  // along with the resize handle, making the box feel like it's
  // rubber-banding the whole image. The canvas applies viewBox as a
  // pure display transform; saved coordinates remain in crop-local
  // frame and project to source frame on confirm.
  const PLATE_VIEW_PADDING = 2.5;
  let plateViewBox = $state<BBoxNorm | null>(null);
  function _seedViewBox(): void {
    if (!editedPlateLocal) {
      plateViewBox = null;
      return;
    }
    const w0 = editedPlateLocal.w;
    const h0 = editedPlateLocal.h;
    if (w0 <= 0 || h0 <= 0) {
      plateViewBox = null;
      return;
    }
    // Expand by padding, then square the viewport (canvas is aspect-
    // square; non-square viewBox would re-introduce letterboxing).
    const side = Math.min(1, Math.max(w0, h0) * PLATE_VIEW_PADDING);
    const half = side / 2;
    const cx = Math.min(1 - half, Math.max(half, editedPlateLocal.cx));
    const cy = Math.min(1 - half, Math.max(half, editedPlateLocal.cy));
    plateViewBox = { cx, cy, w: side, h: side };
  }

  // Reseed whenever the cursor changes (advancing to next crop) or the
  // tab/items reset. Also exit edit mode so the next plate lands in
  // read-only scan mode regardless of where we left the previous one.
  $effect(() => {
    void current?.id;
    _seedPlateFromCurrent();
    // Freeze the zoom viewport on the just-seeded bbox. Wrapped in
    // untrack() so the read of `editedPlateLocal` inside _seedViewBox
    // does NOT make this effect re-run on every drag tick — that
    // would re-fire _seedPlateFromCurrent and overwrite the user's
    // in-progress resize with the server snapshot ("can't edit the
    // bbox" bug).
    untrack(() => _seedViewBox());
    editedPlateText = current?.plate_text ?? '';
    editedPlateStatus = current?.plate_status ?? '';
    editedRejectionReason = current?.plate_rejection_reason ?? '';
    editMode = false;
  });

  // In-flight plate-meta saves, keyed by crop id so concurrent edits to
  // the same crop are aborted-then-replaced (the latest blur wins) and
  // edits to a *different* crop don't interfere with each other.
  const plateMetaAborts = new Map<string, AbortController>();

  async function savePlateMeta(patch: PlateMetaPatch, snapshot: Partial<ReviewItem>): Promise<void> {
    if (!current) return;
    const id = current.id;
    // Look up by id, not cursor — if the user advances mid-save the
    // captured idx would point at the next crop and the revert would
    // corrupt unrelated state.
    const findIdx = () => items.findIndex((x) => x.id === id);
    const idx0 = findIdx();
    const prior: Partial<ReviewItem> = {};
    if (idx0 >= 0) {
      for (const k of Object.keys(snapshot) as Array<keyof ReviewItem>) {
        (prior as Record<string, unknown>)[k] = items[idx0][k];
      }
      items[idx0] = { ...items[idx0], ...snapshot } as ReviewItem;
    }
    // Abort any in-flight save on this crop so we don't get an ABA-style
    // response that overwrites a newer edit.
    plateMetaAborts.get(id)?.abort();
    const ac = new AbortController();
    plateMetaAborts.set(id, ac);
    try {
      await updateCropPlateMeta(id, patch, ac.signal);
    } catch (e) {
      if (ac.signal.aborted) return; // superseded by a newer save
      const idx1 = findIdx();
      if (idx1 >= 0) items[idx1] = { ...items[idx1], ...prior } as ReviewItem;
      // Reseed local inputs only if we're still on the same crop the
      // user was editing; otherwise leave the inputs alone — they're
      // already bound to the new crop's state.
      if (current?.id === id) {
        editedPlateText = items[idx1]?.plate_text ?? '';
        editedPlateStatus = items[idx1]?.plate_status ?? '';
        editedRejectionReason = items[idx1]?.plate_rejection_reason ?? '';
      }
      toastStore.error(`Save failed: ${(e as Error).message}`);
    } finally {
      if (plateMetaAborts.get(id) === ac) plateMetaAborts.delete(id);
    }
  }

  async function commitPlateText(): Promise<void> {
    if (!current) return;
    const next = editedPlateText.trim() || null;
    if ((current.plate_text ?? null) === next) return;
    await savePlateMeta(
      { plate_text: next },
      { plate_text: next, plate_text_source: 'human', plate_text_confidence: next ? 1.0 : null },
    );
  }

  async function commitPlateStatus(): Promise<void> {
    if (!current) return;
    if (!editedPlateStatus) return;
    if (editedPlateStatus === current.plate_status) return;
    // 'no_plate_visible' implies the bbox is gone — call setCropPlate
    // null to keep the bbox + status in sync (avoids the contradiction
    // of "no_plate_visible" with a populated plate_bbox_norm).
    if (editedPlateStatus === 'no_plate_visible') {
      try {
        await setCropPlate(current.id, null);
        const idx = items.findIndex((x) => x.id === current.id);
        if (idx >= 0) {
          items[idx] = {
            ...items[idx],
            plate_bbox_norm: null,
            plate_status: 'no_plate_visible',
          } as ReviewItem;
        }
        editedPlateLocal = null;
      } catch (e) {
        toastStore.error(`Save failed: ${(e as Error).message}`);
      }
      return;
    }
    await savePlateMeta(
      { plate_status: editedPlateStatus as PlateMetaPatch['plate_status'] },
      { plate_status: editedPlateStatus },
    );
  }

  async function commitRejectionReason(): Promise<void> {
    if (!current) return;
    const next = editedRejectionReason.trim() || null;
    if ((current.plate_rejection_reason ?? null) === next) return;
    await savePlateMeta(
      { plate_rejection_reason: next },
      { plate_rejection_reason: next },
    );
  }

  function toggleEdit(): void {
    if (!current) return;
    if (editMode) {
      // Cancel-style exit: drop local edits and reseed from server state.
      _seedPlateFromCurrent();
      _seedViewBox();
      editMode = false;
      return;
    }
    // Re-center the zoom on whatever bbox we're about to edit (could
    // differ from the cursor-advance snapshot if the user already saved
    // once on this crop and is re-editing).
    _seedViewBox();
    editMode = true;
  }

  async function saveBboxAndExit(): Promise<void> {
    if (!current) return;
    if (!editedPlateLocal) {
      toastStore.warn('No bbox to save — draw one or press Backspace to clear.');
      return;
    }
    if (!current.bbox_norm) {
      toastStore.error('Missing parent vehicle bbox; cannot project to source frame.');
      return;
    }
    const id = current.id;
    const sourceBox = cropToSourceFrame(editedPlateLocal, current.bbox_norm);
    const tuple = bboxNormToXYXY(sourceBox) as [number, number, number, number];
    plateSaving = true;
    try {
      await setCropPlate(id, tuple);
      // Server flips plate_status to 'detected'/'human_confirmed' on bbox
      // write; reflect that locally without waiting for a queue refetch.
      const idx = items.findIndex((x) => x.id === id);
      if (idx >= 0) {
        items[idx] = {
          ...items[idx],
          plate_bbox_norm: sourceBox,
          plate_status: 'detected',
          plate_verified: true,
        };
      }
      editMode = false;
      toastStore.success('Bbox saved.');
    } catch (e) {
      toastStore.error(`Save failed: ${(e as Error).message}`);
    } finally {
      plateSaving = false;
    }
  }

  function _advancePastPlate(id: string): void {
    items = items.filter((x) => x.id !== id);
    total = Math.max(0, total - 1);
    cursor = Math.min(cursor, Math.max(0, items.length - 1));
    maybePrefetch();
  }

  async function confirmPlate(): Promise<void> {
    if (!current) return;
    if (!editedPlateLocal) {
      toastStore.warn('No plate bbox to confirm — drag one in or press D to reject.');
      return;
    }
    if (!current.bbox_norm) {
      toastStore.error('Missing parent vehicle bbox; cannot project to source frame.');
      return;
    }
    const id = current.id;
    const sourceBox = cropToSourceFrame(editedPlateLocal, current.bbox_norm);
    const tuple = bboxNormToXYXY(sourceBox) as [number, number, number, number];
    // Snapshot for "Back" before mutating the queue.
    _pushPlateUndo({ item: current, insertAt: cursor, saved: tuple });
    _advancePastPlate(id);
    try {
      await setCropPlate(id, tuple);
      toastStore.success('Plate confirmed. ← to go back.');
    } catch (e) {
      toastStore.error(`Confirm failed: ${(e as Error).message}`);
    }
  }

  async function rejectPlate(): Promise<void> {
    if (!current) return;
    const id = current.id;
    _pushPlateUndo({ item: current, insertAt: cursor, saved: null });
    _advancePastPlate(id);
    try {
      // null bbox = "no plate visible" per setCropPlate contract.
      await setCropPlate(id, null);
      toastStore.success('Plate rejected. ← to go back.');
    } catch (e) {
      toastStore.error(`Reject failed: ${(e as Error).message}`);
    }
  }

  async function markFalsePositive(): Promise<void> {
    if (!current) return;
    const id = current.id;
    // False positive: a detector drew this box but it is NOT a plate.
    // We KEEP the box + all detection metadata (unlike Reject, which
    // clears it) — flipping only plate_status. The retained geometry
    // feeds FP analysis and becomes a hard negative in the dedicated
    // LPR training export.
    _pushPlateUndo({ item: current, insertAt: cursor, saved: null });
    _advancePastPlate(id);
    try {
      await updateCropPlateMeta(id, { plate_status: 'false_positive' });
      toastStore.success('Marked false positive (box kept). ← to go back.');
    } catch (e) {
      toastStore.error(`Mark FP failed: ${(e as Error).message}`);
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

  // Keyboard shortcuts. Per-class letter hotkeys (configured on /classes)
  // are routed through dropOnClassStore by the layout-level keydown
  // listener and work on every tab. The shortcuts below are the
  // tab-action shortcuts; on the plates tab Enter/D get rebound to plate
  // confirm/reject so the same finger pattern works for both flows.
  //
  // On the plates tab, behavior splits between read-only scan mode
  // (default) and edit mode (operator pressed E or Edit bbox):
  //   - read-only: arrows page the queue, Enter confirms-and-advances,
  //     E enters edit mode — matches the other review tabs.
  //   - edit:      arrows nudge the bbox, Enter saves+exits edit mode,
  //     Esc cancels edit, the bbox canvas owns the keystroke flow.
  $effect(() => {
    const offs: Array<() => void> = [];
    const reg = (combo: string, fn: () => void | Promise<void>, desc: string) =>
      offs.push(keyboardStore.register(combo, () => void fn(), 'review', desc));

    if (tab === 'plates') {
      if (editMode) {
        reg('enter', saveBboxAndExit, 'Save bbox & exit edit');
        reg('escape', toggleEdit, 'Cancel edit');
      } else {
        reg('enter', confirmPlate, 'Confirm plate & advance');
        reg('d', rejectPlate, 'Reject (no plate visible)');
        reg('f', markFalsePositive, 'False positive (keep box)');
        reg('e', toggleEdit, 'Edit bbox');
        // Back: re-insert the most-recently-confirmed plate so the operator
        // can correct mistakes without scrolling back through the queue.
        reg('arrowleft', plateBack, 'Back to last confirmed plate');
        reg('b', plateBack, 'Back (alias)');
        reg(
          'arrowright',
          () => {
            cursor = Math.min(items.length - 1, cursor + 1);
            maybePrefetch();
          },
          'Next item',
        );
      }
    } else {
      reg('enter', confirmAndAdvance, 'Confirm proposed & advance');
      reg('d', discard, 'Discard');
    }
    reg('n', skip, 'Skip');
    reg('z', undoLast, 'Undo last');

    let canvasKey: ((e: KeyboardEvent) => void) | null = null;
    if (tab === 'plates' && editMode) {
      // Edit mode only: forward bbox-fine-tune keys (arrows, [ / ],
      // Backspace) into the plate canvas. Outside edit mode arrows page
      // the queue like every other tab.
      canvasKey = (e: KeyboardEvent) => {
        if (!plateCanvas) return;
        const target = e.target as HTMLElement | null;
        if (target && /^(input|textarea|select)$/i.test(target.tagName)) return;
        if (plateCanvas.handleKey(e)) e.preventDefault();
      };
      window.addEventListener('keydown', canvasKey);
    } else if (tab !== 'plates') {
      // On non-plate tabs arrow keys navigate the queue.
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
          maybePrefetch();
        },
        'Next item',
      );
    }

    return () => {
      offs.forEach((off) => off());
      if (canvasKey) window.removeEventListener('keydown', canvasKey);
    };
  });
</script>

<div class="flex h-full flex-col">
  <!-- Tabs — horizontally scrollable on narrow viewports so all tabs stay reachable
       without colliding with the loaded-count chip on the right. -->
  <div class="flex items-center gap-1 border-b border-zinc-800 px-4">
    <div class="flex min-w-0 grow items-center gap-1 overflow-x-auto whitespace-nowrap">
      {#each TABS as t (t.id)}
        <button
          type="button"
          class="shrink-0 px-3 py-2.5 text-sm border-b-2 {tab === t.id
            ? 'border-blue-500 text-white'
            : 'border-transparent text-zinc-400 hover:text-zinc-200'}"
          onclick={() => {
            tab = t.id;
          }}
        >
          {t.label}
        </button>
      {/each}
    </div>
    <span class="shrink-0 pl-2 font-mono text-xs text-zinc-500">
      {items.length > 0 ? `${cursor + 1} / ${items.length}` : '—'} loaded · {total} total
    </span>
    {#if liveNewCount > 0}
      <button
        type="button"
        class="ml-2 shrink-0 animate-pulse rounded-full border border-blue-500/60 bg-blue-500/15 px-2.5 py-1 text-[11px] text-blue-200 hover:bg-blue-500/25"
        onclick={refreshFromLive}
        title="Reload the queue with the latest crops"
      >
        {liveNewCount} new · refresh
      </button>
    {/if}
  </div>

  <!-- Filter bar — flex children keep their width via flex-shrink-0; hotkey hint
       hides below md so it doesn't collide with controls on narrow viewports
       (same content is on the ~ overlay). -->
  <div
    class="flex min-w-0 flex-wrap items-center gap-3 border-b border-zinc-800 bg-zinc-900/40 px-4 py-2 text-xs"
  >
    <label class="flex shrink-0 items-center gap-1.5">
      <span class="text-zinc-400">HDD source</span>
      <input
        type="text"
        bind:value={hddSource}
        placeholder="any"
        class="w-32 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100 focus:border-blue-500 focus:outline-none"
      />
    </label>

    <label class="flex shrink-0 items-center gap-1.5">
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

    <label class="flex shrink-0 items-center gap-1.5">
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

    {#if tab === 'plates'}
      <label class="flex shrink-0 items-center gap-1.5">
        <span class="text-zinc-400">Plate text</span>
        <input
          type="text"
          bind:value={plateTextQuery}
          placeholder="e.g. S14"
          class="w-28 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
    {/if}

    {#if PRIMARY_TABS.includes(tab)}
      <div class="flex shrink-0 items-center gap-1.5">
        <span class="text-zinc-400">subject</span>
        <div class="inline-flex overflow-hidden rounded border border-zinc-700">
          {#each [{ v: 0, l: 'Top 2' }, { v: 1, l: 'Largest' }, { v: 2, l: '+2nd' }] as o (o.v)}
            <button
              type="button"
              class="px-2 py-1 {subjectScope === o.v
                ? 'bg-blue-600 text-white'
                : 'bg-zinc-900 text-zinc-300 hover:bg-zinc-700'}"
              onclick={() => (subjectScope = o.v as 0 | 1 | 2)}
            >
              {o.l}
            </button>
          {/each}
        </div>
      </div>
      <label class="flex shrink-0 items-center gap-1.5" title="Hide crops blurrier than this">
        <span class="text-zinc-400">clarity ≥</span>
        <input
          type="range"
          min="0"
          max={BLUR_MAX}
          step="0.05"
          bind:value={blurSlider}
          onchange={commitBlur}
          class="h-1 w-32 cursor-pointer accent-blue-500"
        />
        <span class="w-10 tabular-nums text-zinc-400">
          {blurSlider > 0 ? blurSlider.toFixed(2) : 'off'}
        </span>
      </label>
    {/if}

    <span class="grow"></span>

    <span class="hidden text-[11px] text-zinc-500 md:inline">
      {#if tab === 'plates' && editMode}
        <kbd>↑↓←→</kbd> nudge · <kbd>[ ]</kbd> right edge · <kbd>Enter</kbd> save ·
        <kbd>Esc</kbd> cancel
      {:else if tab === 'plates'}
        <kbd>Enter</kbd> confirm · <kbd>D</kbd> reject · <kbd>F</kbd> false-pos ·
        <kbd>E</kbd> edit · <kbd>N</kbd> skip · <kbd>←</kbd> back
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
            src={getSourceImageWithBbox(
              current.id,
              1280,
              // Cache-bust on plate-bbox edits so the burned-in overlay
              // refreshes after a save. updated_at would be nicer but
              // not every code path mutates it locally; bbox tuple is
              // a stable enough fingerprint.
              current.plate_bbox_norm
                ? `${current.plate_bbox_norm.cx.toFixed(4)},${current.plate_bbox_norm.cy.toFixed(4)},${current.plate_bbox_norm.w.toFixed(4)},${current.plate_bbox_norm.h.toFixed(4)}`
                : 'none',
            )}
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
          {#if tab === 'plates' && editMode}
            <!-- Edit mode — drag/resize the proposal directly, then hit
                 Enter to save. Square aspect keeps the canvas math
                 stable; the read-only default below shows the crop at
                 natural aspect to match the other review tabs. -->
            <PlateBboxCanvas
              bind:this={plateCanvas}
              cropId={current.id}
              bind:bbox={editedPlateLocal}
              viewBox={plateViewBox}
              busy={plateSaving}
              class="aspect-square w-auto h-full max-h-full min-w-0 max-w-full"
            />
          {:else if tab === 'plates'}
            <!-- Read-only default: same <img> layout as every other tab,
                 with a thin yellow ring overlay on the proposed bbox.
                 No grabbable handles, no pointer capture — the bbox is
                 just shown. Press E to edit. -->
            <PlateBboxCanvas
              cropId={current.id}
              bbox={editedPlateLocal}
              viewBox={plateViewBox}
              readonly
              class="aspect-square w-auto h-full max-h-full min-w-0 max-w-full"
            />
          {:else}
            <img
              src={getThumbUrl(current.id, 384)}
              alt="crop"
              loading="lazy"
              decoding="async"
              class="max-h-full max-w-full object-contain"
            />
          {/if}
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

          {#if current.coco_proposal_name}
            <dt class="text-zinc-500">COCO hint</dt>
            <dd>
              <span
                class="rounded border border-cyan-500/40 bg-cyan-500/15 px-1.5 py-0.5 text-[11px] text-cyan-200"
                title="COCO YOLO11 detected a vehicle here that v6 missed. Coarse class — pick the make below (bicycle/motorcycle/boat may be near one-click)."
              >
                {current.coco_proposal_name}
              </span>
            </dd>
          {/if}

          {#if current.crop_rank_in_image != null || current.blur_lap_ratio != null}
            <dt class="text-zinc-500">Rank · clarity</dt>
            <dd class="font-mono text-zinc-300">
              {current.crop_rank_in_image != null
                ? current.crop_rank_in_image === 1
                  ? '★1 largest'
                  : `#${current.crop_rank_in_image}`
                : '—'}
              {#if current.blur_lap_ratio != null}
                · b{current.blur_lap_ratio.toFixed(2)}
              {/if}
            </dd>
          {/if}
        </dl>

        {#if tab === 'plates'}
          <!-- Plate-detection inline review. The canvas above is live —
               drag/resize the proposal in place and hit Enter to confirm.
               The Reject button (or D) marks no_plate_visible. The whole
               flow is two keystrokes per crop on average: minor twitch
               with arrows / handles, then Enter. -->
          <div class="mt-3 grid grid-cols-2 gap-y-1 text-xs">
            <span class="text-zinc-500">Plate score</span>
            <span class="font-mono text-zinc-200">
              {current.plate_score != null
                ? `${(current.plate_score * 100).toFixed(1)}%`
                : '—'}
            </span>
            <span class="text-zinc-500">Plate status</span>
            <span>
              <select
                bind:value={editedPlateStatus}
                onchange={() => void commitPlateStatus()}
                class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
              >
                <option value="">—</option>
                {#each PLATE_STATUS_OPTIONS as opt (opt.value)}
                  <option value={opt.value}>{opt.label}</option>
                {/each}
              </select>
            </span>
            <span class="text-zinc-500">Detector</span>
            <span class="flex flex-wrap items-center gap-1.5">
              {#if current.plate_detector}
                <DetectorChip
                  detector={current.plate_detector}
                  version={current.plate_detector_version}
                />
                {#if current.plate_verifier}
                  <DetectorChip
                    detector={current.plate_verifier}
                    tag="verify"
                    version={current.plate_verifier_version}
                    size="sm"
                  />
                {/if}
              {:else}
                <span class="text-zinc-500">—</span>
              {/if}
              {#if current.plate_shape_warning}
                <span
                  class="rounded border border-yellow-500/60 bg-yellow-500/15 px-1.5 py-0.5 text-[10px] text-yellow-200"
                  title="Bbox shape fails the plate envelope (aspect ∉ [1.2, 8.0] or covers >50% of vehicle width). Likely legacy / corrupted data — press E to fix."
                >
                  ⚠ shape · press E to fix
                </span>
              {/if}
              {#if !editedPlateLocal && !editMode}
                <span
                  class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-[10px] text-zinc-400"
                  title="No plate bbox on this crop — press E to draw one."
                >
                  no bbox · press E to draw
                </span>
              {/if}
            </span>
            {#if current.plate_detector_chain && current.plate_detector_chain.length > 0}
              <span class="text-zinc-500">Cascade</span>
              <span class="flex flex-wrap items-center gap-1">
                {#each current.plate_detector_chain as entry (entry)}
                  <DetectorChip raw={entry} size="sm" />
                {/each}
              </span>
            {/if}
            <span class="text-zinc-500">Plate text</span>
            <span class="flex items-center gap-1.5">
              <input
                type="text"
                bind:value={editedPlateText}
                onblur={() => void commitPlateText()}
                onkeydown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    (e.currentTarget as HTMLInputElement).blur();
                  }
                }}
                placeholder="ABC123"
                spellcheck="false"
                autocapitalize="characters"
                class="w-28 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 font-mono text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
              />
              {#if current.plate_text_source}
                <DetectorChip detector={current.plate_text_source} size="sm" />
              {/if}
              {#if current.plate_text_confidence != null}
                <span class="text-[10px] text-zinc-500">
                  {(current.plate_text_confidence * 100).toFixed(0)}%
                </span>
              {/if}
            </span>
            {#if editedPlateStatus === 'verify_rejected' || editedPlateStatus === 'no_plate_visible'}
              <span class="text-zinc-500">Rejection reason</span>
              <span>
                <input
                  type="text"
                  bind:value={editedRejectionReason}
                  onblur={() => void commitRejectionReason()}
                  onkeydown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      (e.currentTarget as HTMLInputElement).blur();
                    }
                  }}
                  placeholder="e.g. blurred, occluded, glare"
                  class="w-44 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
                />
              </span>
            {/if}
          </div>
          <div class="mt-3 flex flex-wrap gap-2">
            {#if editMode}
              <button
                class="btn btn-primary"
                type="button"
                onclick={saveBboxAndExit}
                disabled={plateSaving}
              >
                Save bbox
              </button>
              <button class="btn" type="button" onclick={toggleEdit} disabled={plateSaving}>
                Cancel
              </button>
            {:else}
              <button class="btn btn-primary" type="button" onclick={confirmPlate}>
                Confirm Plate
              </button>
              <button class="btn btn-danger" type="button" onclick={rejectPlate}>
                Reject (no plate)
              </button>
              <button
                class="btn"
                type="button"
                onclick={markFalsePositive}
                title="Detector drew a box but it's not a plate — keep the box as a training hard negative (F)"
              >
                False positive
              </button>
              <button class="btn" type="button" onclick={skip}>Skip</button>
              <button
                class="btn"
                type="button"
                onclick={toggleEdit}
                aria-pressed={editMode}
                title="Toggle bbox edit mode (E)"
              >
                Edit bbox
              </button>
              <button
                class="btn"
                type="button"
                onclick={plateBack}
                disabled={plateUndoStack.length === 0}
                title="Re-open the most-recently confirmed plate (←)"
              >
                ← Back
              </button>
            {/if}
          </div>
          {#if plateUndoStack.length > 0}
            <p class="mt-1 text-[10px] text-zinc-500">
              {plateUndoStack.length} confirmed in this session — press ← to step back.
            </p>
          {/if}
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
