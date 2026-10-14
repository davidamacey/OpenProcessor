/**
 * The /classes merge dialog: the source/target pair, the served dry-run
 * preview (re-requested whenever the pair changes) and the real merge.
 * `busy` is the page's shared write flag, read and set through the handle.
 */
import { apiErrorText, mergeClasses, previewClassMerge } from '$lib/api';
import type { ClassMergeDryRun } from '$lib/types';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';

export interface MergeClassesState {
  busy: boolean;
}

export function createMergeClassesController(s: MergeClassesState) {
  let mergeOpen = $state<boolean>(false);
  // Merge form
  let mergeSourceId = $state<number | null>(null);
  let mergeTargetId = $state<number | null>(null);
  let mergePreview = $state<ClassMergeDryRun | null>(null);
  let mergePreviewError = $state<string | null>(null);
  let mergePreviewBusy = $state<boolean>(false);

  function openMerge(): void {
    mergeSourceId = null;
    mergeTargetId = null;
    mergePreview = null;
    mergePreviewError = null;
    mergeOpen = true;
  }

  function openMergeWithSource(sourceId: number): void {
    openMerge();
    mergeSourceId = sourceId;
  }

  function closeMerge(): void {
    if (s.busy) return;
    mergeOpen = false;
  }

  const mergeSource = $derived(
    mergeSourceId != null
      ? (classesStore.classes.find((c) => c.id === mergeSourceId) ?? null)
      : null,
  );
  const mergeTarget = $derived(
    mergeTargetId != null
      ? (classesStore.classes.find((c) => c.id === mergeTargetId) ?? null)
      : null,
  );

  // Dry-run preview: every time the pair changes, ask the server what a
  // real merge would do (would_relabel / validations_carried_over / holdout_blocking
  // / blocked) before it's possible to confirm. Never guessed client-side —
  // the backend already knows about holdout-blocking rows the frontend has
  // no visibility into.
  $effect(() => {
    const sourceId = mergeSourceId;
    const targetId = mergeTargetId;
    mergePreview = null;
    mergePreviewError = null;
    if (!mergeOpen || sourceId == null || targetId == null || sourceId === targetId)
      return;
    const ctrl = new AbortController();
    mergePreviewBusy = true;
    void previewClassMerge({ source_id: sourceId, target_id: targetId }, ctrl.signal)
      .then((res) => {
        mergePreview = res;
      })
      .catch((e: unknown) => {
        if ((e as Error)?.name === 'AbortError') return;
        mergePreviewError = apiErrorText(e);
      })
      .finally(() => {
        mergePreviewBusy = false;
      });
    return () => ctrl.abort();
  });

  async function submitMerge(): Promise<void> {
    if (mergeSourceId == null || mergeTargetId == null) {
      toastStore.error('Pick both source and target classes.');
      return;
    }
    if (mergeSourceId === mergeTargetId) {
      toastStore.error('Source and target must differ.');
      return;
    }
    if (mergePreview == null) return;
    if (mergePreview?.blocked) {
      toastStore.error('Merge is blocked — resolve the holdout conflict first.');
      return;
    }
    const ok = window.confirm(
      `Merge "${mergeSource?.name}" into "${mergeTarget?.name}"? This relabels ` +
        `${mergePreview.would_relabel} crops and marks the ` +
        'source deprecated. The action is recorded in the registry but cannot be undone from the UI.',
    );
    if (!ok) return;
    s.busy = true;
    try {
      const res = await mergeClasses({
        source_id: mergeSourceId,
        target_id: mergeTargetId,
      });
      toastStore.success(`Merged '${res.source_name}' into '${res.target_name}'.`);
      mergeOpen = false;
      await classesStore.clearAndRefetch();
    } catch (e) {
      toastStore.error(`Merge failed: ${apiErrorText(e)}`);
    } finally {
      s.busy = false;
    }
  }

  return {
    get mergeOpen() {
      return mergeOpen;
    },
    get mergeSourceId() {
      return mergeSourceId;
    },
    set mergeSourceId(v: number | null) {
      mergeSourceId = v;
    },
    get mergeTargetId() {
      return mergeTargetId;
    },
    set mergeTargetId(v: number | null) {
      mergeTargetId = v;
    },
    get mergePreview() {
      return mergePreview;
    },
    get mergePreviewError() {
      return mergePreviewError;
    },
    get mergePreviewBusy() {
      return mergePreviewBusy;
    },
    get mergeSource() {
      return mergeSource;
    },
    get mergeTarget() {
      return mergeTarget;
    },
    openMerge,
    openMergeWithSource,
    closeMerge,
    submitMerge,
  };
}

export type MergeClassesController = ReturnType<typeof createMergeClassesController>;
