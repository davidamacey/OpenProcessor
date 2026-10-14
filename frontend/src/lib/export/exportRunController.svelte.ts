/**
 * /export's YOLO export run: the export options, the served export
 * status, the progress modal, and the status poll used only when the
 * backend reports the export still running. The owner calls `dispose()`
 * on teardown so no poll outlives the page.
 */
import { apiErrorText, exportStatus, exportYolo } from '$lib/api';
import { splitExportClasses } from '$lib/export/exportDatasetRows';
import { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';
import { toastStore } from '$stores/toast.svelte';
import type { ExportStatus } from '$lib/types';

export function downloadUrl(url: string, suggestedName: string): void {
  const a = document.createElement('a');
  a.href = url;
  a.download = suggestedName;
  a.target = '_blank';
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** `loadAll` re-reads the page's data after a finished export. */
export function createExportRunController(loadAll: () => Promise<void>) {
  let versionTag = $state<string>('');
  // OpenProcessor 4c9499a: opt-in — drop any exported image that still
  // has an unlabeled object on it, rather than teaching the detector to
  // treat that object as background.
  let requireFullyLabeled = $state<boolean>(false);
  // Optional: limit the export to the items a filter matches. Empty sends
  // no `item_filter` at all (the whole dataset).
  const exportFilter = new ItemFilterState();
  let exportRunning = $state<boolean>(false);
  let exportState = $state<ExportStatus | null>(null);
  let pollHandle: ReturnType<typeof setInterval> | null = null;
  let exportModalOpen = $state<boolean>(false);

  const exportClassSplit = $derived(
    splitExportClasses(exportState?.class_split_counts ?? []),
  );

  // The served pre-scan of `POST /export/yolo`: only an explicit `false`
  // blocks; `null` (the count failed) never does. Row-level refusals still
  // arrive as the export's own 422 and are rendered by `runExport`.
  const exportBlocked = $derived(exportState?.can_export === false);
  const blockingReasons = $derived(exportState?.blocking_reasons ?? []);

  function startPolling(): void {
    if (pollHandle) return;
    pollHandle = setInterval(async () => {
      try {
        const s = await exportStatus();
        exportState = s;
        if (s.status !== 'running' && s.status !== 'pending') {
          exportRunning = false;
          if (pollHandle) {
            clearInterval(pollHandle);
            pollHandle = null;
          }
          if (s.status === 'success') {
            toastStore.success(`Export complete: ${s.export_dir ?? 'see manifest'}`);
            await loadAll();
          } else if (s.status === 'failed') {
            toastStore.error(`Export failed: ${s.error ?? 'unknown'}`);
          }
        }
      } catch (e) {
        toastStore.warn(`Status poll failed: ${apiErrorText(e)}`);
      }
    }, 5000);
  }

  // `POST {API_PREFIX}/export/yolo` is synchronous (see `ExportResult`'s
  // doc comment in `$lib/types` — the OpenProcessor handler `await`s the
  // export before responding, there is no queued-job ack) — the response
  // IS the finished export, not a "started" acknowledgement. Render its
  // own served `status` directly instead of assuming `running` and
  // polling; only fall back to polling when the backend itself reports
  // the job still `running`/`pending` (a future async backend, or a slow
  // one that didn't finish inline).
  async function runExport(): Promise<void> {
    exportRunning = true;
    exportModalOpen = true;
    try {
      const res = await exportYolo({
        version_tag: versionTag.trim() || undefined,
        require_fully_labeled_images: requireFullyLabeled,
        item_filter: exportFilter.toBody(),
      });
      exportState = {
        status: res.status,
        // eslint-disable-next-line svelte/prefer-svelte-reactivity -- converted to an ISO string at once, never stored or mutated
        last_run: res.finished_at ?? res.started_at ?? new Date().toISOString(),
        export_dir: res.export_dir ?? null,
        error: res.status === 'failed' ? (res.message ?? 'unknown error') : null,
        message: res.message ?? null,
        image_count: res.image_count ?? null,
        object_count: res.object_count ?? null,
        split_object_counts:
          (res.split_object_counts as ExportStatus['split_object_counts']) ?? null,
        require_fully_labeled_images: res.require_fully_labeled_images ?? null,
        unlabeled_items_on_exported_images:
          res.unlabeled_items_on_exported_images ?? null,
        images_with_unlabeled_items: res.images_with_unlabeled_items ?? null,
        images_dropped_not_fully_labeled: res.images_dropped_not_fully_labeled ?? null,
        skipped_items: res.skipped_items ?? null,
      };
      if (res.status === 'running' || res.status === 'pending') {
        toastStore.info(`Export started: ${res.status}`);
        startPolling();
        return;
      }
      exportRunning = false;
      if (res.status === 'success') {
        toastStore.success(`Export complete: ${res.export_dir ?? res.status}`);
        // Bug 1: the "frozen multi-class export" card/registry download
        // buttons are gated on `hasMulticlassExport`, which only
        // `loadAll()` (re-fetching `{API_PREFIX}/export/datasets`) can
        // set — reload now instead of waiting for the user to hit Refresh.
        // loadAll() also re-fetches GET /export/status, which fills in
        // class_count/class_split_counts/group_key — fields the
        // synchronous POST response above doesn't carry.
        await loadAll();
      } else if (res.status === 'failed') {
        toastStore.error(`Export failed: ${res.message ?? 'unknown error'}`);
      } else {
        toastStore.info(`Export: ${res.status}`);
      }
    } catch (e) {
      exportRunning = false;
      // 422 "nothing to export: <reason>" (e.g. require_fully_labeled_images
      // dropped every candidate image) surfaces via ApiError's detail text.
      toastStore.error(`Export failed: ${apiErrorText(e)}`);
    }
  }

  function closeExportModal(): void {
    if (exportRunning) return;
    exportModalOpen = false;
  }

  function dispose(): void {
    if (pollHandle) clearInterval(pollHandle);
    pollHandle = null;
  }

  return {
    get versionTag() {
      return versionTag;
    },
    set versionTag(v: string) {
      versionTag = v;
    },
    get requireFullyLabeled() {
      return requireFullyLabeled;
    },
    set requireFullyLabeled(v: boolean) {
      requireFullyLabeled = v;
    },
    exportFilter,
    get exportRunning() {
      return exportRunning;
    },
    get exportState() {
      return exportState;
    },
    set exportState(v: ExportStatus | null) {
      exportState = v;
    },
    get exportModalOpen() {
      return exportModalOpen;
    },
    get exportClassSplit() {
      return exportClassSplit;
    },
    get exportBlocked() {
      return exportBlocked;
    },
    get blockingReasons() {
      return blockingReasons;
    },
    runExport,
    closeExportModal,
    dispose,
  };
}

export type ExportRunController = ReturnType<typeof createExportRunController>;
