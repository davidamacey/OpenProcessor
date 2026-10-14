/**
 * /train's single-class dataset export (the export-capable slot), gated
 * on the dataset controller's capability flag. Call during component
 * init (it registers effects).
 */
import { apiErrorText } from '$lib/api_errors';
import { exportSingleClass, exportSingleClassStatus } from '$lib/api_export';
import { exportDedupDefault } from '$lib/strategies';
import { strategiesStore } from '$stores/strategies.svelte';
import { toastStore } from '$stores/toast.svelte';
import type { TrainDatasetController } from './trainDatasetController.svelte';

export function createTrainSingleClassExport(dataset: TrainDatasetController) {
  const datasetExportSpec = dataset.datasetExportSpec;

  // ---- Single-class dataset export (the export-capable slot) ------------
  // Built on demand via the active slot's `extras.datasetExport` spec,
  // through OpenProcessor's generic `POST /export/single_class`. Backend
  // is synchronous, so we just await it and surface the resulting dir +
  // counts. Only the four typed option controls below stay hand-written —
  // see docs/design/bakeoff-train-genericization-plan-2026-09-21.md
  // §3.3/§3.4 for why generalizing them would produce a form builder with
  // exactly one form to build.
  let singleClassExporting = $state<boolean>(false);
  let singleClassExportMessage = $state<string | null>(null);
  // Export options. whole_frame = full source frame (deployment distribution);
  // item_crop = parent item crop with the region re-projected. 640 for a
  // fast pass, 1280 for the full run. dedup collapses near-dup frames at the served threshold.
  let singleClassImageMode = $state<'whole_frame' | 'item_crop'>('whole_frame');
  let singleClassImgSize = $state<640 | 1280>(1280);
  let singleClassDedup = $state<boolean>(true);
  // The served threshold of this export kind; none = no dedup option.
  const singleClassDedupThreshold = $derived(
    exportDedupDefault(strategiesStore.methods.dataset_exports, datasetExportSpec?.kind),
  );
  // Optional N: sample at most this many positive (region-bearing) frames,
  // spread EVENLY across region clusters. Blank/0 == every positive. Lets us
  // build progressively larger dataset versions from the same labeled pool.
  let singleClassMaxPositives = $state<number | null>(null);

  async function refreshSingleClassExportStatus(): Promise<void> {
    try {
      if (!datasetExportSpec) return;
      const s = await exportSingleClassStatus(datasetExportSpec);
      dataset.singleClassExportDir = s.export_dir ?? '';
    } catch {
      // Non-fatal — the export just hasn't run yet.
    }
  }

  // Only ask for export status once the capability gate says the route
  // exists. Firing it unconditionally on mount would 404 on a backend
  // that doesn't advertise this export kind, and
  // refreshSingleClassExportStatus()'s bare `catch {}` would make that
  // invisible outside the network tab. Plain `let`, not `$state` — writing it must
  // not re-trigger this effect.
  let singleClassStatusRequested = false;
  $effect(() => {
    if (dataset.datasetExportAvailable && !singleClassStatusRequested) {
      singleClassStatusRequested = true;
      void refreshSingleClassExportStatus();
    }
  });

  async function runSingleClassExport(): Promise<void> {
    // Non-null: this handler only runs from a button inside
    // `{#if datasetExportAvailable && datasetExportSpec}`.
    const spec = datasetExportSpec!;
    singleClassExporting = true;
    singleClassExportMessage = null;
    try {
      const r = await exportSingleClass(spec, {
        image_mode: singleClassImageMode,
        img_max_side: singleClassImgSize,
        dedup_threshold: singleClassDedup ? singleClassDedupThreshold : null,
        max_positive_images:
          singleClassMaxPositives && singleClassMaxPositives > 0
            ? singleClassMaxPositives
            : undefined,
      });
      dataset.singleClassExportDir = r.export_dir;
      // Refresh the picker so the new version shows up immediately.
      void dataset.refreshDataset();
      const pos = r.positive_images ?? '?';
      const bg = r.background_images ?? '?';
      singleClassExportMessage = `${spec.label} done — ${r.image_count} images (${pos} positives, ${bg} backgrounds), ${singleClassImageMode} @ ${singleClassImgSize}px${singleClassDedup && singleClassDedupThreshold != null ? `, dedup ${singleClassDedupThreshold}` : ''}. dataset_sha ${r.dataset_sha.slice(0, 12)}`;
      if (r.positives_zero_warning) {
        singleClassExportMessage +=
          ' Warning: zero positives — check labeling for this slot.';
      }
      toastStore.success(`${spec.label} complete`);
    } catch (e) {
      singleClassExportMessage = `${spec.label} failed: ${apiErrorText(e)}`;
      toastStore.error(`${spec.label} failed`);
    } finally {
      singleClassExporting = false;
    }
  }

  return {
    get singleClassExporting() {
      return singleClassExporting;
    },
    get singleClassExportMessage() {
      return singleClassExportMessage;
    },
    get singleClassImageMode() {
      return singleClassImageMode;
    },
    set singleClassImageMode(v: 'whole_frame' | 'item_crop') {
      singleClassImageMode = v;
    },
    get singleClassImgSize() {
      return singleClassImgSize;
    },
    set singleClassImgSize(v: 640 | 1280) {
      singleClassImgSize = v;
    },
    get singleClassDedup() {
      return singleClassDedup;
    },
    set singleClassDedup(v: boolean) {
      singleClassDedup = v;
    },
    get singleClassDedupThreshold() {
      return singleClassDedupThreshold;
    },
    get singleClassMaxPositives() {
      return singleClassMaxPositives;
    },
    set singleClassMaxPositives(v: number | null) {
      singleClassMaxPositives = v;
    },
    runSingleClassExport,
  };
}

export type TrainSingleClassExport = ReturnType<typeof createTrainSingleClassExport>;
