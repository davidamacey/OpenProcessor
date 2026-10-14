/**
 * /train's dataset picker: which frozen export the training run targets
 * (the multi-class `yolo` export or the registered slot's single-class
 * dataset), the materialized versions on disk, and the current export's
 * served counts. Call during component init (it registers effects).
 */
import { apiErrorText, exportStatus, listDatasets } from '$lib/api';
import { registeredSlots } from '$lib/annotations/registeredSlots';
import { datasetExportForSlot } from '$lib/annotations/datasetExport';
import { isDatasetExportAvailable } from '$lib/strategies';
import { strategiesStore } from '$stores/strategies.svelte';
import type { ExportDataset, ExportStatus } from '$lib/types';

// Which frozen export the training run targets: the multi-class dataset
// (`MULTI_CLASS`, the `yolo` export's `current`) or the registered slot's
// single-class dataset. `datasetKind` is `MULTI_CLASS` or whatever
// string the active slot's `extras.datasetExport.datasetKind` declares
// — never a hardcoded second literal, so a second registered slot with
// its own dataset export needs no change here.
export const MULTI_CLASS = 'yolo';

export function createTrainDatasetController() {
  /**
   * The one registered slot that declares a dataset export, read off
   * `registeredSlots` rather than importing a specific profile, so
   * `/train` follows the same "register a new domain in exactly one
   * file" rule every other slot-aware route follows (see
   * `registeredSlots.ts`'s header). `undefined` when no registered slot
   * declares one, in which case the panel never renders at all.
   */
  const datasetExportSpec = registeredSlots
    .map(datasetExportForSlot)
    .find((s) => s !== undefined);

  // Capability discovery is a cached, never-rejecting one-shot
  // (strategiesStore.init() is idempotent; a failed load leaves
  // EMPTY_METHODS), so calling it from an $effect is
  // the same pattern /clusters uses for the embedding-plot toggle.
  $effect(() => {
    void strategiesStore.init();
  });

  /**
   * Whether the backend advertises this export kind on
   * `GET {API_PREFIX}/methods`'s `export` axis. Absent ⇒ the panel and
   * the dataset-kind toggle that selects it are not rendered at all —
   * absent, not disabled — and `refreshSingleClassExportStatus()` is never
   * called, so
   * a deployment whose backend does not serve this export produces zero 404s
   * on this route. Never probe the export endpoint to find out; see
   * `isDatasetExportAvailable`.
   */
  const datasetExportAvailable = $derived(
    datasetExportSpec != null &&
      isDatasetExportAvailable(
        strategiesStore.methods.dataset_exports,
        datasetExportSpec.kind,
      ),
  );

  let datasetKind = $state<string>(MULTI_CLASS);
  let multiClassExportDir = $state<string>('');
  // Full GET {API_PREFIX}/export/status response for the current
  // multi-class export — `class_split_counts`/`split_counts`/
  // `image_count`/`class_count` drive the dataset card's "current
  // export" numbers below, in place of the classesStore-wide validated
  // total (which double-counted holdout crops). It describes only the
  // current export, so a picked past version falls back to the labelled
  // global-pool numbers — see the card markup.
  let multiClassExportState = $state<ExportStatus | null>(null);
  let singleClassExportDir = $state<string>('');
  // All materialized dataset versions on disk (both kinds), newest first.
  let datasets = $state<ExportDataset[]>([]);
  // Explicit operator pick. Empty => fall back to the `current` symlink for the
  // selected kind, so the default behaviour (train the latest export) is
  // unchanged. Picking any past export lets us reuse the exact same data when
  // upsizing nano -> small, etc.
  let selectedExportDir = $state<string>('');
  // /export/status describes only the latest export, so its counts apply
  // when nothing is picked or the pick is that same directory.
  const showsCurrentExport = $derived(
    !selectedExportDir ||
      selectedExportDir === multiClassExportState?.export_dir ||
      selectedExportDir === multiClassExportState?.path,
  );
  // Multi-class rows are `kind: 'yolo'`; a slot's rows are `single_class`
  // rows written under its own `profile_name`.
  const kindDatasets = $derived(
    datasets.filter((d) =>
      datasetExportSpec && datasetKind === datasetExportSpec.datasetKind
        ? d.kind === datasetExportSpec.kind &&
          d.profile_name === datasetExportSpec.profileName
        : d.kind === MULTI_CLASS,
    ),
  );
  const datasetExportDir = $derived(
    selectedExportDir ||
      (datasetKind === datasetExportSpec?.datasetKind
        ? singleClassExportDir
        : multiClassExportDir),
  );
  let datasetMessage = $state<string | null>(null);
  let refreshing = $state<boolean>(false);

  function selectDatasetKind(kind: string): void {
    datasetKind = kind;
    selectedExportDir = ''; // reset to the current export of the new kind
  }

  // `/methods` resolves after mount, so the single-class toggle can
  // disappear while its kind is selected. Fall back to the always-present
  // multi-class dataset rather than leaving the picker pointed at a kind with no UI
  // behind it — same "force the stale selection off" pattern
  // /clusters uses for the embedding-plot toggle.
  $effect(() => {
    if (!datasetExportAvailable && datasetKind !== MULTI_CLASS) {
      selectDatasetKind(MULTI_CLASS);
    }
  });

  async function refreshDataset(): Promise<void> {
    refreshing = true;
    datasetMessage = null;
    try {
      const [e, ds] = await Promise.all([exportStatus(), listDatasets()]);
      multiClassExportDir = e.export_dir ?? '';
      multiClassExportState = e;
      datasets = ds.datasets ?? [];
      if (!multiClassExportDir) {
        datasetMessage =
          'No frozen export available — run /export first to produce a dataset.';
      }
    } catch (e) {
      datasetMessage = `Export status fetch failed: ${apiErrorText(e)}`;
    } finally {
      refreshing = false;
    }
  }

  return {
    datasetExportSpec,
    get datasetExportAvailable() {
      return datasetExportAvailable;
    },
    get datasetKind() {
      return datasetKind;
    },
    get multiClassExportState() {
      return multiClassExportState;
    },
    get singleClassExportDir() {
      return singleClassExportDir;
    },
    set singleClassExportDir(v: string) {
      singleClassExportDir = v;
    },
    get selectedExportDir() {
      return selectedExportDir;
    },
    set selectedExportDir(v: string) {
      selectedExportDir = v;
    },
    get showsCurrentExport() {
      return showsCurrentExport;
    },
    get kindDatasets() {
      return kindDatasets;
    },
    get datasetExportDir() {
      return datasetExportDir;
    },
    get datasetMessage() {
      return datasetMessage;
    },
    get refreshing() {
      return refreshing;
    },
    selectDatasetKind,
    refreshDataset,
  };
}

export type TrainDatasetController = ReturnType<typeof createTrainDatasetController>;
