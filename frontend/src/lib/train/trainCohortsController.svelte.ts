/**
 * /train's training-cohort matrix: the served classes x cohorts counts (one request), the table's
 * filter and "hide empty" switch, and the preview of one cell (class, cohort) the Drawer shows,
 * with the click-through to the Workspace Review.
 */
import { goto } from '$app/navigation';
import { resolve } from '$app/paths';
import { getCrops } from '$lib/api_cropRead';
import { apiErrorText } from '$lib/api_errors';
import {
  getTrainingCandidates,
  type RegionBrowseItem,
  type TrainingCohortMode,
} from '$lib/api_regions';
import {
  getTrainingCohortSummary,
  type CohortColumn,
  type CohortRow,
  type TrainingCohortSummary,
} from '$lib/api_train';
import { projectHref } from '$lib/projectPaths';
import { reviewSearch } from '$lib/workspace/workspaceUrl';
import { classesStore } from '$stores/classes.svelte';
import { slotRegistry } from '$lib/annotations/registeredSlots';
import type { SlotSpec } from '$lib/annotations/types';
import { cohortEndpointKind } from '$lib/annotations/cohorts';
import type { Crop } from '$lib/types';
import type { ReviewItem } from '$lib/types_review';

export type CohortPreviewItem = RegionBrowseItem | Crop | ReviewItem;

/** One cell of the matrix. */
export interface CohortCell {
  row: CohortRow;
  column: CohortColumn;
}

/** The "detector proposals not decided" cohort: items with no class yet, counted (and previewed)
 *  by the detector's own class name (`detector_class_name`), not by a class. */
export const UNDECIDED_ID = 'detector_proposals';

export function createTrainCohortsController() {
  let summary = $state<TrainingCohortSummary | null>(null);
  let loading = $state(false);
  let failure = $state<unknown>(null);
  let filter = $state('');
  let hideEmpty = $state(true);

  async function load(signal?: AbortSignal): Promise<void> {
    loading = true;
    failure = null;
    try {
      summary = await getTrainingCohortSummary(signal);
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      summary = null;
      failure = e;
    } finally {
      loading = false;
    }
  }

  const rows = $derived.by<CohortRow[]>(() => {
    if (!summary) return [];
    const terms = filter.toLowerCase().split(/\s+/).filter(Boolean);
    return summary.rows.filter(
      (r) =>
        terms.every((t) => r.class_name.toLowerCase().includes(t)) &&
        (!hideEmpty || Object.values(r.counts).some((n) => n > 0)),
    );
  });
  const hiddenEmpty = $derived(
    summary && hideEmpty
      ? summary.rows.filter((r) => !Object.values(r.counts).some((n) => n > 0)).length
      : 0,
  );

  let selected = $state<CohortCell | null>(null);
  let cohortPreview = $state<CohortPreviewItem[]>([]);
  let cohortPreviewLoading = $state<boolean>(false);
  let cohortPreviewError = $state<string | null>(null);

  /** A cell opens a preview when its class is in the registry and its cohort is a class filter. */
  function previewable(row: CohortRow, column: CohortColumn): boolean {
    if ((row.counts[column.id] ?? 0) === 0) return false;
    // The undecided cohort is counted by the detector's own label, so a detector-only row has one.
    return column.id === UNDECIDED_ID || row.class_id != null;
  }

  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- rebuilt whole on every $derived recompute and never mutated; reactivity comes from the $derived
  const classesById = $derived(new Map(classesStore.classes.map((c) => [c.id, c.name])));

  /** Dispatches a cohort's `endpoint`/`params` (served verbatim by
   *  `{API_PREFIX}/training_cohorts`, or a tier-2 declared fallback
   *  compiled the same way) to the one existing api.ts function that
   *  already answers that endpoint shape — keyed structurally by
   *  `cohortEndpointKind()` rather than a path string-equality check (see cohorts.ts's doc comment
   *  on `cohortEndpointKind` for why the old check silently broke on a
   *  base-path rename). Anything else (a future slot's endpoint cohort
   *  naming a path none of these three recognize) fails closed to an
   *  empty/null result rather than guessing at an endpoint shape. */
  async function runCohortQuery(
    column: CohortColumn,
    row: CohortRow,
    pageSize: number,
  ): Promise<{ total: number; items: CohortPreviewItem[] }> {
    const { endpoint: path, params } = column;
    const classId = row.class_id ?? undefined;
    const className = row.class_name;
    const kind = cohortEndpointKind(path);

    if (column.id === UNDECIDED_ID) {
      const res = await getCrops({
        review_tab: typeof params.review_tab === 'string' ? params.review_tab : undefined,
        detector_class_name: [className],
        limit: pageSize,
      });
      return { total: res.total, items: res.items };
    }

    if (kind === 'training_candidates') {
      const mode = params.mode as TrainingCohortMode;
      const res = await getTrainingCandidates(mode, {
        limit: pageSize,
        class_id: classId,
      });
      return { total: res.total, items: res.items };
    }
    if (kind === 'crops') {
      // `GET /crops` takes the class by NAME (v0.4.0); only
      // `/regions/training_candidates` still takes `class_id`.
      const res = await getCrops({
        class_name: [className],
        label_validated:
          typeof params.label_validated === 'boolean'
            ? params.label_validated
            : undefined,
        classifier_conf_lt:
          typeof params.classifier_conf_lt === 'number'
            ? params.classifier_conf_lt
            : undefined,
        review_tab: typeof params.review_tab === 'string' ? params.review_tab : undefined,
        limit: pageSize,
      });
      return { total: res.total, items: res.items };
    }
    return { total: 0, items: [] };
  }

  async function openPreview(row: CohortRow, column: CohortColumn): Promise<void> {
    selected = { row, column };
    cohortPreviewLoading = true;
    cohortPreviewError = null;
    cohortPreview = [];
    try {
      cohortPreview = (await runCohortQuery(column, row, 24)).items;
    } catch (e) {
      cohortPreviewError = apiErrorText(e);
    } finally {
      cohortPreviewLoading = false;
    }
  }

  function closePreview(): void {
    selected = null;
    cohortPreview = [];
    cohortPreviewError = null;
  }

  /** The slot a region cohort's rows render with: the slot bound to this
   *  class group, else the only registered queue slot with a sub-box.
   *  `undefined` renders no region cards rather than guessing a slot. */
  function regionSlotForGroup(row: CohortRow): SlotSpec | undefined {
    const bound =
      row.class_id == null
        ? undefined
        : slotRegistry.forClass(row.class_id, classesById)[0];
    if (bound) return bound;
    const regionSlots = slotRegistry.queues.filter((s) => s.capabilities.subBox != null);
    return regionSlots.length === 1 ? regionSlots[0] : undefined;
  }

  /** Click target from `reviewTarget` (§9.3 step 7): Review at the item, on the owning slot's
   *  queue, else on the Detections order. */
  function openCohortItem(
    row: CohortRow,
    column: CohortColumn,
    item: CohortPreviewItem,
  ): void {
    const cropId = 'crop_id' in item ? item.crop_id : item.id;
    if (column.row_kind === 'region') {
      const slot =
        row.class_id == null
          ? undefined
          : slotRegistry.forClass(row.class_id, classesById)[0];
      const queue = slot?.capabilities.queue?.endpointId ?? null;
      void goto(resolve(projectHref(`/workspace${reviewSearch({ queue, cropId })}`)));
      return;
    }
    void goto(resolve(projectHref(`/workspace${reviewSearch({ cropId })}`)));
  }

  return {
    get summary() {
      return summary;
    },
    get loading() {
      return loading;
    },
    get failure() {
      return failure;
    },
    get filter() {
      return filter;
    },
    set filter(v: string) {
      filter = v;
    },
    get hideEmpty() {
      return hideEmpty;
    },
    set hideEmpty(v: boolean) {
      hideEmpty = v;
    },
    get rows() {
      return rows;
    },
    get hiddenEmpty() {
      return hiddenEmpty;
    },
    get selected() {
      return selected;
    },
    get cohortPreview() {
      return cohortPreview;
    },
    get cohortPreviewLoading() {
      return cohortPreviewLoading;
    },
    get cohortPreviewError() {
      return cohortPreviewError;
    },
    load,
    previewable,
    openPreview,
    closePreview,
    regionSlotForGroup,
    openCohortItem,
  };
}

export type TrainCohortsController = ReturnType<typeof createTrainCohortsController>;
