/**
 * /train's training-cohort picker: per-class cohort definitions and
 * counts (loaded lazily as each class group scrolls into view), the
 * selected cohort's preview grid, and the click-through to /review.
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
import { getReviewQueue } from '$lib/api_review';
import { getTrainingCohorts, type ServedTrainingCohort } from '$lib/api_train';
import { projectHref } from '$lib/projectPaths';
import { classesStore } from '$stores/classes.svelte';
import { slotRegistry } from '$lib/annotations/registeredSlots';
import type { SlotSpec } from '$lib/annotations/types';
import {
  CORE_COHORTS,
  cohortsForClass,
  cohortEndpointKind,
  type CohortSpec,
} from '$lib/annotations/cohorts';
import { splitCohortGroups } from '$lib/trainCohortGroups';
import type { Crop } from '$lib/types';
import type { ReviewItem } from '$lib/types_review';

interface CohortGroup {
  classId: number;
  className: string;
  cohorts: CohortSpec[];
}

type CohortPreviewItem = RegionBrowseItem | Crop | ReviewItem;

export function createTrainCohortsController() {
  // -- Training-cohort picker ---------------------------------------------
  //
  // Cohort definitions now come from the backend's own `GET
  // {API_PREFIX}/training_cohorts?class_id=` — `id`/`label`/`description`/
  // `endpoint`/`params`/`row_kind` are served verbatim, already resolved
  // for the requested class (no client `{classId}` template compilation
  // for these). This replaces the old client-side `cohortsForClass()`
  // (CORE_COHORTS + a slot's own declared modes) as the
  // primary source: the backend serves the same 4 core + N region
  // cohorts today, so nothing the operator sees changes, but a
  // threshold like `low_confidence`'s `classifier_conf_lt` now comes
  // from the server rather than a client constant.
  //
  // `cohortsForClass()`/`CORE_COHORTS`/`cohorts.ts`'s tier-2 mechanism
  // is NOT deleted — a deployment can still register a brand-new slot
  // via `annotation-profiles.json` (parseSlotConfig.ts) that the
  // backend has no region profile for, and that slot's own hand-declared
  // `capabilities.trainingCohorts.cohorts` (checked at `loadGroupCohorts`
  // below) still surfaces here as a fallback for any cohort id the
  // server didn't already send — the server always wins on an id
  // collision. See secondSlotIntegration.test.ts / cohorts.test.ts for
  // the tier-2 mechanism's own coverage, independent of this page.
  //
  // Scope note: cohorts are computed for every non-deprecated class
  // (the "default to all classes" recommendation, §9.11) rather than
  // synced to TrainForm's own class-subset selection — that tighter
  // coupling (lifting `selectedClasses` into this page) is deliberately
  // NOT done here; TrainForm's selection continues to drive the actual
  // training run unchanged.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- rebuilt whole on every $derived recompute and never mutated; reactivity comes from the $derived
  const classesById = $derived(new Map(classesStore.classes.map((c) => [c.id, c.name])));
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- constant lookup set built once from CORE_COHORTS, never mutated
  const coreCohortIds = new Set(CORE_COHORTS.map((c) => c.id));

  // Cohort *definitions* load lazily per class (see lazyLoadGroupCounts
  // below) — `classCohorts` starts empty for every class and is filled
  // in the same intersection-observer callback that used to load only
  // counts, so mounting this page never fires an eager N-classes
  // request storm.
  let classCohorts = $state<Record<number, CohortSpec[]>>({});

  const cohortGroups = $derived.by<CohortGroup[]>(() =>
    classesStore.classes
      .filter((c) => !c.deprecated)
      .map((c) => ({
        classId: c.id,
        className: c.name,
        cohorts: classCohorts[c.id] ?? [],
      })),
  );

  /** Converts one served cohort into the local `CohortSpec` shape the
   *  preview grid already understands — `params` are used verbatim
   *  (already resolved for this class by the server), never
   *  re-templated. `row_kind: 'region'` renders via `SlotCard`
   *  ('slot') and jumps to the owning slot's review queue on click;
   *  `'crop'` renders via `CropCard` and jumps to the All review tab. */
  function fromServedCohort(c: ServedTrainingCohort): CohortSpec {
    return {
      id: c.id,
      label: c.label,
      description: c.description,
      query: {
        kind: 'endpoint',
        path: c.endpoint,
        params: c.params as Record<string, string | number | boolean>,
      },
      rowKind: c.row_kind === 'region' ? 'slot' : 'crop',
      reviewTarget: c.row_kind === 'region' ? 'slotQueue' : 'all',
    };
  }

  /** Server cohorts first; a slot's own tier-2-declared cohort fills in
   *  only an id the server didn't already send — see the header comment
   *  above. `predicateCohortsAvailable=false`: no backend support for a
   *  tier-2 predicate cohort exists yet, so `cohortsForClass` here can
   *  only ever contribute CORE_COHORTS (filtered out below, since the
   *  server already sent its own core cohorts) or a slot's declared
   *  cohorts. */
  async function loadGroupCohorts(group: CohortGroup): Promise<CohortSpec[]> {
    const served = await getTrainingCohorts(group.classId).catch(
      () => ({ cohorts: [] }) as { cohorts: ServedTrainingCohort[] },
    );
    const servedCohorts = served.cohorts.map(fromServedCohort);
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local, built and consumed synchronously within this call, never stored in reactive state
    const servedIds = new Set(servedCohorts.map((c) => c.id));
    const declaredFallback = cohortsForClass(
      group.classId,
      group.className,
      slotRegistry,
      classesById,
      false,
    ).filter((c) => !coreCohortIds.has(c.id) && !servedIds.has(c.id));
    return [...servedCohorts, ...declaredFallback];
  }

  let cohortCounts = $state<Record<string, number | null>>({});
  let selectedCohortKey = $state<string | null>(null);
  let cohortPreview = $state<CohortPreviewItem[]>([]);
  let cohortPreviewLoading = $state<boolean>(false);
  let cohortPreviewError = $state<string | null>(null);

  /** Stable key across (class, cohort) pairs — a cohort id alone isn't
   *  unique once multiple classes are shown (e.g. every class has a
   *  'validated' core cohort). */
  function cohortKey(classId: number, cohort: CohortSpec): string {
    return `${classId}:${cohort.id}`;
  }

  // m-train-cohorts (2026-09-24 interactive pass): with ~85 classes × up
  // to 8 cohorts, most classes' chips are all 0 — a wall of zeros
  // (long-tail class names, each with a handful of examples) dominating
  // the section above the actually-useful Past runs table. Collapsing logic lives in
  // `$lib/trainCohortGroups.ts` (pure, unit-tested) — a class only
  // collapses once every one of its cohorts SERVED a count of exactly 0.
  const cohortGroupSplit = $derived(splitCohortGroups(cohortGroups, cohortCounts));
  const visibleCohortGroups = $derived(cohortGroupSplit.visible);
  const zeroCandidateGroups = $derived(cohortGroupSplit.zero);
  const cohortGroupsPending = $derived(cohortGroupSplit.pending);

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
    cohort: CohortSpec,
    pageSize: number,
    className: string,
  ): Promise<{ total: number; items: CohortPreviewItem[] }> {
    if (cohort.query.kind !== 'endpoint') return { total: 0, items: [] };
    const { path, params } = cohort.query;
    const classId =
      typeof params.class_id === 'number'
        ? params.class_id
        : typeof params.class_id === 'string'
          ? Number(params.class_id)
          : undefined;
    const kind = cohortEndpointKind(path);

    if (kind === 'training_candidates') {
      const mode = params.mode as TrainingCohortMode;
      const res = await getTrainingCandidates(mode, {
        page_size: pageSize,
        class_id: classId,
      });
      return { total: res.total, items: res.items };
    }
    if (kind === 'crops') {
      // `GET /crops` and `/review/{tab}` take the class by NAME (v0.4.0);
      // only `/regions/training_candidates` still takes `class_id`.
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
        limit: pageSize,
      });
      return { total: res.total, items: res.items };
    }
    if (kind === 'model_disagreements') {
      const res = await getReviewQueue('model_disagreements', 1, pageSize, {
        class_name: [className],
      });
      return { total: res.total, items: res.items };
    }
    return { total: 0, items: [] };
  }

  // Lazy per-group cohorts + counts (§9.10's mitigation for the
  // N-classes × M-cohorts request storm): a group's cohort definitions
  // AND counts load once, the first time its header scrolls into the
  // viewport, instead of every group firing requests on mount.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- one-shot load guard, never read reactively by a template/derived
  const groupCountsLoaded = new Set<number>();
  function lazyLoadGroupCounts(node: HTMLElement, group: CohortGroup) {
    const observer = new IntersectionObserver(
      (entries) => {
        if (
          entries.some((e) => e.isIntersecting) &&
          !groupCountsLoaded.has(group.classId)
        ) {
          groupCountsLoaded.add(group.classId);
          void loadGroupCounts(group);
        }
      },
      { rootMargin: '200px' },
    );
    observer.observe(node);
    return { destroy: () => observer.disconnect() };
  }

  async function loadGroupCounts(group: CohortGroup): Promise<void> {
    const cohorts = await loadGroupCohorts(group);
    classCohorts = { ...classCohorts, [group.classId]: cohorts };
    const results = await Promise.allSettled(
      cohorts.map((c) => runCohortQuery(c, 1, group.className)),
    );
    const next: Record<string, number | null> = { ...cohortCounts };
    cohorts.forEach((cohort, i) => {
      const r = results[i];
      next[cohortKey(group.classId, cohort)] =
        r?.status === 'fulfilled' ? r.value.total : null;
    });
    cohortCounts = next;
  }

  async function refreshCohortCounts(): Promise<void> {
    const all = cohortGroups.flatMap((g) =>
      g.cohorts.map((c) => ({ group: g, cohort: c })),
    );
    const results = await Promise.allSettled(
      all.map(({ group, cohort }) => runCohortQuery(cohort, 1, group.className)),
    );
    const next: Record<string, number | null> = { ...cohortCounts };
    all.forEach(({ group, cohort }, i) => {
      const r = results[i];
      next[cohortKey(group.classId, cohort)] =
        r?.status === 'fulfilled' ? r.value.total : null;
    });
    cohortCounts = next;
  }

  async function loadCohortPreview(
    group: CohortGroup,
    cohort: CohortSpec,
  ): Promise<void> {
    selectedCohortKey = cohortKey(group.classId, cohort);
    cohortPreviewLoading = true;
    cohortPreviewError = null;
    cohortPreview = [];
    try {
      const res = await runCohortQuery(cohort, 24, group.className);
      cohortPreview = res.items;
    } catch (e) {
      cohortPreviewError = apiErrorText(e);
    } finally {
      cohortPreviewLoading = false;
    }
  }

  /** The slot a region cohort's rows render with: the slot bound to this
   *  class group, else the only registered queue slot with a sub-box.
   *  `undefined` renders no region cards rather than guessing a slot. */
  function regionSlotForGroup(group: CohortGroup): SlotSpec | undefined {
    const bound = slotRegistry.forClass(group.classId, classesById)[0];
    if (bound) return bound;
    const regionSlots = slotRegistry.queues.filter((s) => s.capabilities.subBox != null);
    return regionSlots.length === 1 ? regionSlots[0] : undefined;
  }

  /** Click target from `reviewTarget` (§9.3 step 7): the owning slot's
   *  queue tab, else the All tab. */
  function openCohortItem(
    group: CohortGroup,
    cohort: CohortSpec,
    item: CohortPreviewItem,
  ): void {
    const cropId = 'crop_id' in item ? item.crop_id : item.id;
    if (cohort.reviewTarget === 'slotQueue') {
      const slot = slotRegistry.forClass(group.classId, classesById)[0];
      const urlId = slot?.capabilities.queue?.urlId ?? 'all';
      void goto(
        resolve(
          projectHref(`/review?tab=${urlId}&crop_id=${encodeURIComponent(cropId)}`),
        ),
      );
      return;
    }
    void goto(
      resolve(projectHref(`/review?tab=all&crop_id=${encodeURIComponent(cropId)}`)),
    );
  }

  return {
    get cohortCounts() {
      return cohortCounts;
    },
    get selectedCohortKey() {
      return selectedCohortKey;
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
    get visibleCohortGroups() {
      return visibleCohortGroups;
    },
    get zeroCandidateGroups() {
      return zeroCandidateGroups;
    },
    get cohortGroupsPending() {
      return cohortGroupsPending;
    },
    cohortKey,
    lazyLoadGroupCounts,
    refreshCohortCounts,
    loadCohortPreview,
    regionSlotForGroup,
    openCohortItem,
  };
}

export type TrainCohortsController = ReturnType<typeof createTrainCohortsController>;
