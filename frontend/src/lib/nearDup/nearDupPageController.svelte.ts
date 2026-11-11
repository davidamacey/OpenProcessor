/**
 * The near-duplicate page: the per-level status and run history, a run start
 * that is followed to its end, and the paged groups of the chosen level and
 * purpose. Every figure is served; this module only holds the view state.
 */
import { apiErrorText } from '$lib/api_errors';
import {
  getNearDupStatus,
  listNearDupGroups,
  listNearDupRuns,
  previewItemsForImages,
  startNearDupRun,
} from '$lib/api_nearDup';
import { followJob } from '$lib/jobs/followJob';
import type {
  NearDupGroupSummary,
  NearDupLevel,
  NearDupPurpose,
  NearDupRunMode,
  NearDupRunView,
  NearDupStatus,
} from '$lib/types_nearDup';

const RUN_HISTORY = 10;
const GROUP_PAGE = 24;

export function createNearDupPageController() {
  let status = $state<NearDupStatus | null>(null);
  let runs = $state<NearDupRunView[]>([]);
  let loadError = $state<string | null>(null);
  let loading = $state(false);

  let level = $state<NearDupLevel>('image');
  let purpose = $state<NearDupPurpose>('dedup');
  let groups = $state<NearDupGroupSummary[]>([]);
  let nextCursor = $state<string | null>(null);
  let groupsLoaded = $state(false);
  let groupsLoading = $state(false);
  let groupsError = $state<string | null>(null);
  /** Image id -> an item standing on it, for thumbnails at the image level. */
  let previews = $state<Record<string, string>>({});

  let busyLevel = $state<NearDupLevel | null>(null);
  let runMessage = $state<string | null>(null);
  let runError = $state<string | null>(null);

  let abort: AbortController | null = null;
  let groupsToken = 0;

  async function loadOverview(): Promise<void> {
    loading = true;
    loadError = null;
    abort ??= new AbortController();
    const signal = abort.signal;
    try {
      const [s, r] = await Promise.all([
        getNearDupStatus(signal),
        listNearDupRuns(undefined, RUN_HISTORY, signal),
      ]);
      status = s;
      runs = r.items;
    } catch (e) {
      if (!signal.aborted) loadError = apiErrorText(e);
    } finally {
      loading = false;
    }
  }

  async function addPreviews(page: NearDupGroupSummary[], forLevel: NearDupLevel) {
    if (forLevel !== 'image' || !abort) return;
    const ids = page.map((g) => g.representative_id).filter((id) => !(id in previews));
    if (ids.length === 0) return;
    try {
      previews = { ...previews, ...(await previewItemsForImages(ids, abort.signal)) };
    } catch {
      // A missing thumbnail is cosmetic; the group row still renders its ids and figures.
    }
  }

  async function loadGroups(append = false): Promise<void> {
    if (append && !nextCursor) return;
    const token = ++groupsToken;
    abort ??= new AbortController();
    groupsLoading = true;
    groupsError = null;
    const forLevel = level;
    try {
      const page = await listNearDupGroups(
        {
          level: forLevel,
          purpose,
          limit: GROUP_PAGE,
          ...(append && nextCursor ? { cursor: nextCursor } : {}),
        },
        abort.signal,
      );
      if (token !== groupsToken) return;
      groups = append ? [...groups, ...page.items] : page.items;
      nextCursor = page.next_cursor;
      groupsLoaded = true;
      await addPreviews(page.items, forLevel);
    } catch (e) {
      if (token === groupsToken && !abort.signal.aborted) groupsError = apiErrorText(e);
    } finally {
      if (token === groupsToken) groupsLoading = false;
    }
  }

  function showGroups(
    nextLevel: NearDupLevel,
    nextPurpose: NearDupPurpose,
  ): Promise<void> {
    level = nextLevel;
    purpose = nextPurpose;
    groups = [];
    nextCursor = null;
    groupsLoaded = false;
    return loadGroups();
  }

  /** Start a run of one level and follow its job; the served refusal (409) is shown as sent. */
  async function startRun(runLevel: NearDupLevel, mode: NearDupRunMode): Promise<void> {
    if (busyLevel) return;
    busyLevel = runLevel;
    runError = null;
    runMessage = null;
    abort ??= new AbortController();
    const signal = abort.signal;
    try {
      const started = await startNearDupRun({ level: runLevel, mode }, signal);
      const done = await followJob(started.job_id, {
        signal,
        onUpdate: (job) => {
          runMessage = `${runLevel} level: ${job.status}`;
        },
      });
      if (done.status === 'completed') {
        runMessage = `${runLevel} level: groups rebuilt.`;
      } else {
        runMessage = null;
        runError = `${runLevel} level run ${done.status}${done.error_message ? `: ${done.error_message}` : ''}`;
      }
    } catch (e) {
      runMessage = null;
      if (!signal.aborted) runError = apiErrorText(e);
    } finally {
      busyLevel = null;
    }
    if (!signal.aborted) {
      await loadOverview();
      await showGroups(level, purpose);
    }
  }

  function dispose(): void {
    abort?.abort();
    abort = null;
  }

  return {
    get status() {
      return status;
    },
    get runs() {
      return runs;
    },
    get loadError() {
      return loadError;
    },
    get loading() {
      return loading;
    },
    get level() {
      return level;
    },
    get purpose() {
      return purpose;
    },
    get groups() {
      return groups;
    },
    get hasMore() {
      return nextCursor !== null;
    },
    get groupsLoaded() {
      return groupsLoaded;
    },
    get groupsLoading() {
      return groupsLoading;
    },
    get groupsError() {
      return groupsError;
    },
    get previews() {
      return previews;
    },
    get busyLevel() {
      return busyLevel;
    },
    get runMessage() {
      return runMessage;
    },
    get runError() {
      return runError;
    },
    loadOverview,
    loadGroups,
    showGroups,
    startRun,
    dispose,
  };
}

export type NearDupPageController = ReturnType<typeof createNearDupPageController>;
