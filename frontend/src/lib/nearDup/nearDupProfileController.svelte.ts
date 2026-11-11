/**
 * The project's `near_dup_profile`: the thresholds of both levels. The draft
 * is the active revision's body; a save writes the next revision and activates
 * it, and the server then marks both levels `rebuild_required` (nothing
 * regroups silently). The seeded defaults are revision 1 of `default`, read
 * from the API. Validation is the server's: its refusal reasons are shown as
 * sent.
 */
import { ApiError, apiErrorText } from '$lib/api_errors';
import {
  activateNearDupProfile,
  getActiveNearDupProfile,
  getNearDupProfile,
  saveNearDupProfile,
} from '$lib/api_nearDup';
import type { NearDupProfileBody } from '$lib/types_nearDup';

const SEEDED_NAME = 'default';
const SEEDED_REVISION = 1;

function clone(body: NearDupProfileBody): NearDupProfileBody {
  return JSON.parse(JSON.stringify(body)) as NearDupProfileBody;
}

/** The served `detail.reasons` of a 422 `validation_failed`, when there are any. */
function refusalReasons(e: unknown): string[] {
  if (!(e instanceof ApiError)) return [];
  const detail = (e.body as { detail?: { reasons?: unknown } } | null)?.detail;
  const reasons = detail && typeof detail === 'object' ? detail.reasons : null;
  return Array.isArray(reasons)
    ? reasons.filter((r): r is string => typeof r === 'string')
    : [];
}

export function createNearDupProfileController(onSaved: () => void) {
  let name = $state<string | null>(null);
  let revision = $state<number | null>(null);
  let saved = $state<NearDupProfileBody | null>(null);
  let draft = $state<NearDupProfileBody | null>(null);
  let seeded = $state<NearDupProfileBody | null>(null);
  let loading = $state(false);
  let loadError = $state<string | null>(null);
  let saving = $state(false);
  let saveError = $state<string | null>(null);
  let reasons = $state<string[]>([]);
  let notice = $state<string | null>(null);
  const abort = new AbortController();

  async function load(): Promise<void> {
    loading = true;
    loadError = null;
    try {
      const active = await getActiveNearDupProfile(abort.signal);
      if (active.state !== 'active' || !active.name || active.revision == null) {
        // Fail closed: with no active profile nothing groups, so there is nothing to edit.
        name = null;
        revision = null;
        saved = null;
        draft = null;
        loadError = 'No near-duplicate profile is active for this project.';
        return;
      }
      const doc = await getNearDupProfile(active.name, active.revision, abort.signal);
      name = doc.name;
      revision = doc.revision;
      saved = clone(doc.body);
      draft = clone(doc.body);
      try {
        seeded = (await getNearDupProfile(SEEDED_NAME, SEEDED_REVISION, abort.signal))
          .body;
      } catch {
        // A project older than the seed has no revision 1; the defaults column is then absent.
        seeded = null;
      }
    } catch (e) {
      if (!abort.signal.aborted) loadError = apiErrorText(e);
    } finally {
      loading = false;
    }
  }

  async function save(): Promise<void> {
    if (saving || !draft || !name || revision == null) return;
    saving = true;
    saveError = null;
    reasons = [];
    notice = null;
    try {
      const doc = await saveNearDupProfile(name, draft, revision, abort.signal);
      await activateNearDupProfile(
        doc.name,
        doc.revision,
        { name, revision },
        abort.signal,
      );
      name = doc.name;
      revision = doc.revision;
      saved = clone(doc.body);
      draft = clone(doc.body);
      notice = `Saved as revision ${doc.revision} and activated. Both levels need a rebuild before the groups reflect it.`;
      onSaved();
    } catch (e) {
      if (abort.signal.aborted) return;
      saveError = apiErrorText(e);
      reasons = refusalReasons(e);
    } finally {
      saving = false;
    }
  }

  /** Put the seeded defaults into the draft; saving is still a separate step. */
  function useSeeded(): void {
    if (seeded) draft = clone(seeded);
  }

  function reset(): void {
    if (saved) draft = clone(saved);
    saveError = null;
    reasons = [];
  }

  function dispose(): void {
    abort.abort();
  }

  return {
    get name() {
      return name;
    },
    get revision() {
      return revision;
    },
    get draft() {
      return draft;
    },
    get seeded() {
      return seeded;
    },
    get dirty() {
      return draft !== null && JSON.stringify(draft) !== JSON.stringify(saved);
    },
    get loading() {
      return loading;
    },
    get loadError() {
      return loadError;
    },
    get saving() {
      return saving;
    },
    get saveError() {
      return saveError;
    },
    get reasons() {
      return reasons;
    },
    get notice() {
      return notice;
    },
    load,
    save,
    useSeeded,
    reset,
    dispose,
  };
}

export type NearDupProfileController = ReturnType<typeof createNearDupProfileController>;
