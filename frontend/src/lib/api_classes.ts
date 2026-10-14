/**
 * The class registry: read, add, rename, deprecate, merge, sources, and the
 * structured class-route refusals.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import { ApiError } from '$lib/api_errors';
import type {
  ClassesResponse,
  ClassMergeDryRun,
  ClassThresholds,
  RegistryClassCreate,
  RegistryClassMerge,
  RegistryClassUpdate,
} from '$lib/types';

/** `PUT /classes/{id}` 422 `hotkey_reserved` (plan §4.5). */
export interface HotkeyReservedDetail {
  message: string;
  actions: Array<{ action_id: string; context: string; label: string }>;
}

export function hotkeyReservedDetail(e: unknown): HotkeyReservedDetail | null {
  if (!(e instanceof ApiError) || e.status !== 422) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'hotkey_reserved') return null;
  return {
    message: typeof d.message === 'string' ? d.message : 'That letter is reserved.',
    actions: Array.isArray(d.actions)
      ? (d.actions as HotkeyReservedDetail['actions'])
      : [],
  };
}

/** `PUT /classes/{id}` 409 `hotkey_taken` (plan §4.5). */
export interface HotkeyTakenDetail {
  message: string;
  class_id: number;
  class_name: string;
}

export function hotkeyTakenDetail(e: unknown): HotkeyTakenDetail | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'hotkey_taken') return null;
  if (typeof d.class_id !== 'number' || typeof d.class_name !== 'string') return null;
  return {
    message:
      typeof d.message === 'string' ? d.message : `Already bound to ${d.class_name}.`,
    class_id: d.class_id,
    class_name: d.class_name,
  };
}

export async function getClasses(signal?: AbortSignal): Promise<ClassesResponse> {
  // Map the served `ClassEntry` rows to the labeler's RegistryClass shape
  // (`id`/`name`/`count`); `thresholds` and `reserved_hotkeys` pass through
  // verbatim — the server's own adequacy/hotkey rules, never recomputed
  // client-side.
  type RawClass = {
    class_id: number;
    class_name: string;
    group: string;
    sample_count: number;
    validated_count: number;
    cluster_size: number;
    deprecated: boolean;
    added_at: string | null;
    hotkey_letter: string | null;
    adequacy: string;
    kind: 'item' | 'region';
    trainable: number;
    trainable_gap: number;
    merged_into: number | null;
  };
  const res = await apiFetch<{
    classes: RawClass[];
    thresholds: ClassThresholds;
    reserved_hotkeys: string[];
  }>(`${scoped()}/classes`, {}, signal);
  const classes = res.classes.map((c) => ({
    id: c.class_id,
    name: c.class_name,
    group: c.group || null,
    count: c.sample_count,
    validated_count: c.validated_count,
    cluster_size: c.cluster_size,
    added_at: c.added_at ?? '',
    color: null,
    deprecated: c.deprecated,
    hotkey_letter: c.hotkey_letter,
    adequacy: c.adequacy,
    kind: c.kind,
    trainable: c.trainable,
    trainable_gap: c.trainable_gap,
    merged_into: c.merged_into,
  }));
  return { classes, thresholds: res.thresholds, reserved_hotkeys: res.reserved_hotkeys };
}

// -- classes mutators ----------------------------------------------------

export function addClass(
  payload: RegistryClassCreate,
  signal?: AbortSignal,
): Promise<{ class_id: number; class_name: string; group: string }> {
  return apiFetch<{ class_id: number; class_name: string; group: string }>(
    `${scoped()}/classes`,
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

export function renameClass(
  classId: number,
  payload: RegistryClassUpdate,
  signal?: AbortSignal,
): Promise<unknown> {
  return apiFetch<unknown>(
    `${scoped()}/classes/${classId}`,
    { method: 'PUT', body: JSON.stringify(payload) },
    signal,
  );
}

/**
 * The 409 `detail` `POST {API_PREFIX}/classes/{id}/deprecate` produces when
 * the class still has data referencing it (`{error:"class_still_referenced",
 * message, class_id, item_count, confirmed_label_count}`) — merge is the
 * only way to retire a class in that state, so the caller uses this to
 * offer the existing merge flow instead of a raw error toast.
 */
export interface ClassStillReferencedDetail {
  error: 'class_still_referenced';
  message: string;
  class_id: number;
  item_count: number;
  confirmed_label_count: number;
}

export function classStillReferencedDetail(
  e: unknown,
): ClassStillReferencedDetail | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'class_still_referenced') return null;
  if (typeof d.message !== 'string') return null;
  if (typeof d.class_id !== 'number') return null;
  if (typeof d.item_count !== 'number' || typeof d.confirmed_label_count !== 'number') {
    return null;
  }
  return {
    error: 'class_still_referenced',
    message: d.message,
    class_id: d.class_id,
    item_count: d.item_count,
    confirmed_label_count: d.confirmed_label_count,
  };
}

/** `POST /classes/{id}/restore`'s structured 409 for a class that was
 *  merged into another */
export interface ClassMergedDetail {
  error: 'class_merged';
  message: string;
  class_id: number;
  merged_into: { class_id: number; class_name: string };
  hint: string | null;
}

export function classMergedDetail(e: unknown): ClassMergedDetail | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'class_merged' || typeof d.message !== 'string') return null;
  const into = d.merged_into as Record<string, unknown> | null | undefined;
  if (!into || typeof into.class_id !== 'number') return null;
  return {
    error: 'class_merged',
    message: d.message,
    class_id: typeof d.class_id === 'number' ? d.class_id : -1,
    merged_into: {
      class_id: into.class_id,
      class_name:
        typeof into.class_name === 'string' ? into.class_name : String(into.class_id),
    },
    hint: typeof d.hint === 'string' ? d.hint : null,
  };
}

/** The operator-facing text for a merged-class restore refusal: the
 *  merge target by its served name, then the server's own message and
 *  hint verbatim. */
export function classMergedRestoreText(d: ClassMergedDetail): string {
  return [
    `Merged into ${d.merged_into.class_name}; un-merge isn't supported.`,
    d.message,
    d.hint ?? '',
  ]
    .filter((s) => s.length > 0)
    .join(' ');
}

/**
 * Retire a class with no data yet — no merge target needed. Idempotent
 * (calling on an already-deprecated class just returns it) and clears any
 * bound `hotkey_letter` server-side. 404 for an unknown id; 409 (see
 * `classStillReferencedDetail`) while any item/confirmed-label doc still
 * carries this `class_id` — merge instead. The response is the backend's
 * own `RegistryClassEntry` shape (`class_id`/`class_name`/…, not the
 * labeler's `RegistryClass`); callers refetch `classesStore` rather than
 * mapping it.
 */
export function deprecateClass(classId: number, signal?: AbortSignal): Promise<unknown> {
  return apiFetch<unknown>(
    `${scoped()}/classes/${classId}/deprecate`,
    { method: 'POST' },
    signal,
  );
}

/**
 * Undo `deprecateClass`. 404 for an unknown id; 409 with a PLAIN STRING
 * `detail` (not the structured shape above) when a live class already
 * uses this class's name — show `ApiError.detail` verbatim.
 */
export function restoreClass(classId: number, signal?: AbortSignal): Promise<unknown> {
  return apiFetch<unknown>(
    `${scoped()}/classes/${classId}/restore`,
    { method: 'POST' },
    signal,
  );
}

export function mergeClasses(
  payload: RegistryClassMerge,
  signal?: AbortSignal,
): Promise<{
  source_id: number;
  target_id: number;
  deprecated: boolean;
  source_name: string;
  target_name: string;
}> {
  // The backend does not return a relabeled count -- don't claim one.
  return apiFetch<{
    source_id: number;
    target_id: number;
    deprecated: boolean;
    source_name: string;
    target_name: string;
  }>(
    `${scoped()}/classes/merge`,
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

/**
 * `POST {API_PREFIX}/classes/merge?dry_run=true` — reports what a real merge
 * would do (`would_relabel`, `validations_carried_over`, `holdout_blocking`,
 * `blocked`) and writes nothing. The merge dialog calls this before every
 * real merge so the operator sees the blast radius first.
 */
export function previewClassMerge(
  payload: RegistryClassMerge,
  signal?: AbortSignal,
): Promise<ClassMergeDryRun> {
  return apiFetch<ClassMergeDryRun>(
    `${scoped()}/classes/merge${qs({ dry_run: true })}`,
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

/** What kind of writer a `class_source` value names. */
export type ClassSourceRole =
  | 'proposal'
  | 'low_conf'
  | 'model'
  | 'vlm'
  | 'vlm_unmatched'
  | 'vlm_new_class_pending'
  | 'vlm_reclassified'
  | 'cluster'
  | 'human'
  | 'merge'
  | 'label_import'
  | 'open_vocab'
  | (string & {});

/** One `class_source` value this deployment can write. */
export interface ClassSource {
  id: string;
  label: string;
  role: ClassSourceRole;
}

/** `GET {API_PREFIX}/class_sources` — every `class_source` value this
 *  deployment can write, including the ingest detectors' config-derived
 *  ones (`{primary}_proposal`, …). */
export async function getClassSources(signal?: AbortSignal): Promise<ClassSource[]> {
  const res = await apiFetch<{ class_sources?: ClassSource[] }>(
    `${scoped()}/class_sources`,
    {},
    signal,
  );
  return (res.class_sources ?? []).filter(
    (c) => typeof c?.id === 'string' && c.id.length > 0 && typeof c.label === 'string',
  );
}

export function syncClassesToOpensearch(
  signal?: AbortSignal,
): Promise<{ upserted: number; n_classes: number }> {
  return apiFetch<{ upserted: number; n_classes: number }>(
    `${scoped()}/classes/sync_to_opensearch`,
    { method: 'POST' },
    signal,
  );
}
