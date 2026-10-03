/**
 * `/projects` state and actions (P3 lifecycle), in the same factory
 * convention as `clusterController`/`reviewController`: the page owns
 * dialogs and form fields; this owns the served list and every write.
 *
 * Thin frontend: every write sends exactly what the served API asks for
 * (`expected_revision` from the served `revision`, `confirm` = the slug)
 * and every refusal is rendered from the served
 * `{detail: {error, message}}` — `message` verbatim, `error` only to pick
 * the follow-up (a `revision_conflict` offers a reload). Which actions a
 * row offers is decided by the page from served flags alone.
 */

import {
  archiveProject,
  cloneProjectSettings,
  createProject,
  deleteProject,
  deleteProjectDryRun,
  getProjects,
  patchProject,
  projectErrorDetail,
  projectErrorText,
  unarchiveProject,
} from '$lib/api';
import type {
  CreateProjectRequest,
  DeleteDryRunResponse,
  KeymapCloneConflict,
  ProjectCapacity,
  ProjectErrorDetail,
  ProjectLabels,
  ProjectLifecycleResponse,
  ProjectLimits,
  ProjectSummary,
  ProjectWarning,
} from '$lib/types_projects';
import { projectsStore } from '$stores/projects.svelte';
import { toastStore } from '$stores/toast.svelte';

/** The outcome of one lifecycle write. `code` is the served
 *  `detail.error` (e.g. `revision_conflict`, `project_protected`) when
 *  the refusal was structured, else `null`; `message` is always the text
 *  to show, verbatim from the server when it sent one. */
export type ActionResult =
  | { ok: true; project: ProjectSummary }
  | {
      ok: false;
      code: string | null;
      message: string;
      detail: ProjectErrorDetail | null;
    };

export type DryRunResult =
  | { ok: true; report: DeleteDryRunResponse }
  | {
      ok: false;
      code: string | null;
      message: string;
      detail: ProjectErrorDetail | null;
    };

/** The served warnings on a lifecycle envelope, each as its own toast. */
export function toastWarnings(warnings: ProjectWarning[] | undefined): void {
  for (const w of warnings ?? []) {
    toastStore.push({ kind: 'warn', text: w.message, ttl_ms: 8000 });
  }
}

/** The keymap clone axis drops any action whose combo is already a class
 *  hotkey in the target: a report, never a silent unbind. One toast lists
 *  every served conflict (action id, combo, the class holding the key). */
export function keymapConflictText(conflicts: KeymapCloneConflict[]): string {
  const rows = conflicts.map(
    (c) => `${c.action_id} (${c.combo} is the hotkey of class "${c.class_name}")`,
  );
  return `${conflicts.length} keyboard shortcut${conflicts.length === 1 ? ' was' : 's were'} not copied: ${rows.join('; ')}.`;
}

export function toastKeymapConflicts(conflicts: KeymapCloneConflict[] | undefined): void {
  if (!conflicts?.length) return;
  toastStore.push({ kind: 'warn', text: keymapConflictText(conflicts), ttl_ms: 15000 });
}

function failure(e: unknown): Extract<ActionResult, { ok: false }> {
  const detail = projectErrorDetail(e);
  return {
    ok: false,
    code: detail?.error ?? null,
    message: projectErrorText(e),
    detail,
  };
}

/** Served statuses that are transient: the server is still working, so
 *  the row's next state only shows up on a re-read. */
const TRANSIENT_STATUSES = ['building', 'deleting'];

/** How often the list is re-read while any row is transient. */
export const TRANSIENT_POLL_MS = 2000;

export function createProjectsAdmin() {
  let list = $state<ProjectSummary[]>([]);
  let capacity = $state<ProjectCapacity | null>(null);
  let limits = $state<ProjectLimits | null>(null);
  let labels = $state<ProjectLabels | null>(null);
  let defaultSlug = $state<string | null>(null);
  let includeArchived = $state(false);
  let loading = $state(false);
  let loadError = $state<string | null>(null);
  let seq = 0;
  let watching = false;
  let pollTimer: ReturnType<typeof setTimeout> | null = null;

  function clearPoll(): void {
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = null;
  }

  /** While watching, re-read on an interval for as long as any listed
   *  row has a transient served status; stops itself once none remain. */
  function schedulePoll(): void {
    clearPoll();
    if (!watching || !list.some((p) => TRANSIENT_STATUSES.includes(p.status))) return;
    pollTimer = setTimeout(() => void load(), TRANSIENT_POLL_MS);
  }

  async function load(): Promise<void> {
    const mine = ++seq;
    loading = true;
    try {
      const res = await getProjects(undefined, includeArchived);
      if (mine !== seq) return;
      list = res.projects;
      capacity = res.capacity;
      limits = res.limits;
      labels = res.labels ?? null;
      defaultSlug = res.default_slug;
      loadError = null;
    } catch (e) {
      if (mine !== seq) return;
      loadError = projectErrorText(e);
    } finally {
      if (mine === seq) {
        loading = false;
        schedulePoll();
      }
    }
  }

  /** After any successful write: re-read this page's list and the
   *  switcher's, so both show the served state (capacity included). */
  async function afterWrite(res: ProjectLifecycleResponse): Promise<ProjectSummary> {
    toastWarnings(res.warnings);
    toastKeymapConflicts(res.keymap_clone_conflicts);
    projectsStore.adopt(res.project);
    await Promise.all([load(), projectsStore.refresh()]);
    return res.project;
  }

  async function run(
    write: () => Promise<ProjectLifecycleResponse>,
  ): Promise<ActionResult> {
    try {
      const res = await write();
      return { ok: true, project: await afterWrite(res) };
    } catch (e) {
      return failure(e);
    }
  }

  return {
    get list() {
      return list;
    },
    get capacity() {
      return capacity;
    },
    get limits() {
      return limits;
    },
    get defaultSlug() {
      return defaultSlug;
    },
    get includeArchived() {
      return includeArchived;
    },
    get loading() {
      return loading;
    },
    get loadError() {
      return loadError;
    },
    /** Served display copy for a status; the raw status when unlabelled. */
    statusLabel(status: string): string {
      return labels?.status?.[status] ?? status;
    },
    load,
    /** Begin keeping the list fresh while rows are transient. */
    start(): void {
      watching = true;
      schedulePoll();
    },
    /** Stop polling (page unmount). */
    stop(): void {
      watching = false;
      clearPoll();
    },
    async setIncludeArchived(v: boolean): Promise<void> {
      includeArchived = v;
      await load();
    },
    create(body: CreateProjectRequest): Promise<ActionResult> {
      return run(() => createProject(body));
    },
    /** Sends only the fields that changed, with the served revision. */
    edit(
      p: ProjectSummary,
      fields: { display_name?: string; description?: string },
    ): Promise<ActionResult> {
      return run(() =>
        patchProject(p.slug, { ...fields, expected_revision: p.revision }),
      );
    },
    archive(p: ProjectSummary): Promise<ActionResult> {
      return run(() => archiveProject(p.slug, { expected_revision: p.revision }));
    },
    unarchive(p: ProjectSummary): Promise<ActionResult> {
      return run(() => unarchiveProject(p.slug, { expected_revision: p.revision }));
    },
    cloneSettings(
      target: ProjectSummary,
      from: string,
      axes: string[],
    ): Promise<ActionResult> {
      return run(() =>
        cloneProjectSettings(target.slug, {
          from,
          axes,
          expected_revision: target.revision,
        }),
      );
    },
    async dryRunDelete(p: ProjectSummary): Promise<DryRunResult> {
      try {
        return { ok: true, report: await deleteProjectDryRun(p.slug) };
      } catch (e) {
        return failure(e);
      }
    },
    /** A real, guarded delete. `confirm` is what the operator typed; the
     *  server checks it equals the slug (422 `confirm_mismatch`). */
    remove(p: ProjectSummary, confirm: string): Promise<ActionResult> {
      return run(() => deleteProject(p.slug, confirm));
    },
    /** The freshest served summary for a slug (after a reload). */
    find(slug: string): ProjectSummary | undefined {
      return list.find((p) => p.slug === slug);
    },
  };
}

export type ProjectsAdmin = ReturnType<typeof createProjectsAdmin>;
