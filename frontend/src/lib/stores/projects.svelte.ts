/**
 * ProjectsStore — the served project list and the active project
 * (`docs/design/any-domain-rev3-and-projects-contract-review-2026-09-26.md`
 * §7). The active project lives in the URL path ONLY (`/p/<slug>/...`,
 * owner decision): nothing is persisted. `load()` reads the global
 * `GET {globalApi()}/projects` once at boot; the `/p/[project]` layout
 * then resolves its slug (`resolve()`) and makes it active (`select()`),
 * which moves `scoped()` to the project's served `prefix` and runs every
 * registered project-change reset hook ($lib/projectChange).
 *
 * Thin frontend: which projects exist, which can be opened
 * (`selectable`), written (`writable`) or deleted (`deletable`) are all
 * served; nothing here decides them.
 *
 * A failed boot load leaves `error` set — the root layout renders a
 * blocking error state instead of the app.
 */

import { ApiError, getProject, getProjects, setScopedPrefix } from '$lib/api';
import { onProjectChange, resetForProjectChange } from '$lib/projectChange';
import type {
  ProjectCapacity,
  ProjectLabels,
  ProjectLimits,
  ProjectSummary,
  ProjectsResponse,
} from '$lib/types_projects';

export { onProjectChange, resetForProjectChange };

const LOAD_TIMEOUT_MS = 2000;
const RETRY_DELAYS_MS = [250, 750];

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/** What a `/p/<slug>` URL resolves to. */
export type ProjectResolution =
  | { kind: 'ok'; project: ProjectSummary }
  /** No such project (unknown, deleted, or never existed). */
  | { kind: 'not_found'; slug: string }
  /** Exists, but the server says it can't be opened (`selectable:
   *  false`, e.g. `building`/`failed`/`deleting`). */
  | { kind: 'unavailable'; project: ProjectSummary };

class ProjectsStore {
  list = $state<ProjectSummary[]>([]);
  defaultSlug = $state<string | null>(null);
  capacity = $state<ProjectCapacity | null>(null);
  limits = $state<ProjectLimits | null>(null);
  labels = $state<ProjectLabels | null>(null);
  current = $state<ProjectSummary | null>(null);
  /** Bumped on every change of active project. */
  generation = $state<number>(0);
  loaded = $state<boolean>(false);
  error = $state<string | null>(null);

  /** The served selectable projects, i.e. the switcher's vocabulary. The
   *  active project is always included even when the default list
   *  doesn't carry it (an archived project opened by deep link). */
  get selectable(): ProjectSummary[] {
    const out = this.list.filter((p) => p.selectable);
    const cur = this.current;
    if (cur && !out.some((p) => p.slug === cur.slug)) out.push(cur);
    return out;
  }

  /** Where `/` and the legacy bare paths go: the served `default_slug`
   *  when it is selectable, else the first selectable project, else
   *  `null` (the caller sends the operator to `/projects`). */
  get defaultProject(): ProjectSummary | null {
    const sel = this.list.filter((p) => p.selectable);
    return sel.find((p) => p.slug === this.defaultSlug) ?? sel[0] ?? null;
  }

  /** Served display copy for a status; the raw status when unlabelled. */
  statusLabel(status: string): string {
    return this.labels?.status?.[status] ?? status;
  }

  #apply(res: ProjectsResponse): void {
    this.list = res.projects;
    this.defaultSlug = res.default_slug;
    this.capacity = res.capacity;
    this.limits = res.limits;
    this.labels = res.labels ?? null;
    const cur = this.current;
    if (cur) {
      const fresh = res.projects.find((p) => p.slug === cur.slug);
      if (fresh) this.current = fresh;
    }
  }

  /** Boot entry point, called once from the root layout's `load()`.
   *  Bounded retries like `loadRegionProfile`; never throws — a
   *  persistent failure sets `error`, which the layout renders as a
   *  blocking error state. */
  async load(): Promise<void> {
    if (this.loaded) return;
    for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
      if (attempt > 0) await delay(RETRY_DELAYS_MS[attempt - 1]!);
      try {
        this.#apply(await getProjects(AbortSignal.timeout(LOAD_TIMEOUT_MS)));
        this.loaded = true;
        this.error = null;
        return;
      } catch {
        // timeout / network error — retry within the budget
      }
    }
    this.error = "Can't reach the backend's project list.";
  }

  /** Re-reads the list (after a lifecycle action, or on a global
   *  `project.*` event). A failure keeps the last good list. */
  async refresh(): Promise<void> {
    try {
      this.#apply(await getProjects());
      this.loaded = true;
      this.error = null;
    } catch {
      // keep the last good list
    }
  }

  /** Upserts a summary a lifecycle response returned, so the switcher
   *  reflects it without a re-read. A summary the server no longer
   *  lists as selectable stays in the list: `selectable` is served. */
  adopt(project: ProjectSummary): void {
    const i = this.list.findIndex((p) => p.slug === project.slug);
    if (i >= 0) this.list[i] = project;
    else this.list = [...this.list, project];
    if (this.current?.slug === project.slug) this.current = project;
  }

  /** Resolves a `/p/<slug>` segment. A slug the default list doesn't
   *  carry (e.g. archived) is read from `GET /projects/{slug}`; a 404
   *  there is "not found". Any other failure is also reported as not
   *  found — the page offers a link to the project list either way. */
  async resolve(slug: string): Promise<ProjectResolution> {
    let project = this.list.find((p) => p.slug === slug) ?? null;
    if (!project) {
      try {
        project = await getProject(slug);
      } catch (e) {
        if (e instanceof ApiError && e.status !== 404) {
          console.warn(`[projects] resolving '${slug}' failed: ${e.message}`);
        }
        return { kind: 'not_found', slug };
      }
    }
    if (!project.selectable) return { kind: 'unavailable', project };
    return { kind: 'ok', project };
  }

  /**
   * Makes `project` active: `scoped()` moves to its served `prefix`,
   * and — only when this is a CHANGE of project, not the first
   * selection or a fresh summary for the same one — every registered
   * reset hook runs. Returns whether the project changed.
   */
  select(project: ProjectSummary): boolean {
    const prev = this.current;
    this.current = project;
    setScopedPrefix(project.prefix);
    if (prev && prev.slug === project.slug) return false;
    this.generation += 1;
    if (prev) resetForProjectChange();
    return true;
  }

  /** Test-only. */
  reset(): void {
    this.list = [];
    this.defaultSlug = null;
    this.capacity = null;
    this.limits = null;
    this.labels = null;
    this.current = null;
    this.generation = 0;
    this.loaded = false;
    this.error = null;
  }
}

export const projectsStore = new ProjectsStore();
