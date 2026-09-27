/**
 * ProjectsStore — P1 of the projects cutover
 * (`docs/design/any-domain-rev3-and-projects-contract-review-2026-09-26.md`
 * §7). Loads the global project list once at boot, picks the active
 * project (the served `is_default: true` entry that is `selectable`),
 * and seeds `setScopedPrefix()` from its served `prefix` — every scoped
 * `api.ts` call fails closed (`ProjectNotSelectedError`) until this has
 * run. There is no URL param or switcher yet (a later task); today
 * there is exactly one active project for the whole session.
 *
 * A failed load leaves `error` set — the root layout renders a blocking
 * error state instead of the app, never a half-rendered page with every
 * scoped call throwing.
 */

import { getProjects, setScopedPrefix } from '$lib/api';
import type { ProjectCapacity, ProjectSummary } from '$lib/types_projects';
import { undoStore } from '$stores/undo.svelte';
import { resetForProjectChange as resetSourceImageOverlayCache } from '$lib/components/SourceImageOverlay.svelte';

const LOAD_TIMEOUT_MS = 2000;
const RETRY_DELAYS_MS = [250, 750];

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/** Hooks registered by client-side stores/caches that must never bleed
 *  data across projects. There is only ever one project active today,
 *  so nothing calls these yet — wired up ahead of the future switcher. */
const resetHooks = new Set<() => void>();

export function onProjectChange(hook: () => void): () => void {
  resetHooks.add(hook);
  return () => resetHooks.delete(hook);
}

/** Runs every registered reset hook — call after the active project
 *  changes (not called yet in P1, since there is no switcher). */
export function resetForProjectChange(): void {
  for (const hook of resetHooks) hook();
}

class ProjectsStore {
  list = $state<ProjectSummary[]>([]);
  defaultSlug = $state<string | null>(null);
  capacity = $state<ProjectCapacity | null>(null);
  current = $state<ProjectSummary | null>(null);
  loaded = $state<boolean>(false);
  error = $state<string | null>(null);

  /** Boot entry point, called once from the root layout's `load()`
   *  before anything scoped fires. Bounded retries like
   *  `loadRegionProfile`; never throws — a persistent failure sets
   *  `error`, which the layout renders as a blocking error state. */
  async load(): Promise<void> {
    if (this.loaded) return;
    for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
      if (attempt > 0) await delay(RETRY_DELAYS_MS[attempt - 1]!);
      try {
        const res = await getProjects(AbortSignal.timeout(LOAD_TIMEOUT_MS));
        this.list = res.projects;
        this.defaultSlug = res.default_slug;
        this.capacity = res.capacity;
        const active =
          res.projects.find((p) => p.is_default && p.selectable) ??
          res.projects.find((p) => p.selectable) ??
          null;
        if (!active) {
          this.error = "The backend has no selectable project — can't continue.";
          return;
        }
        this.current = active;
        setScopedPrefix(active.prefix);
        this.loaded = true;
        this.error = null;
        return;
      } catch {
        // timeout / network error — retry within the budget
      }
    }
    this.error = "Can't reach the backend's project list.";
  }

  /** Test-only. */
  reset(): void {
    this.list = [];
    this.defaultSlug = null;
    this.capacity = null;
    this.current = null;
    this.loaded = false;
    this.error = null;
  }
}

export const projectsStore = new ProjectsStore();

// Wire the existing per-store `resetForProjectChange()` hooks (the undo
// ring buffer, the source-image-context cache) into the central
// registry above. Nothing calls `resetForProjectChange()` yet — there
// is no switcher in P1 — but the wiring is in place for it.
onProjectChange(() => undoStore.resetForProjectChange());
onProjectChange(() => resetSourceImageOverlayCache());
