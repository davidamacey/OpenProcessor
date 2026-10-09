/**
 * Per-project pipeline pause (projects P2, `projects_plan.md` §5.1): the
 * last served `PipelinePauseState` of a project, keyed by slug. Each
 * project is addressed through its own served `prefix` (`GET|POST
 * {prefix}/pause`, `POST {prefix}/resume`), so `/projects` can pause any
 * row and the switcher can explain the active project's chip.
 *
 * Thin frontend: the value is exactly what the server last answered —
 * `paused` (the project's own flag OR the global GPU-training claim),
 * `paused_by` and the served `reason`; nothing here guesses a cause. The
 * list view's plain `paused` comes from the served `ProjectSummary`, so
 * `/projects` needs no per-row read. Not reset on a project change: it is
 * keyed by slug and every value belongs to its project.
 */
import { SvelteMap } from 'svelte/reactivity';
import { getProjectPause, pauseProject, apiErrorText, resumeProject } from '$lib/api';
import type { PipelinePauseState, ProjectSummary } from '$lib/types_projects';

type PauseTarget = Pick<ProjectSummary, 'slug' | 'prefix'>;

export type PauseResult = { ok: true; paused: boolean } | { ok: false; message: string };

class ProjectPauseStore {
  #state = new SvelteMap<string, PipelinePauseState>();
  #seq = new Map<string, number>();

  /** The last served pause state for a slug, or `undefined` until it's
   *  loaded (or when the read failed). */
  stateFor(slug: string): PipelinePauseState | undefined {
    return this.#state.get(slug);
  }

  /** The served `paused` for a slug, or `undefined` until it's loaded
   *  (or when the read failed). */
  pausedFor(slug: string): boolean | undefined {
    return this.#state.get(slug)?.paused;
  }

  #bump(slug: string): number {
    const n = (this.#seq.get(slug) ?? 0) + 1;
    this.#seq.set(slug, n);
    return n;
  }

  /** Reads `GET {prefix}/pause`. Never throws; a failure forgets the
   *  value so no stale chip survives. */
  async load(project: PauseTarget): Promise<void> {
    const mine = this.#bump(project.slug);
    try {
      const res = await getProjectPause(project);
      if (this.#seq.get(project.slug) === mine) this.#state.set(project.slug, res);
    } catch {
      if (this.#seq.get(project.slug) === mine) this.#state.delete(project.slug);
    }
  }

  /** `POST {prefix}/pause` or `/resume`; stores the served result. */
  async set(project: PauseTarget, paused: boolean): Promise<PauseResult> {
    const mine = this.#bump(project.slug);
    try {
      const res = paused ? await pauseProject(project) : await resumeProject(project);
      if (this.#seq.get(project.slug) === mine) this.#state.set(project.slug, res);
      return { ok: true, paused: res.paused };
    } catch (e) {
      return { ok: false, message: apiErrorText(e) };
    }
  }

  /** Test-only. */
  reset(): void {
    this.#state.clear();
    this.#seq.clear();
  }
}

export const projectPauseStore = new ProjectPauseStore();
