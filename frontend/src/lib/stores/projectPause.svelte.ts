/**
 * Per-project pipeline pause (projects P2, `projects_plan.md` §5.1):
 * the served `paused` flag of each project, keyed by slug. Each project
 * is addressed through its own served `prefix` (`GET|POST {prefix}/pause`,
 * `POST {prefix}/resume`), so `/projects` can pause any row and the
 * switcher can show the active project's chip.
 *
 * Thin frontend: the value is exactly what the server last answered.
 * The served state is only the project's own flag — the server doesn't
 * say whether the global GPU-training claim is also holding its
 * workers, so nothing here guesses a reason. Not reset on a project
 * change: it is keyed by slug and every value belongs to its project.
 */
import { SvelteMap } from 'svelte/reactivity';
import { getProjectPause, pauseProject, projectErrorText, resumeProject } from '$lib/api';
import type { ProjectSummary } from '$lib/types_projects';

type PauseTarget = Pick<ProjectSummary, 'slug' | 'prefix'>;

export type PauseResult = { ok: true; paused: boolean } | { ok: false; message: string };

class ProjectPauseStore {
  #paused = new SvelteMap<string, boolean>();
  #seq = new Map<string, number>();

  /** The served `paused` for a slug, or `undefined` until it's loaded
   *  (or when the read failed). */
  pausedFor(slug: string): boolean | undefined {
    return this.#paused.get(slug);
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
      if (this.#seq.get(project.slug) === mine)
        this.#paused.set(project.slug, res.paused);
    } catch {
      if (this.#seq.get(project.slug) === mine) this.#paused.delete(project.slug);
    }
  }

  /** `POST {prefix}/pause` or `/resume`; stores the served result. */
  async set(project: PauseTarget, paused: boolean): Promise<PauseResult> {
    const mine = this.#bump(project.slug);
    try {
      const res = paused ? await pauseProject(project) : await resumeProject(project);
      if (this.#seq.get(project.slug) === mine)
        this.#paused.set(project.slug, res.paused);
      return { ok: true, paused: res.paused };
    } catch (e) {
      return { ok: false, message: projectErrorText(e) };
    }
  }

  /** Test-only. */
  reset(): void {
    this.#paused.clear();
    this.#seq.clear();
  }
}

export const projectPauseStore = new ProjectPauseStore();
