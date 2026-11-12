/**
 * The project picker's list: asks the server for one bounded page of projects matching what is
 * typed (`GET /projects?q=&limit=`), so a long project list is never loaded into, or filtered by,
 * the browser. The server also says how many matched (`total`); when that exceeds the page the
 * picker offers the next page by the served `next_cursor`.
 */
import { apiErrorText } from '$lib/api_errors';
import { getProjects } from '$lib/api_projects';
import type { ProjectSummary } from '$lib/types_projects';

const RECENT_KEY = 'op.recentProjects';
const RECENT_MAX = 5;

export interface RecentProject {
  slug: string;
  display_name: string;
}

/** Rows one page holds; the list scrolls inside this bound. */
export const PICKER_PAGE = 25;
const TYPE_DELAY_MS = 150;

export class ProjectPickerController {
  query = $state('');
  projects = $state<ProjectSummary[]>([]);
  total = $state(0);
  loading = $state(false);
  error = $state<string | null>(null);
  /** The served cursor of the next page; `null` when this is the last. */
  #cursor = $state<string | null>(null);
  /** Last-opened projects, newest first. View state only: `sessionStorage`, never the server. */
  recents = $state<RecentProject[]>(readRecents());
  /** Index of the highlighted row (keyboard focus stays in the filter box). */
  active = $state(0);

  #abort: AbortController | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;

  /** Loads the first page now (opening the picker). */
  open(): void {
    this.query = '';
    void this.#load();
  }

  /** The filter text changed: reload after a short pause in typing. */
  type(text: string): void {
    this.query = text;
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = setTimeout(() => void this.#load(), TYPE_DELAY_MS);
  }

  get hasMore(): boolean {
    return this.#cursor !== null;
  }

  /** Appends the next served page (the list scrolls; nothing is cut to a "type to narrow"). */
  async more(): Promise<void> {
    if (this.#cursor === null || this.loading) return;
    const abort = new AbortController();
    this.#abort = abort;
    this.loading = true;
    try {
      const res = await getProjects(abort.signal, false, {
        q: this.query.trim(),
        limit: PICKER_PAGE,
        selectable: true,
        cursor: this.#cursor,
      });
      if (abort.signal.aborted) return;
      this.projects = [...this.projects, ...res.projects];
      this.total = res.total;
      this.#cursor = res.next_cursor ?? null;
    } catch (e) {
      if (abort.signal.aborted) return;
      this.error = apiErrorText(e);
    } finally {
      if (this.#abort === abort) this.loading = false;
    }
  }

  remember(p: { slug: string; display_name: string }): void {
    const next = [
      { slug: p.slug, display_name: p.display_name },
      ...this.recents.filter((r) => r.slug !== p.slug),
    ].slice(0, RECENT_MAX);
    this.recents = next;
    try {
      sessionStorage.setItem(RECENT_KEY, JSON.stringify(next));
    } catch {
      // storage unavailable: recents just do not persist
    }
  }

  move(delta: number): void {
    const n = this.projects.length;
    if (n === 0) return;
    this.active = Math.min(n - 1, Math.max(0, this.active + delta));
  }

  home(): void {
    this.active = 0;
  }

  end(): void {
    this.active = Math.max(0, this.projects.length - 1);
  }

  get highlighted(): ProjectSummary | null {
    return this.projects[this.active] ?? null;
  }

  dispose(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#abort?.abort();
  }

  async #load(): Promise<void> {
    this.#abort?.abort();
    const abort = new AbortController();
    this.#abort = abort;
    this.loading = true;
    this.error = null;
    try {
      const res = await getProjects(abort.signal, false, {
        q: this.query.trim(),
        limit: PICKER_PAGE,
        selectable: true,
      });
      if (abort.signal.aborted) return;
      this.projects = res.projects;
      this.total = res.total;
      this.#cursor = res.next_cursor ?? null;
      this.active = 0;
    } catch (e) {
      if (abort.signal.aborted) return;
      this.error = apiErrorText(e);
    } finally {
      if (this.#abort === abort) this.loading = false;
    }
  }
}

function readRecents(): RecentProject[] {
  try {
    const raw = JSON.parse(sessionStorage.getItem(RECENT_KEY) ?? '[]') as unknown;
    return Array.isArray(raw) ? (raw as RecentProject[]).slice(0, RECENT_MAX) : [];
  } catch {
    return [];
  }
}
