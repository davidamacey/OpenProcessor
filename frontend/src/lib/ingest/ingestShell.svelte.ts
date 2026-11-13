/**
 * What the Ingest layout shares with its pages: the served config (fetched once), the tab the
 * URL names, the upload progress the Pipeline rail refreshes on, and the region worklog the rail
 * polls and the Activity tab shows. The layout creates one and sets it as context; the pages
 * read it. The tab is the address: `?tab=`, so reload and Back keep it, and the import job page
 * (`/ingest/imports/<id>`) is always the Import tab.
 */
import { getContext, setContext } from 'svelte';
import { apiErrorText } from '$lib/api_errors';
import { getIngestConfig } from '$lib/api_ingest';
import { resolveIngestConfig, type ResolvedIngestConfig } from '$lib/ingest/ingestConfig';
import type { IngestRunState } from '$lib/ingest/ingestRunController.svelte';
import type { IngestConfig, RegionDrain } from '$lib/types_ingest';

export const INGEST_TAB_IDS = [
  'upload',
  'folder',
  'sources',
  'import',
  'activity',
] as const;
export type IngestTabId = (typeof INGEST_TAB_IDS)[number];

interface UrlLike {
  pathname: string;
  searchParams: { get(name: string): string | null };
}

/** The tab `url` names; an unknown or missing `tab` is Upload. */
export function ingestTabOf(url: UrlLike): IngestTabId {
  if (/\/ingest\/imports(\/|$)/.test(url.pathname)) return 'import';
  const t = url.searchParams.get('tab');
  return INGEST_TAB_IDS.find((id) => id === t) ?? 'upload';
}

export class IngestShell {
  servedConfig = $state<IngestConfig | null>(null);
  configError = $state<string | null>(null);
  statusRefreshToken = $state(0);
  runState = $state<IngestRunState>('idle');
  drain = $state<RegionDrain | null>(null);
  drainError = $state(false);
  drainObservedAt = $state<number | null>(null);

  readonly config: ResolvedIngestConfig | null = $derived(
    this.servedConfig ? resolveIngestConfig(this.servedConfig) : null,
  );
  readonly #url: () => UrlLike;

  constructor(url: () => UrlLike) {
    this.#url = url;
  }

  get tab(): IngestTabId {
    return ingestTabOf(this.#url());
  }

  load(): void {
    void getIngestConfig()
      .then((c) => (this.servedConfig = c))
      .catch((e: unknown) => {
        this.configError = apiErrorText(e) ?? 'failed to load the ingest config';
      });
  }

  onDrainUpdate = (next: RegionDrain | null, observedAt: number): void => {
    this.drain = next;
    this.drainError = next === null;
    this.drainObservedAt = observedAt;
  };
}

const KEY = Symbol('ingest-shell');
export const setIngestShell = (shell: IngestShell): IngestShell => setContext(KEY, shell);
export const getIngestShell = (): IngestShell => getContext<IngestShell>(KEY);
