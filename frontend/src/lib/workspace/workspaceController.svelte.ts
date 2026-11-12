/**
 * The Workspace's one controller. URL state (view, queue, scope, focus,
 * sort, filter) is derived from `page.url`; nothing copies it into `$state`
 * except the filter bar's draft, which is re-seeded whenever the URL
 * changes, so Back and Forward restore view, filter, queue and scope. The
 * selection lives here, not in the URL, and survives a view switch because
 * the shell stays mounted.
 *
 * `itemQuery()` and `itemBody()` are the only places that combine the
 * filter, the queue and the scope: views call them and never build a filter.
 */
import { MediaQuery, SvelteURLSearchParams } from 'svelte/reactivity';
import { goto } from '$app/navigation';
import { resolve } from '$app/paths';
import { page } from '$app/state';
import { projectHref } from '$lib/projectPaths';
import { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';
import { nextStepsStore } from '$stores/nextSteps.svelte';
import {
  createSelectionSummary,
  type SelectionSummaryState,
} from './selectionSummary.svelte';
import type { Cluster, HiddenCounts } from '$lib/types';
import type { ItemFilter, ItemFilterQuery, ItemSelection } from '$lib/types_itemFilter';
import {
  NO_SELECTION,
  pressSelectAll,
  selectionKindFor,
  toItemSelection,
  type SelectionKind,
  type WorkspaceSelection,
} from './selection';
import {
  parseWorkspaceUrl,
  writeWorkspaceUrl,
  type FocusRef,
  type ScopeRef,
  type TableRows,
  type ViewId,
  type WorkspaceUrlState,
} from './workspaceUrl';

const DETAILS_MEDIA = 'min-width: 1280px';

export class WorkspaceController {
  /** The bar's draft. Applying it writes the URL; the URL re-seeds it. */
  readonly filter = new ItemFilterState();
  selection = $state<WorkspaceSelection>(NO_SELECTION);
  /** Set by a second select-all on a full page: the served total to confirm. */
  pendingSelectAll = $state<{ total: number } | null>(null);
  /** The user's choice for the details panel; unset, it follows the screen width. */
  #detailsChoice = $state<boolean | null>(null);
  readonly #wide =
    typeof matchMedia === 'function' ? new MediaQuery(DETAILS_MEDIA) : null;
  /** The active view's served total and what its filter leaves out, for the filter row. */
  matching = $state<{ total: number; hidden: HiddenCounts | null } | null>(null);
  /** The class picker the bulk bar owns; `workspace.class_picker` opens it. */
  pickerOpen = $state(false);
  /** The queues and classes rail as a drawer under `lg`. */
  railOpen = $state(false);
  /** The server's account of the selection: what a bulk action would touch. */
  summary!: SelectionSummaryState;
  /** The grid Review was opened from; `closeReview` returns to it. A Review that was
   *  deep-linked has no origin and returns to the Detections grid. */
  #reviewOrigin = $state<ViewId>('detections');
  /** The item Review stands on, kept here as well as in the URL so a quick `Esc` right after a
   *  confirm still returns to it, before the URL write has landed. */
  #reviewFocus: string | null = null;
  /** The open cluster's served card, set by the cluster scope for the details panel. */
  scopeCard = $state<Cluster | null>(null);

  readonly #url = $derived(parseWorkspaceUrl(page.url.searchParams));
  readonly #applied = $derived.by(() => {
    const f = new ItemFilterState();
    f.fromUrl(page.url.searchParams);
    return f;
  });
  readonly #stop: () => void;

  constructor() {
    this.#stop = $effect.root(() => {
      this.summary = createSelectionSummary(() => this.itemSelection());
      $effect(() => {
        this.filter.fromUrl(page.url.searchParams);
      });
      $effect(() => {
        const served = nextStepsStore.defaultView;
        if (this.#url.view === null && served) this.#write({ view: served }, true);
      });
    });
  }

  get url(): WorkspaceUrlState {
    return this.#url;
  }

  get view(): ViewId {
    return this.#url.view ?? nextStepsStore.defaultView ?? 'gallery';
  }

  /** The kind of selection the current view makes. */
  get selectionKind(): SelectionKind {
    return selectionKindFor(this.view, this.#url.rows);
  }

  #write(next: Partial<WorkspaceUrlState>, replace: boolean): void {
    const params = new SvelteURLSearchParams(page.url.search);
    writeWorkspaceUrl(params, next);
    this.#go(params, replace);
  }

  /** The URL's query, for the keys a view owns (cluster-list sort and filters). */
  get searchParams(): Pick<URLSearchParams, 'get' | 'getAll' | 'has'> {
    return page.url.searchParams;
  }

  /** Rewrites the keys a view owns, leaving every other key as it is. */
  patchUrl(mutate: (params: URLSearchParams) => void, replace = true): void {
    const params = new SvelteURLSearchParams(page.url.search);
    mutate(params);
    this.#go(params, replace);
  }

  /** The open-vocabulary "Matching items" mode of the Clusters view. */
  get matchingMode(): boolean {
    return (
      page.url.searchParams.get('mode') === 'matching' ||
      !!this.#applied.openVocabSet ||
      !!this.#applied.sourcePrompt
    );
  }

  setMatchingMode(on: boolean): void {
    if (!on) {
      this.filter.openVocabSet = null;
      this.filter.sourcePrompt = null;
    }
    this.patchUrl((params) => {
      this.filter.toUrl(params);
      if (on) params.set('mode', 'matching');
      else params.delete('mode');
    });
  }

  #go(params: URLSearchParams, replace: boolean): void {
    const search = params.toString();
    void goto(resolve(projectHref(`/workspace${search ? `?${search}` : ''}`)), {
      replace,
      reset: false,
    });
  }

  setView(view: ViewId): void {
    this.#write({ view }, false);
  }

  /** Another queue opens in its own order: the Review sort belongs to the queue it was picked in. */
  setQueue(id: string | null): void {
    this.#write({ queue: id, sort: { review: '' } }, false);
  }

  setScope(scope: ScopeRef | null): void {
    this.#write({ scope }, false);
  }

  /** Opens a cluster in place; the filter and the view stay. */
  openCluster(uid: string | null): void {
    this.setScope(uid ? { kind: 'cluster', id: uid } : null);
  }

  /** Opens Review at one item, over the same filter, queue and scope; the item stays focused. */
  openReview(cropId: string): void {
    if (this.view !== 'review') this.#reviewOrigin = this.view;
    this.#reviewFocus = cropId;
    this.#write({ view: 'review', focus: { kind: 'item', id: cropId } }, false);
  }

  /** Review moved on to another item: the focus follows it (a replace, not a history entry). */
  setReviewFocus(cropId: string): void {
    this.#reviewFocus = cropId;
    this.setFocus({ kind: 'item', id: cropId });
  }

  /** Leaves Review for the grid it was opened from, focused on the item Review stopped on. */
  closeReview(): void {
    const focus: FocusRef | null = this.#reviewFocus
      ? { kind: 'item', id: this.#reviewFocus }
      : this.#url.focus;
    this.#reviewFocus = null;
    this.#write({ view: this.#reviewOrigin, focus }, false);
  }

  setFocus(focus: FocusRef | null): void {
    this.#write({ focus }, true);
  }

  setSort(view: ViewId, sortId: string): void {
    this.#write({ sort: { [view]: sortId } }, false);
  }

  setTable(rows: TableRows, columns: string[] | null = null): void {
    this.#write({ rows, columns }, false);
  }

  /** Writes the draft to the URL (and `view`, in the same navigation); the filter keys are the draft's own. */
  applyFilter(view?: ViewId): void {
    const params = new SvelteURLSearchParams(page.url.search);
    this.filter.toUrl(params);
    if (view) writeWorkspaceUrl(params, { view });
    this.#go(params, false);
  }

  removeChip(param: string, value?: string): void {
    const params = new SvelteURLSearchParams(page.url.search);
    const state = new ItemFilterState();
    state.fromUrl(params);
    const chip = state
      .chips()
      .find((c) => c.param === param && (value === undefined || c.label === value));
    chip?.clear();
    state.toUrl(params);
    this.#go(params, false);
  }

  /** The applied filter, the queue as `review_tab` and the scope, as query parameters.
   *  `include` drops the filter keys a view does not apply. */
  itemQuery(include?: (param: string) => boolean): ItemFilterQuery & {
    review_tab?: string;
  } {
    const q: ItemFilterQuery & { review_tab?: string } = this.#applied.toQuery(include);
    const { queue, scope } = this.#url;
    if (queue) q.review_tab = queue;
    if (scope?.kind === 'cluster') q.cluster_uid = [scope.id];
    if (scope?.kind === 'near_dup') q.near_dup_group = scope.id;
    if (scope?.kind === 'image') q.image_id = [scope.id];
    return q;
  }

  /** The same combination in `ItemFilter` body form. */
  itemBody(): ItemFilter {
    const b: ItemFilter = this.#applied.toBody();
    const { queue, scope } = this.#url;
    if (queue) b.review_tab = queue;
    if (scope?.kind === 'cluster') b.cluster_uids = [scope.id];
    if (scope?.kind === 'near_dup') b.near_dup_group = scope.id;
    if (scope?.kind === 'image') b.image_ids = [scope.id];
    return b;
  }

  itemSelection(): ItemSelection | null {
    return toItemSelection(this.selection, this.itemBody());
  }

  select(selection: WorkspaceSelection): void {
    this.selection = selection;
    this.pendingSelectAll = null;
  }

  clearSelection(): void {
    this.select(NO_SELECTION);
  }

  /** Ctrl/Cmd+A over the loaded page; `total` is the served total. */
  selectAll(pageIds: string[], total: number): void {
    const r = pressSelectAll(this.selection, this.selectionKind, pageIds, total);
    this.selection = r.selection;
    this.pendingSelectAll = r.pending;
  }

  confirmSelectAllMatching(): void {
    this.select({ kind: 'matching' });
  }

  toggleRail(): void {
    this.railOpen = !this.railOpen;
  }

  get detailsOpen(): boolean {
    return this.#detailsChoice ?? this.#wide?.current ?? false;
  }

  toggleDetails(): void {
    this.#detailsChoice = !this.detailsOpen;
  }

  dispose(): void {
    this.#stop();
  }
}

export function createWorkspaceController(): WorkspaceController {
  return new WorkspaceController();
}
