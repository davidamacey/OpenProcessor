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
import { editorHref } from '$lib/editor/editorUrl';
import { getCrop } from '$lib/api_cropRead';
import { projectHref } from '$lib/projectPaths';
import { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';
import { nextStepsStore } from '$stores/nextSteps.svelte';
import { buildItemQuery } from './itemQuery';
import {
  createSelectionSummary,
  type SelectionSummaryState,
} from './selectionSummary.svelte';
import type { Cluster, Crop, HiddenCounts } from '$lib/types';
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

/** What select-all acts on: the ids a view has loaded, and the served total behind them. */
export interface SelectablePage {
  kind: SelectionKind;
  ids: () => string[];
  total: () => number;
  /** Puts `ids` into the view's own selection, which the view mirrors back to the controller. */
  apply: (ids: string[]) => void;
}

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

  /** The query a write started from last, until the navigation it asked for has landed: a second
   *  write in the same tick starts from it, not from the URL that has not changed yet. */
  #pending: string | null = null;

  #base(): SvelteURLSearchParams {
    return new SvelteURLSearchParams(this.#pending ?? page.url.search);
  }

  #write(next: Partial<WorkspaceUrlState>, replace: boolean): void {
    const params = this.#base();
    writeWorkspaceUrl(params, next);
    this.#go(params, replace);
  }

  /** The URL's query, for the keys a view owns (cluster-list sort and filters). */
  get searchParams(): Pick<URLSearchParams, 'get' | 'getAll' | 'has'> {
    return page.url.searchParams;
  }

  /** Rewrites the keys a view owns, leaving every other key as it is. */
  patchUrl(mutate: (params: URLSearchParams) => void, replace = true): void {
    const params = this.#base();
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
    this.#pending = search;
    void Promise.resolve(
      goto(resolve(projectHref(`/workspace${search ? `?${search}` : ''}`)), {
        replace,
        reset: false,
      }),
    ).finally(() => {
      if (this.#pending === search) this.#pending = null;
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

  /** Opens the focused image, or the focused item's image, in the editor with this query. */
  async openEditor(): Promise<void> {
    const focus = this.#url.focus;
    if (focus?.kind !== 'image' && focus?.kind !== 'item') return;
    const imageId =
      focus.kind === 'image' ? focus.id : (await getCrop(focus.id)).image_id;
    if (imageId) void goto(resolve(editorHref(imageId, page.url.search)));
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

  #adoptReviewItem: ((crop: Crop) => void) | null = null;

  /** The Review view's way to take a crop the details panel re-read (after a Reprocess). */
  setReviewAdopt(adopt: ((crop: Crop) => void) | null): void {
    this.#adoptReviewItem = adopt;
  }

  adoptReviewItem(crop: Crop): void {
    this.#adoptReviewItem?.(crop);
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
    const params = this.#base();
    this.filter.toUrl(params);
    if (view) writeWorkspaceUrl(params, { view });
    this.#go(params, false);
  }

  removeChip(param: string, value?: string): void {
    const params = this.#base();
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
    return buildItemQuery(this.#applied, this.#url, include);
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

  #page: SelectablePage | null = null;

  /** The active view's loaded page, for select-all. Returns the function that unregisters it. */
  registerPage(page: SelectablePage): () => void {
    this.#page = page;
    return () => {
      if (this.#page === page) this.#page = null;
    };
  }

  /** Ctrl/Cmd+A over the view's loaded page; a second press on a full page asks for all `total`
   *  matching. False when the view has nothing to select, so the key keeps its browser meaning. */
  pressSelectAll(): boolean {
    const page = this.#page;
    if (!page) return false;
    const ids = page.ids();
    const r = pressSelectAll(this.selection, page.kind, ids, page.total());
    this.pendingSelectAll = r.pending;
    if (!r.pending) {
      this.selection = r.selection;
      page.apply(ids);
    }
    return true;
  }

  /** Makes `ids` the selection (a served sample, say) and shows it in the view's own grid. */
  selectItems(ids: string[]): void {
    this.select(ids.length > 0 ? { kind: 'items', cropIds: ids } : NO_SELECTION);
    if (this.#page?.kind === 'items') this.#page.apply(ids);
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
