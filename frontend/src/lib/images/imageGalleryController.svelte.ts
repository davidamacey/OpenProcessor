/**
 * The Gallery view's list: whole images in the order the server serves them, paged by
 * `next_cursor`. Counts, boxes, the empty sentence and the sort options are served; nothing here
 * sorts, filters or counts.
 */
import { listImages } from '$lib/api_images';
import { createPager } from '$lib/pager.svelte';
import { createSelection } from '$lib/selection.svelte';
import type { ImageCardWire } from '$lib/types_images';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type { HiddenCounts } from '$lib/types';
import type { SortOption } from '$lib/types_cursor';
import type { WorkspaceSelection } from '$lib/workspace/selection';
import { imageListQuery } from './imageQuery';

/** What the controller reads from the Workspace: the one place filter, queue and scope combine. */
export interface ImageGalleryDeps {
  itemQuery(): ItemFilterQuery;
  /** The gallery's sort id from the URL; null = the server's default. */
  sortId(): string | null;
}

export type GalleryStatus = 'loading' | 'error' | 'empty' | 'ready';

export function createImageGalleryController(deps: ImageGalleryDeps) {
  let unlabeledOnly = $state(false);
  let sortApplied = $state<string | null>(null);
  let sortOptions = $state<SortOption[]>([]);
  let emptyMessage = $state<string | null>(null);
  let hidden = $state<HiddenCounts | null>(null);
  let joinTruncated = $state(false);
  const selection = createSelection({ plainClick: 'replace' });

  const query = () =>
    imageListQuery(deps.itemQuery(), { unlabeledOnly, sort: deps.sortId() });

  const pager = createPager<ImageCardWire>({
    fetchPage: async (cursor) => {
      const page = await listImages(query(), cursor);
      sortApplied = page.sort_applied;
      sortOptions = page.sort_options;
      emptyMessage = page.empty_message;
      hidden = page.hidden;
      joinTruncated = page.join_truncated;
      return page;
    },
    keyOf: (c) => c.image_id,
  });

  const status = $derived<GalleryStatus>(
    pager.error != null
      ? 'error'
      : !pager.loaded
        ? 'loading'
        : pager.items.length === 0
          ? 'empty'
          : 'ready',
  );

  return {
    get cards() {
      return pager.items;
    },
    get total() {
      return pager.total;
    },
    get status() {
      return status;
    },
    get error() {
      return pager.error;
    },
    get hasMore() {
      return pager.hasMore;
    },
    get loadingMore() {
      return pager.loadingMore;
    },
    get emptyMessage() {
      return emptyMessage;
    },
    get sortApplied() {
      return sortApplied;
    },
    get sortOptions() {
      return sortOptions;
    },
    get hidden() {
      return hidden;
    },
    get joinTruncated() {
      return joinTruncated;
    },
    get unlabeledOnly() {
      return unlabeledOnly;
    },
    set unlabeledOnly(v: boolean) {
      unlabeledOnly = v;
    },
    /** Changes with anything that re-fetches the list; the view reloads on it. */
    get queryKey() {
      return JSON.stringify(query());
    },
    get selectedIds(): string[] {
      return [...selection.ids];
    },
    isSelected: (imageId: string) => selection.has(imageId),
    select(imageId: string, e: MouseEvent | undefined) {
      selection.click(
        imageId,
        e,
        pager.items.map((c) => c.image_id),
      );
    },
    clearSelection: () => selection.clear(),
    /** The selection as the Workspace holds it; `ws.itemSelection()` adds the filter. */
    get workspaceSelection(): WorkspaceSelection {
      return selection.size > 0
        ? { kind: 'images', imageIds: [...selection.ids] }
        : { kind: 'none' };
    },
    replaceSelection(imageIds: string[]) {
      selection.selectAll(imageIds);
    },
    loadFirst: () => pager.loadFirst(),
    loadMore: () => pager.loadMore(),
  };
}

export type ImageGalleryController = ReturnType<typeof createImageGalleryController>;
