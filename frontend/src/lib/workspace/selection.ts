/**
 * What the Workspace has selected, and the exact `ItemSelection` it sends
 * for a bulk write. No selection ever carries an exclusion list: a gallery
 * selection is the filter plus `image_ids`, a Clusters selection the filter
 * plus `cluster_uids`, "all matching" the filter alone.
 */
import type { ItemFilter, ItemSelection } from '$lib/types_itemFilter';
import type { TableRows, ViewId } from './workspaceUrl';

export type WorkspaceSelection =
  | { kind: 'none' }
  | { kind: 'items'; cropIds: string[] }
  | { kind: 'images'; imageIds: string[] }
  | { kind: 'clusters'; clusterUids: string[] }
  | { kind: 'matching' };

export type SelectionKind = 'items' | 'images' | 'clusters';

export const NO_SELECTION: WorkspaceSelection = { kind: 'none' };

function withFilter(filter: ItemFilter): ItemSelection {
  return { filter };
}

export function toItemSelection(
  sel: WorkspaceSelection,
  filter: ItemFilter,
): ItemSelection | null {
  switch (sel.kind) {
    case 'none':
      return null;
    case 'items':
      return { crop_ids: [...sel.cropIds] };
    case 'images':
      return withFilter({ ...filter, image_ids: [...sel.imageIds] });
    case 'clusters':
      return withFilter({ ...filter, cluster_uids: [...sel.clusterUids] });
    case 'matching':
      return withFilter({ ...filter });
  }
}

export function selectionKindFor(
  view: ViewId,
  rows: TableRows = 'detections',
): SelectionKind {
  switch (view) {
    case 'gallery':
    case 'map':
      return 'images';
    case 'clusters':
      return 'clusters';
    case 'table':
      return rows === 'images' ? 'images' : 'items';
    default:
      return 'items';
  }
}

/** The ids a selection holds, `[]` for `none` and `matching`. */
function selectedIds(sel: WorkspaceSelection): string[] {
  switch (sel.kind) {
    case 'items':
      return sel.cropIds;
    case 'images':
      return sel.imageIds;
    case 'clusters':
      return sel.clusterUids;
    default:
      return [];
  }
}

function selectionOf(kind: SelectionKind, ids: string[]): WorkspaceSelection {
  if (ids.length === 0) return NO_SELECTION;
  switch (kind) {
    case 'items':
      return { kind, cropIds: ids };
    case 'images':
      return { kind, imageIds: ids };
    case 'clusters':
      return { kind, clusterUids: ids };
  }
}

/**
 * Ctrl/Cmd+A. The first press selects the loaded page; a second press, when
 * the whole page is already selected, asks to select all `total` matching
 * (`pending`). Confirming it sets `{ kind: 'matching' }`.
 */
export function pressSelectAll(
  current: WorkspaceSelection,
  kind: SelectionKind,
  pageIds: string[],
  total: number,
): { selection: WorkspaceSelection; pending: { total: number } | null } {
  const held = new Set(current.kind === kind ? selectedIds(current) : []);
  const pageSelected = pageIds.length > 0 && pageIds.every((id) => held.has(id));
  if (pageSelected && total > pageIds.length) {
    return { selection: current, pending: { total } };
  }
  return { selection: selectionOf(kind, pageIds), pending: null };
}
