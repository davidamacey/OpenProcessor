/**
 * Which views can show the details of each kind of focus. Kept apart from the registry so the
 * controller can ask without importing the detail components.
 */
import type { ViewId } from './workspaceUrl';

export type DetailsKind = 'item' | 'review' | 'image' | 'cluster' | 'cell' | 'selection';

export const DETAILS_VIEWS: Record<DetailsKind, readonly ViewId[]> = {
  item: ['clusters', 'detections', 'table', 'embedding', 'review'],
  review: ['review'],
  image: ['gallery', 'detections', 'table', 'map'],
  cluster: ['clusters'],
  cell: ['map'],
  selection: ['gallery', 'clusters', 'detections', 'table', 'review', 'map', 'embedding'],
};

/** True when `view` can show the details of a focus of this kind. A kind the registry does not
 *  know is never allowed. */
export function focusFitsView(kind: string, view: ViewId): boolean {
  return (
    (DETAILS_VIEWS as Record<string, readonly ViewId[]>)[kind]?.includes(view) ?? false
  );
}
