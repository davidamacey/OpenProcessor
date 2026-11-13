/**
 * What the details panel shows: the focused thing, unless two or more are selected, which the
 * panel then counts; else a lone selection, else the open cluster. A focus the view cannot show
 * is not shown.
 */
import { DETAILS } from './detailsRegistry';
import { focusFitsView, type DetailsKind } from './detailsViews';
import type { WorkspaceSelection } from './selection';
import type { FocusRef, ScopeRef, ViewId } from './workspaceUrl';

export interface DetailsTarget {
  kind: DetailsKind;
  id: string | null;
}

function selectedCount(sel: WorkspaceSelection): number {
  switch (sel.kind) {
    case 'none':
      return 0;
    case 'items':
      return sel.cropIds.length;
    case 'images':
      return sel.imageIds.length;
    case 'clusters':
      return sel.clusterUids.length;
    case 'matching':
      return Infinity;
  }
}

export function detailsTarget(
  view: ViewId,
  focus: FocusRef | null,
  selection: WorkspaceSelection,
  scope: ScopeRef | null,
): DetailsTarget | null {
  // Beside the Review pane the item shows what the pane does not.
  if (focus?.kind === 'item' && view === 'review')
    return { kind: 'review', id: focus.id };
  if (view !== 'review' && selectedCount(selection) >= 2)
    return { kind: 'selection', id: null };
  if (focus && focus.kind in DETAILS && focusFitsView(focus.kind, view))
    return { kind: focus.kind as DetailsKind, id: focus.id };
  if (selection.kind !== 'none') return { kind: 'selection', id: null };
  if (scope?.kind === 'cluster') return { kind: 'cluster', id: scope.id };
  return null;
}
