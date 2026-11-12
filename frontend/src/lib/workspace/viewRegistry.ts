/**
 * The Workspace's views. Each view lane appends one entry; the shell renders
 * whichever entry the URL names and never branches on a view id.
 */
import type { Component } from 'svelte';
import ClustersView from './views/ClustersView.svelte';
import GalleryView from './views/GalleryView.svelte';
import DetectionsView from './views/DetectionsView.svelte';
import type { WorkspaceController } from './workspaceController.svelte';
import type { ViewId } from './workspaceUrl';

export interface ViewEntry {
  id: ViewId;
  /** A fixed product term, not data. */
  label: string;
  component: Component<{ ws: WorkspaceController }>;
  /** The keymap context the view's keys are registered under. */
  keyContext: 'cluster';
}

export const VIEWS: readonly ViewEntry[] = [
  { id: 'gallery', label: 'Gallery', component: GalleryView, keyContext: 'cluster' },
  { id: 'clusters', label: 'Clusters', component: ClustersView, keyContext: 'cluster' },
  {
    id: 'detections',
    label: 'Detections',
    component: DetectionsView,
    keyContext: 'cluster',
  },
];

/** The entry for `id`; a view this build does not have opens the first one. */
export function viewEntry(id: ViewId): ViewEntry {
  return VIEWS.find((v) => v.id === id) ?? VIEWS[0]!;
}
