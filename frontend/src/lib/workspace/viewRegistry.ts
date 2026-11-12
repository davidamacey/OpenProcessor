/**
 * The Workspace's views. Each view lane appends one entry; the shell renders
 * whichever entry the URL names and never branches on a view id.
 */
import type { Component } from 'svelte';
import ClustersView from './views/ClustersView.svelte';
import GalleryView from './views/GalleryView.svelte';
import DetectionsView from './views/DetectionsView.svelte';
import MapView from './views/MapView.svelte';
import { mapAvailabilityStore } from '$stores/mapAvailability.svelte';
import { page } from '$app/state';
import type { WorkspaceController } from './workspaceController.svelte';
import type { ViewId } from './workspaceUrl';

export interface ViewEntry {
  id: ViewId;
  /** A fixed product term, not data. */
  label: string;
  component: Component<{ ws: WorkspaceController }>;
  /** The keymap context the view's keys are registered under. */
  keyContext: 'cluster' | 'map';
  /** Absent means always available; the Map view needs a project that serves a map. */
  available?: (ws: WorkspaceController) => boolean;
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
  {
    id: 'map',
    label: 'Map',
    component: MapView,
    keyContext: 'map',
    available: () => mapAvailabilityStore.availableFor(page.params.project),
  },
];

/** The views the tabs and the view keys offer for this project. */
export function availableViews(ws: WorkspaceController): ViewEntry[] {
  return VIEWS.filter((v) => !v.available || v.available(ws));
}

/** The entry for `id`; a view this build does not have opens the first one. */
export function viewEntry(id: ViewId): ViewEntry {
  return VIEWS.find((v) => v.id === id) ?? VIEWS[0]!;
}
