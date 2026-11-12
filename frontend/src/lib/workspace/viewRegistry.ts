/**
 * The Workspace's views. Each view lane appends one entry; the shell renders
 * whichever entry the URL names and never branches on a view id.
 */
import type { Component } from 'svelte';
import ClustersView from './views/ClustersView.svelte';
import GalleryView from './views/GalleryView.svelte';
import DetectionsView from './views/DetectionsView.svelte';
import TableView from './views/TableView.svelte';
import MapView from './views/MapView.svelte';
import ReviewView from './views/ReviewView.svelte';
import EmbeddingView from './views/EmbeddingView.svelte';
import { isEmbeddingVizAvailable } from '$lib/strategies';
import { mapAvailabilityStore } from '$stores/mapAvailability.svelte';
import { strategiesStore } from '$stores/strategies.svelte';
import { page } from '$app/state';
import type { WorkspaceController } from './workspaceController.svelte';
import type { ViewId } from './workspaceUrl';

export interface ViewEntry {
  id: ViewId;
  /** A fixed product term, not data. */
  label: string;
  component: Component<{ ws: WorkspaceController }>;
  /** The keymap context the view's keys are registered under. */
  keyContext: 'cluster' | 'table' | 'map' | 'review';
  /** Absent means always available; the Map view needs a project that serves a map and the
   *  Embedding view a deployment that serves the projection. */
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
  { id: 'table', label: 'Table', component: TableView, keyContext: 'table' },
  { id: 'review', label: 'Review', component: ReviewView, keyContext: 'review' },
  {
    id: 'map',
    label: 'Map',
    component: MapView,
    keyContext: 'map',
    available: () => mapAvailabilityStore.availableFor(page.params.project),
  },
  {
    id: 'embedding',
    label: 'Embedding',
    component: EmbeddingView,
    keyContext: 'cluster',
    // The overlay is offered only where `/methods` reports the projection.
    available: () => isEmbeddingVizAvailable(strategiesStore.methods.overlays),
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
