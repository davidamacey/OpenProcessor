/**
 * What the details panel shows for the focused thing. Each view lane appends
 * the kinds it makes focusable.
 */
import type { Component } from 'svelte';
import ClusterDetails from './details/ClusterDetails.svelte';
import ImageDetails from './details/ImageDetails.svelte';
import ItemDetails from './details/ItemDetails.svelte';
import SelectionDetails from './details/SelectionDetails.svelte';
import type { WorkspaceController } from './workspaceController.svelte';

export type DetailsKind = 'item' | 'image' | 'cluster' | 'selection';

export const DETAILS: Record<
  DetailsKind,
  Component<{ ws: WorkspaceController; id: string | null }>
> = {
  item: ItemDetails,
  image: ImageDetails,
  cluster: ClusterDetails,
  selection: SelectionDetails,
};
