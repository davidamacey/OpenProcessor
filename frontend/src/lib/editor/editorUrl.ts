/**
 * The image editor's links. The editor keeps the Workspace's query string as it came, so the
 * list the user walks (filter, queue, scope, sort) is the one they opened the image from, and the
 * way back lands on the same Workspace view.
 */
import { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';
import { projectHref } from '$lib/projectPaths';
import { buildItemQuery } from '$lib/workspace/itemQuery';
import { parseWorkspaceUrl } from '$lib/workspace/workspaceUrl';
import { imageListQuery } from '$lib/images/imageQuery';
import type { ImageViewParams } from '$lib/types_images';
import type { ItemFilterQuery } from '$lib/types_itemFilter';

function withSearch(search: string): string {
  const s = search.replace(/^\?/, '');
  return s ? `?${s}` : '';
}

/** `/p/<slug>/images/<id>` carrying the Workspace query. */
export function editorHref(imageId: string, search: string): string {
  return projectHref(`/images/${encodeURIComponent(imageId)}${withSearch(search)}`);
}

/** The Workspace the editor was opened from. */
export function workspaceBackHref(search: string): string {
  return projectHref(`/workspace${withSearch(search)}`);
}

/** The gallery query of the carried Workspace state: the filter, queue and scope, and the gallery sort. */
export function editorListQuery(
  params: URLSearchParams,
): ItemFilterQuery & ImageViewParams {
  const applied = new ItemFilterState();
  applied.fromUrl(params);
  const url = parseWorkspaceUrl(params);
  return imageListQuery(buildItemQuery(applied, url), {
    unlabeledOnly: false,
    sort: url.sort.gallery ?? null,
  });
}
