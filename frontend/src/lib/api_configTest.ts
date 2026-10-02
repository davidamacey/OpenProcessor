/**
 * Backend wrappers for W5 test-on-crop (owner: Track C): pack and
 * region-profile tests. Neither route writes. Import from
 * `$lib/api_configTest` directly; never re-exported from `api.ts` (that
 * would make the two modules circular).
 */
import { apiFetch, mapRawCrop, scoped } from '$lib/api';
import type { RawCrop } from '$lib/api';
import type { Crop } from '$lib/types';
import type {
  PackTestRequest,
  PackTestResponse,
  RegionTestRequest,
  RegionTestResponse,
} from '$lib/types_configTest';

/** `POST /prompt_packs/test`: runs one call on real crops. Each result's
 *  `preview_item` (the item as the write would leave it) is also mapped
 *  into `preview`; the raw doc stays for the JSON view. */
export async function testPromptPack(
  body: PackTestRequest,
  signal?: AbortSignal,
): Promise<PackTestResponse<Crop>> {
  const res = await apiFetch<PackTestResponse>(
    `${scoped()}/prompt_packs/test`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
  return {
    ...res,
    results: (res.results ?? []).map((r) => ({
      ...r,
      preview: r.preview_item ? mapRawCrop(r.preview_item as unknown as RawCrop) : null,
    })),
  };
}

/** `POST /region_profiles/test`: runs a profile's legs over one stored
 *  crop. `preview_item` is mapped into `preview`. */
export async function testRegionProfile(
  body: RegionTestRequest,
  signal?: AbortSignal,
): Promise<RegionTestResponse> {
  const res = await apiFetch<Omit<RegionTestResponse, 'preview'>>(
    `${scoped()}/region_profiles/test`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
  return {
    ...res,
    legs: res.legs ?? [],
    preview: mapRawCrop(res.preview_item as unknown as RawCrop),
  };
}
