/**
 * What a region route's 409 `no_active_profile` (`{detail: {error, message}}`)
 * becomes in the UI. Region features are hidden whenever `/health.region_profile` is
 * null, so this only happens when the backend drops its profile
 * mid-session. It is never a crash and never a raw "API 409 …" toast:
 * `apiFetch` throws `RegionProfileUnavailableError` (whose message is
 * `REGION_PROFILE_UNAVAILABLE_MESSAGE`), `toastStore` drops any toast
 * that carries that message, and the region-profile store shows one
 * "reload to apply" notice instead.
 *
 * Kept free of imports so both `api.ts` and the toast store can use it.
 */

/** The `detail.error` code of a region route's 409 with no active profile. */
export const NO_ACTIVE_PROFILE_ERROR = 'no_active_profile';

export const REGION_PROFILE_UNAVAILABLE_MESSAGE =
  'Region features are unavailable: the backend has no region profile configured';

/** `detail` is the structured body's `detail.error` code (`apiFetch` reads it
 *  from the body itself; `ApiError.detail` carries the served message). */
export function isNoRegionProfileDetail(detail: string | null | undefined): boolean {
  return detail === NO_ACTIVE_PROFILE_ERROR;
}

export function mentionsRegionProfileUnavailable(text: string): boolean {
  return text.includes(REGION_PROFILE_UNAVAILABLE_MESSAGE);
}

let listener: (() => void) | null = null;

/** Registered once by the region-profile store. */
export function setRegionProfileUnavailableListener(cb: (() => void) | null): void {
  listener = cb;
}

export function notifyRegionProfileUnavailable(): void {
  listener?.();
}
