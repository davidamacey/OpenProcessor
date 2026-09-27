/**
 * Tier-2 deployment profile loading (contract §2 tier 2, §9 step 1).
 *
 * Fetches `/annotation-profiles.json` — a file the DEPLOYMENT supplies,
 * either by dropping it in `static/` before a build or by bind-mounting
 * it over `/usr/share/nginx/html/annotation-profiles.json` in the
 * running container — parses it through the hardened tier-2 parser, and
 * installs the result into the slot registry before first render.
 *
 * ABSENT IS THE NORMAL CASE AND MUST BE SILENT. nginx's SPA fallback
 * (`try_files $uri $uri/ /index.html`, nginx.conf:64) answers a missing
 * file with index.html and HTTP 200 — not a 404 — and Vite's dev server
 * does the same. So "no deployment config" is detected by CONTENT TYPE,
 * not status, and produces zero warnings: every stock Cropwright
 * deployment takes this path on every page load.
 *
 * This function never throws and never rejects: a missing or broken
 * optional deployment file degrades to the built-in behavior, it never
 * becomes an error surface the operator has to dismiss.
 */

import type { SlotSpec } from './types';
import { LIMITS } from './config/allowLists';
import { parseProfileDocument } from './config/parseSlotConfig';
import { installDeploymentSlots } from './registeredSlots';

export const DEPLOYMENT_PROFILE_URL = '/annotation-profiles.json';
export const DEPLOYMENT_PROFILE_TIMEOUT_MS = 2000;

export interface DeploymentProfileLoad {
  slots: SlotSpec[];
  warnings: string[];
}

const EMPTY: DeploymentProfileLoad = { slots: [], warnings: [] };

/** Pure-ish: takes its fetch so tests drive every branch with no network
 *  and no global patching. */
export async function fetchDeploymentProfiles(
  fetchImpl?: typeof fetch,
  url: string = DEPLOYMENT_PROFILE_URL,
): Promise<DeploymentProfileLoad> {
  const impl = fetchImpl ?? (typeof fetch === 'function' ? fetch : undefined);
  if (!impl) return EMPTY;

  let res: Response;
  try {
    res = await impl(url, {
      cache: 'no-store',
      redirect: 'error',
      headers: { accept: 'application/json' },
      signal: AbortSignal.timeout(DEPLOYMENT_PROFILE_TIMEOUT_MS),
    });
  } catch {
    // Network error, abort/timeout, redirect rejection, etc. — a file
    // server that is not answering is indistinguishable from no file at
    // all, and this runs on every page load, so it must be silent.
    return EMPTY;
  }

  if (res.status === 404) return EMPTY;

  const contentType = res.headers.get('content-type') ?? '';
  if (!contentType.includes('application/json')) {
    // The nginx / Vite SPA-fallback "absent" case (§1.6) — silent.
    return EMPTY;
  }

  if (!res.ok) {
    return {
      slots: [],
      warnings: [`deployment profile request failed: HTTP ${res.status}`],
    };
  }

  const lengthHeader = res.headers.get('content-length');
  if (lengthHeader != null && Number(lengthHeader) > LIMITS.documentBytes) {
    return {
      slots: [],
      warnings: [
        `deployment profile exceeds the 256 KB deployment-profile size cap — ignored`,
      ],
    };
  }

  let body: string;
  try {
    body = await res.text();
  } catch {
    return EMPTY;
  }

  if (body.length > LIMITS.documentBytes) {
    return {
      slots: [],
      warnings: [
        `deployment profile exceeds the 256 KB deployment-profile size cap — ignored`,
      ],
    };
  }

  if (body.trim().length === 0) return EMPTY;

  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch (e) {
    return {
      slots: [],
      warnings: [`deployment profile is not valid JSON: ${(e as Error).message}`],
    };
  }

  return parseProfileDocument(parsed);
}

let inflight: Promise<void> | null = null;
let settled = false;

/** Boot entry point. Idempotent and memoized — client-side navigations
 *  re-run `load()` and must not re-fetch. */
export async function loadDeploymentProfiles(fetchImpl?: typeof fetch): Promise<void> {
  if (settled) return;
  if (inflight) return inflight;
  inflight = (async () => {
    const { slots, warnings } = await fetchDeploymentProfiles(fetchImpl);
    if (slots.length > 0 || warnings.length > 0) {
      installDeploymentSlots(slots, warnings);
    }
    for (const w of warnings) {
      console.warn(`[annotation-profiles] ${w}`);
    }
  })();
  try {
    await inflight;
  } finally {
    settled = true;
    inflight = null;
  }
}

/** Test-only: clears the memo so a later test can drive a different
 *  branch in the same file. */
export function resetDeploymentProfileLoad(): void {
  settled = false;
  inflight = null;
}
