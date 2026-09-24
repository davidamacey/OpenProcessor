/**
 * Loader tests (§4.3's branch table) plus the propagation proof that ESM
 * live bindings actually carry a tier-2 install through to every derived
 * module-level value — the one assumption §1.5 of the tier-2 plan leans
 * on.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  fetchDeploymentProfiles,
  loadDeploymentProfiles,
  resetDeploymentProfileLoad,
  DEPLOYMENT_PROFILE_TIMEOUT_MS,
} from './deploymentProfiles';
import {
  registeredSlots,
  slotRegistry,
  resetDeploymentSlots,
  builtinSlots,
} from './registeredSlots';
import { REVIEW_TABS, tabFromUrlId, endpointForTab } from '../reviewTabs';
import { LIMITS } from './config/allowLists';

const here = path.dirname(fileURLToPath(import.meta.url));
const exampleDoc = JSON.parse(
  readFileSync(
    path.resolve(here, '../../../static/annotation-profiles.example.json'),
    'utf-8',
  ),
);

function jsonRes(body: string, status = 200, type = 'application/json'): Response {
  return new Response(body, { status, headers: { 'content-type': type } });
}

function fakeFetchServing(body: unknown, status = 200, type = 'application/json') {
  return vi.fn(async () => jsonRes(JSON.stringify(body), status, type));
}

afterEach(() => {
  resetDeploymentProfileLoad();
  resetDeploymentSlots();
});

describe('fetchDeploymentProfiles — branch table', () => {
  it('1. 404 is silent', async () => {
    const impl = vi.fn(async () => new Response('', { status: 404 }));
    const r = await fetchDeploymentProfiles(impl as unknown as typeof fetch);
    expect(r).toEqual({ slots: [], warnings: [] });
  });

  it('2. an absent file served as the SPA fallback (200, text/html) is silent, not a warning', async () => {
    const impl = vi.fn(
      async () =>
        new Response('<html></html>', {
          status: 200,
          headers: { 'content-type': 'text/html' },
        }),
    );
    const r = await fetchDeploymentProfiles(impl as unknown as typeof fetch);
    expect(r).toEqual({ slots: [], warnings: [] });
  });

  it('3. 500 with application/json produces one warning, no slots', async () => {
    const impl = vi.fn(async () => jsonRes('{}', 500));
    const r = await fetchDeploymentProfiles(impl as unknown as typeof fetch);
    expect(r.slots).toEqual([]);
    expect(r.warnings).toHaveLength(1);
  });

  it('4. network rejection does not throw, no warning', async () => {
    const impl = vi.fn(async () => {
      throw new Error('network down');
    });
    const r = await fetchDeploymentProfiles(impl as unknown as typeof fetch);
    expect(r).toEqual({ slots: [], warnings: [] });
  });

  it('5. AbortError (timeout) does not throw, no warning', async () => {
    const impl = vi.fn(async () => {
      const e = new Error('aborted');
      e.name = 'AbortError';
      throw e;
    });
    const r = await fetchDeploymentProfiles(impl as unknown as typeof fetch);
    expect(r).toEqual({ slots: [], warnings: [] });
  });

  it('6. malformed JSON body names "not valid JSON"', async () => {
    const impl = vi.fn(async () => jsonRes('{'));
    const r = await fetchDeploymentProfiles(impl as unknown as typeof fetch);
    expect(r.slots).toEqual([]);
    expect(r.warnings[0]).toMatch(/not valid JSON/);
  });

  it('7. an oversize body names the size cap', async () => {
    const big = JSON.stringify({
      version: 1,
      slots: [],
      pad: 'x'.repeat(LIMITS.documentBytes + 1),
    });
    const impl = vi.fn(async () => jsonRes(big));
    const r = await fetchDeploymentProfiles(impl as unknown as typeof fetch);
    expect(r.slots).toEqual([]);
    expect(r.warnings[0]).toMatch(/256 KB/);
  });

  it('8. an empty body is silent', async () => {
    const impl = vi.fn(async () => jsonRes(''));
    const r = await fetchDeploymentProfiles(impl as unknown as typeof fetch);
    expect(r).toEqual({ slots: [], warnings: [] });
  });

  it('9. the request carries redirect:error, cache:no-store, accept:application/json, and a signal', async () => {
    let seenInit: RequestInit | undefined;
    const impl = vi.fn(async (_url: string, init?: RequestInit) => {
      seenInit = init;
      return jsonRes('{"version":1,"slots":[]}');
    });
    await fetchDeploymentProfiles(impl as unknown as typeof fetch);
    expect(seenInit?.redirect).toBe('error');
    expect(seenInit?.cache).toBe('no-store');
    expect((seenInit?.headers as Record<string, string>)?.accept).toBe(
      'application/json',
    );
    expect(seenInit?.signal).toBeDefined();
  });
});

describe('loadDeploymentProfiles — propagation and memoization', () => {
  it('10. installing a deployment profile updates every live binding, including REVIEW_TABS', async () => {
    expect(REVIEW_TABS.map((t) => t.id)).not.toContain('slot:pallet_label');
    await loadDeploymentProfiles(fakeFetchServing(exampleDoc));
    expect(registeredSlots.map((s) => s.key)).toEqual(['license_plate', 'pallet_label']);
    expect(slotRegistry.byKey('pallet_label')).toBeDefined();
    expect(REVIEW_TABS.map((t) => t.id)).toContain('slot:pallet_label');
    // 6 core tabs (2026-09-24 adds new_class_proposals) + license_plate +
    // the newly installed pallet_label slot tab.
    expect(REVIEW_TABS).toHaveLength(7);
    expect(tabFromUrlId('pallet_labels')).toBe('slot:pallet_label');
    expect(tabFromUrlId('plates')).toBe('slot:license_plate');
    expect(endpointForTab('slot:pallet_label')).toBe('pallet_labels');
  });

  it('11. calling loadDeploymentProfiles twice performs exactly one fetch', async () => {
    const impl = fakeFetchServing({ version: 1, slots: [] });
    await loadDeploymentProfiles(impl as unknown as typeof fetch);
    await loadDeploymentProfiles(impl as unknown as typeof fetch);
    expect(impl).toHaveBeenCalledTimes(1);
  });

  it('12. an empty {version:1,slots:[]} document leaves registeredSlots identical by reference to builtinSlots', async () => {
    await loadDeploymentProfiles(fakeFetchServing({ version: 1, slots: [] }));
    expect(registeredSlots).toBe(builtinSlots);
  });
});

describe('DEPLOYMENT_PROFILE_TIMEOUT_MS', () => {
  it('is 2 seconds', () => {
    expect(DEPLOYMENT_PROFILE_TIMEOUT_MS).toBe(2000);
  });
});
