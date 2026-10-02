/**
 * `/settings` VLM axis (W9), mounted against stubbed `/methods`,
 * `/settings` and registry reads: the dropdown lists the served `vlm`
 * entries with their served status label and warning; an external entry
 * with no recorded acknowledgement is disabled and points at Settings →
 * Models; a PUT the server refuses (`vlm_external_not_acknowledged`) shows
 * the served message and the Models link without blanking the page, and
 * `unknown_vlm` shows the requested and valid ids; the Models card appears
 * only when the registry is served; the served `axes[]` copy replaces the
 * axis label.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import SettingsPage from './+page.svelte';
import { curationSettingsStore } from '$stores/curationSettings.svelte';
import { strategiesStore } from '$stores/strategies.svelte';
import { vlmAvailability } from '$lib/vlm/vlmAvailability.svelte';
import { packsAvailability } from '$lib/packs/packsAvailability.svelte';
import { profilesAvailability } from '$lib/profiles/profilesAvailability.svelte';
import { listFixture, EXTERNAL_WARNING } from '$lib/test/fixtures/vlm';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

const METHODS = {
  strategies: [
    {
      id: 'local_vlm',
      axis: 'vlm',
      label: 'Local VLM',
      status: 'stable',
      default: true,
      settable: true,
      endpoint_status: 'ready',
      endpoint_status_label: 'Ready',
      sends_images_externally: false,
      default_ack_recorded: null,
    },
    {
      id: 'cloud_vlm',
      axis: 'vlm',
      label: 'Cloud VLM',
      status: 'stable',
      settable: true,
      endpoint_status: 'unprobed',
      endpoint_status_label: 'Not probed yet',
      sends_images_externally: true,
      warning: EXTERNAL_WARNING,
      default_ack_recorded: false,
    },
    {
      id: 'cloud_ack',
      axis: 'vlm',
      label: 'Cloud acknowledged',
      status: 'stable',
      settable: true,
      endpoint_status_label: 'Ready',
      sends_images_externally: true,
      warning: EXTERNAL_WARNING,
      default_ack_recorded: true,
    },
    { id: 'off', axis: 'vlm', label: 'Off', status: 'stable', settable: true },
  ],
  flags: {},
  axes: [{ axis: 'vlm', label: 'Served VLM label', description: 'Served VLM words.' }],
};

let puts: { body: unknown }[];
let putResponse: () => Response;
let registryServed: boolean;
let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

const q = (id: string) => target.querySelector<HTMLElement>(`[data-testid="${id}"]`);

beforeEach(() => {
  puts = [];
  registryServed = true;
  putResponse = () =>
    json({ defaults: { vlm: 'cloud_ack' }, updated_at: null, updated_by: null });
  for (const s of [vlmAvailability, packsAvailability, profilesAvailability]) s.reset();
  curationSettingsStore.reset();
  strategiesStore.reset();
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      const method = init.method ?? 'GET';
      if (u.endsWith('/methods')) return json(METHODS);
      if (u.endsWith('/settings') && method === 'PUT') {
        puts.push({ body: JSON.parse(String(init.body)) });
        return putResponse();
      }
      if (u.endsWith('/settings')) {
        return json({ defaults: {}, updated_at: null, updated_by: null });
      }
      if (u.endsWith('/vlm/endpoints')) {
        return registryServed ? json(listFixture()) : json({ detail: 'Not Found' }, 404);
      }
      return json({ detail: 'Not Found' }, 404);
    }),
  );
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
  document.querySelectorAll('[role="dialog"]').forEach((d) => d.remove());
  vi.unstubAllGlobals();
  for (const s of [vlmAvailability, packsAvailability, profilesAvailability]) s.reset();
});

async function render(): Promise<void> {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SettingsPage, { target });
  flushSync();
  await vi.waitFor(() => expect(target.textContent).toContain('Served VLM label'));
  await vi.waitFor(() => expect(vlmAvailability.available).not.toBeNull());
  flushSync();
}

const vlmSelect = () =>
  [...target.querySelectorAll('select')].find((s) =>
    [...s.options].some((o) => o.value === 'cloud_vlm'),
  )!;

describe('/settings VLM dropdown', () => {
  it('lists the served entries with the served status and warning; off is an entry', async () => {
    await render();
    const opts = [...vlmSelect().options].map((o) => [
      o.value,
      o.textContent?.replace(/\s+/g, ' ').trim(),
    ]);
    expect(opts).toEqual([
      ['local_vlm', 'Local VLM · Ready'],
      [
        'cloud_vlm',
        `Cloud VLM · Not probed yet · warning: ${EXTERNAL_WARNING} · activate it on Settings → Models first`,
      ],
      ['cloud_ack', `Cloud acknowledged · Ready · warning: ${EXTERNAL_WARNING}`],
      ['off', 'Off'],
    ]);
  });

  it('disables only the external entry with no recorded acknowledgement, and links to Models', async () => {
    await render();
    const by = (v: string) => [...vlmSelect().options].find((o) => o.value === v)!;
    expect(by('cloud_vlm').disabled).toBe(true);
    expect(by('cloud_ack').disabled).toBe(false);
    expect(by('local_vlm').disabled).toBe(false);
    expect(by('off').disabled).toBe(false);
    const hint = q('settings-ack-hint')!;
    expect(hint.querySelector('a')?.getAttribute('href')).toMatch(/\/settings\/models$/);
  });

  it('uses the served axes[] label', async () => {
    await render();
    expect(target.textContent).toContain('Served VLM label');
    expect(target.textContent).toContain('Served VLM words.');
  });

  it('a refused PUT shows the served message and the Models link, and the page stays', async () => {
    putResponse = () =>
      json(
        {
          detail: {
            error: 'vlm_external_not_acknowledged',
            message: 'cloud_ack must be acknowledged on Settings → Models.',
            endpoint: 'cloud_ack',
            activate_via: 'settings/models',
          },
        },
        422,
      );
    await render();
    const select = vlmSelect();
    select.value = 'cloud_ack';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    const save = [...target.querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'Save' && !b.disabled,
    )!;
    save.click();
    flushSync();
    [...document.querySelectorAll('[role="dialog"] button')]
      .find((b) => b.textContent?.trim() === 'Confirm')!
      .dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.waitFor(() => expect(q('settings-save-error')).not.toBeNull());
    expect(puts[0]!.body).toEqual({ defaults: { vlm: 'cloud_ack' } });
    expect(q('settings-save-error')?.textContent).toContain(
      'cloud_ack must be acknowledged on Settings → Models.',
    );
    expect(q('settings-save-error')?.querySelector('a')?.getAttribute('href')).toMatch(
      /\/settings\/models$/,
    );
    // The control is still there to pick something else.
    expect(vlmSelect()).toBeDefined();
  });

  it('a non-422 refusal (409 vlm_endpoint_unavailable) shows the served message and the page stays', async () => {
    putResponse = () =>
      json(
        {
          detail: {
            error: 'vlm_endpoint_unavailable',
            message: 'local_vlm is unreachable right now.',
          },
        },
        409,
      );
    await render();
    const select = vlmSelect();
    select.value = 'off';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    [...target.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Save' && !b.disabled)!
      .click();
    flushSync();
    [...document.querySelectorAll('[role="dialog"] button')]
      .find((b) => b.textContent?.trim() === 'Confirm')!
      .dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.waitFor(() => expect(q('settings-save-error')).not.toBeNull());
    expect(q('settings-save-error')?.textContent).toContain(
      'local_vlm is unreachable right now.',
    );
    // Not the load-error replacement: the control and its Save are still there.
    expect(target.textContent).not.toContain('Retry');
    expect(vlmSelect()).toBeDefined();
  });

  it('unknown_vlm names the requested and valid ids', async () => {
    putResponse = () =>
      json(
        {
          detail: {
            error: 'unknown_vlm',
            message: 'Unknown VLM.',
            axis: 'vlm',
            requested: 'gone_vlm',
            valid_ids: ['local_vlm', 'off'],
          },
        },
        422,
      );
    await render();
    const select = vlmSelect();
    select.value = 'off';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    [...target.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Save' && !b.disabled)!
      .click();
    flushSync();
    [...document.querySelectorAll('[role="dialog"] button')]
      .find((b) => b.textContent?.trim() === 'Confirm')!
      .dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.waitFor(() => expect(q('settings-save-error')).not.toBeNull());
    expect(q('settings-save-error')?.textContent).toContain(
      'Unknown vlm "gone_vlm" — valid: local_vlm, off.',
    );
    expect(q('settings-save-error')?.querySelector('a')).toBeNull();
  });
});

describe('/settings Models card', () => {
  it('links to Settings → Models when the registry is served', async () => {
    await render();
    const card = q('vlm-models-card')!;
    expect(card.querySelector('a')?.getAttribute('href')).toMatch(/\/settings\/models$/);
  });

  it('is absent, not disabled, when the registry is not served', async () => {
    registryServed = false;
    await render();
    expect(q('vlm-models-card')).toBeNull();
  });
});
