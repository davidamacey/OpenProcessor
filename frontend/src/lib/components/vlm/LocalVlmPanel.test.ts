/**
 * The local-model panel: not configured = the served reason only; a table
 * of served facts (null = "—"), serving / requested chips, Switch on the
 * entries not being served behind a confirm; "Select anyway" only after a
 * served `vlm_catalog_does_not_fit` and then Switch resends with force; a
 * restart banner with the served reason and the copyable command, and a
 * confirm-gated "Cancel the request". Nothing says the model switched.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { catalogFixture, localFixture } from '$lib/test/fixtures/vlm';
import type { VlmCatalogResponse } from '$lib/types_vlm';
import LocalVlmPanel from './LocalVlmPanel.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

function render(
  catalog: VlmCatalogResponse = catalogFixture(),
  over: Record<string, unknown> = {},
) {
  const props = {
    catalog,
    busy: false,
    error: null,
    errorCode: null,
    onselect: vi.fn().mockResolvedValue(true),
    onclear: vi.fn().mockResolvedValue(true),
    ...over,
  };
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(LocalVlmPanel, { target, props });
  flushSync();
  return props;
}

const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);
const row = (id: string) =>
  target.querySelector<HTMLElement>(`[data-testid="local-vlm-row"][data-id="${id}"]`)!;

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
  document.querySelectorAll('[role="dialog"]').forEach((d) => d.remove());
});

describe('LocalVlmPanel', () => {
  it('not configured: only the served reason, no table', () => {
    render(
      catalogFixture({
        local: localFixture({
          configured: false,
          reason: 'No local model server is set up.',
        }),
      }),
    );
    expect(q('local-vlm-reason')?.textContent).toBe('No local model server is set up.');
    expect(q('local-vlm-table')).toBeNull();
    expect(target.textContent).not.toContain('Switch');
  });

  it('prints the served facts, unknowns as "—", and the served status label', () => {
    render();
    const a = row('vision-7b').textContent!;
    expect(a).toContain('Vision 7B');
    expect(a).toContain('Apache-2.0');
    expect(a).toContain('7B fp8');
    expect(a).toContain('Tested');
    // text_reading_verified is null: unknown, never "no".
    expect(row('vision-7b').querySelectorAll('td')[10]!.textContent).toBe('—');
    expect(row('vision-30b').textContent).toContain('To verify');
    expect(row('vision-30b').textContent).toContain('gated');
    // fits is false: a served "no", distinct from unknown.
    expect(row('vision-30b').querySelectorAll('td')[8]!.textContent).toBe('no');
    expect(row('vision-30b').querySelectorAll('td')[9]!.textContent).toBe('—');
  });

  it('links the license only for an http(s) URL', () => {
    render();
    expect(row('vision-7b').querySelector('a')?.getAttribute('href')).toBe(
      'https://example.com/license/apache',
    );
    const cat = catalogFixture();
    cat.entries[0]!.license_url = 'javascript:alert(1)';
    unmount(instance!);
    target.remove();
    render(cat);
    expect(row('vision-7b').querySelector('a')).toBeNull();
    expect(row('vision-7b').textContent).toContain('Apache-2.0');
  });

  it('serving and requested chips come from the served flags; Switch is not offered for the serving row', () => {
    const cat = catalogFixture();
    cat.entries[1]!.desired = true;
    render(cat);
    expect(
      row('vision-7b').querySelector('[data-testid="local-vlm-serving"]'),
    ).not.toBeNull();
    expect(row('vision-7b').querySelector('[data-testid="local-vlm-switch"]')).toBeNull();
    expect(
      row('vision-30b').querySelector('[data-testid="local-vlm-desired"]'),
    ).not.toBeNull();
    expect(
      row('vision-30b').querySelector('[data-testid="local-vlm-switch"]'),
    ).not.toBeNull();
  });

  it('Switch is confirm-gated and sends the entry id without force', async () => {
    const p = render();
    row('vision-30b')
      .querySelector<HTMLElement>('[data-testid="local-vlm-switch"]')!
      .click();
    flushSync();
    expect(p.onselect).not.toHaveBeenCalled();
    expect(q('local-vlm-force')).toBeNull();
    [...document.querySelectorAll('[role="dialog"] button')]
      .find((b) => b.textContent?.trim() === 'Switch')!
      .dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.waitFor(() => expect(p.onselect).toHaveBeenCalledWith('vision-30b', false));
  });

  it('after a served does-not-fit refusal, "Select anyway" resends with force', async () => {
    const p = render(catalogFixture(), {
      error: 'vision-30b needs 80 GB; this GPU has 48.',
      errorCode: 'vlm_catalog_does_not_fit',
    });
    row('vision-30b')
      .querySelector<HTMLElement>('[data-testid="local-vlm-switch"]')!
      .click();
    flushSync();
    expect(q('local-vlm-switch-error')?.textContent).toBe(
      'vision-30b needs 80 GB; this GPU has 48.',
    );
    q('local-vlm-force')!.click();
    flushSync();
    [...document.querySelectorAll('[role="dialog"] button')]
      .find((b) => b.textContent?.trim() === 'Select anyway')!
      .dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.waitFor(() => expect(p.onselect).toHaveBeenCalledWith('vision-30b', true));
  });

  it('other served refusals offer no force', () => {
    render(catalogFixture(), {
      error: 'Unknown catalog id.',
      errorCode: 'unknown_catalog_id',
    });
    row('vision-30b')
      .querySelector<HTMLElement>('[data-testid="local-vlm-switch"]')!
      .click();
    flushSync();
    expect(q('local-vlm-force')).toBeNull();
  });

  it('a pending restart shows the served reason and the copyable command, never "switched"', async () => {
    const cat = catalogFixture({
      local: localFixture({
        restart_required: true,
        poll_after_s: 5,
        reason: 'A restart is needed to serve vision-30b.',
        desired: {
          catalog_id: 'vision-30b',
          requested_at: null,
          command: 'docker compose up -d vlm',
        },
      }),
    });
    cat.entries[1]!.desired = true;
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });
    render(cat);
    const banner = q('local-vlm-restart')!;
    expect(banner.textContent).toContain('A restart is needed to serve vision-30b.');
    expect(q('local-vlm-command')?.textContent).toBe('docker compose up -d vlm');
    q('local-vlm-copy')!.click();
    await vi.waitFor(() =>
      expect(writeText).toHaveBeenCalledWith('docker compose up -d vlm'),
    );
    expect(target.textContent!.toLowerCase()).not.toContain('switched');
    // The serving flag is still the served one.
    expect(
      row('vision-30b').querySelector('[data-testid="local-vlm-serving"]'),
    ).toBeNull();
  });

  it('Cancel the request is confirm-gated', async () => {
    const p = render(
      catalogFixture({
        local: localFixture({
          restart_required: true,
          desired: { catalog_id: 'vision-30b', requested_at: null, command: 'x' },
        }),
      }),
    );
    q('local-vlm-cancel')!.click();
    flushSync();
    expect(p.onclear).not.toHaveBeenCalled();
    [...document.querySelectorAll('[role="dialog"] button')]
      .filter((b) => b.textContent?.trim() === 'Cancel the request')
      .at(-1)!
      .dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.waitFor(() => expect(p.onclear).toHaveBeenCalledTimes(1));
  });

  it('no restart banner when none is required', () => {
    render();
    expect(q('local-vlm-restart')).toBeNull();
  });
});
